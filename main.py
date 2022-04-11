import json
import numpy as np
import torch
from model import TransformerModel
from model import QModelContainer
from plot import Plotter
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from dataset import QDataset
from metric import QMetric
from transformers import BertTokenizer


def _split_data(generated, qmc):
    original_from_generated = QTimeSeries(TimeSeries(generated.ts.x[:qmc.window_length],
                                                generated.ts.y[:qmc.window_length],
                                                generated.id()),
                                    generated.tokens[:qmc.window_length],
                                    generated.tokens_y[:qmc.window_length])
    generated.tokens = generated.tokens[qmc.window_length:]
    generated.ts.x = np.array([generated.ts.x[0] + qmc.window_length])
    generated.tokens_y = generated.tokens_y[qmc.window_length:]
    return original_from_generated, generated


def _get_original_ts(qds,ts_id):
    """
    Return original ts without first window
    """
    unbatched_ts = qds.get_unbatched(ts_id, _quantized=True)
    return QTimeSeries(ts=qds.get_unbatched(ts_id),
                       bin_idx=np.asarray(unbatched_ts.tokens[qmc.window_length:]),
                       bin_val=np.asarray(unbatched_ts.tokens_y[qmc.window_length:]))
    

def _plot(ts,train_qds, test_qds, eval_qds,
          train_generated, train_org,
          eval_generated, eval_org,
          test_generated, test_org):
    plot = Plotter(TimeSeriesQuantizer(), "plots/")
    plot.plot(ts[i], label="true")
    plot.plot(train_qds.get_unbatched(f"ts{i}"), label="train")
    plot.plot(QDataset(ts, split="eval", batch=False).raw_data[i].ts, label="eval") #error with eval_qds
    plot.plot(test_qds.get_unbatched(f"ts{i}"),  label="test")
    plot.plot(train_org, mfc='blue', mec='blue', c="blue")
    plot.plot(train_generated)
    plot.plot(eval_org, mfc='blue', mec='blue', c="blue")
    plot.plot(eval_generated)
    plot.plot(test_org, mfc='blue', mec='blue', c="blue")
    plot.plot(test_generated)
    
    plot.save(f"ts{i}.png")
    
def _get_avg_metrics(metrics):
    return {
        'accuracy': np.array([m.accuracy for m in metrics]).mean(),
        'soft_accuracy': np.array([m.soft_accuracy for m in metrics]).mean(),
        'mae': np.array([m.mae for m in metrics]).mean()
    }
    
def _generate(model,y,id,horizon):
    (qts, time_series), _, _ = y.get_batched(id, _all=True)
    model.eval()
    full_time_series = time_series[1:-1].clone()
    _cls= torch.tensor([qmc.quantizer.special_tokens['cls']])
    mask = torch.tensor([qmc.quantizer.special_tokens['mask']])
    sep =  torch.tensor([qmc.quantizer.special_tokens['sep']])
    for i in range(horizon):
        if i >0:
            time_series = torch.cat((_cls,
                                    time_series,
                                    mask,
                                    sep))
        outputs = model(input_ids=time_series.unsqueeze(0))
        predicted_token = outputs.logits[0][-2]
        predicted_token = torch.nn.functional.softmax(predicted_token)
        predicted_token = torch.argmax(predicted_token)
        
        time_series = torch.cat((time_series[2:-2], predicted_token.unsqueeze(0)))
        full_time_series = torch.cat((full_time_series, predicted_token.unsqueeze(0)), dim=-1)
    full_time_series = full_time_series

    qts.tokens_y = torch.take(torch.from_numpy(qmc.quantizer.bins_values),full_time_series).numpy()
    qts.tokens = full_time_series.numpy()
    
    return qts

if __name__ == "__main__":
    trans = TransformerModel()
    quant = TimeSeriesQuantizer()
    qmc = QModelContainer(trans, quant)
    
    random_shifts = qmc.random_shifts
    print("Random shifts: ", random_shifts)
    soft_labels = qmc.soft_labels
    print("Soft labels: ", soft_labels)

    x = np.arange(366)
    ts  = [TimeSeries(
                    x,
                    np.loadtxt(f"./data/basic_linear/linear{i}.csv",delimiter =","),
                    f"ts{i}"
            ) for i in range(10)
    ]
    
    train_qds = QDataset(ts, split="train", batch=True, soft_labels=soft_labels)
    eval_qds = QDataset(ts, split="eval", batch=True, soft_labels=soft_labels)
    test_qds = QDataset(ts, split="test", batch=True)
    ###############################
    from torch.optim import AdamW
    from transformers import  BertForMaskedLM, BertConfig
    from torch.utils.data.dataloader import DataLoader
    config = BertConfig(vocab_size=len(qmc.quantizer.bins_indices),
                        max_length = 42, #magic number!!!
                        num_hidden_size=128,
                        num_hidden_layers=2,
                        mask_token_id=qmc.quantizer.special_tokens['mask'],
                        cls_token_id = qmc.quantizer.special_tokens['cls'],
                        sep_token_id=qmc.quantizer.special_tokens['sep'],
                        pad_token_id=qmc.quantizer.special_tokens['pad'])
    #tokenizer = BertTokenizer(config)
    model = BertForMaskedLM(config)
    
    optimizer = AdamW(model.parameters(), lr=5e-5)
    for epoch in range(100):
        for batch in DataLoader(train_qds, batch_size=qmc.batch_size, shuffle=qmc.shuffle):
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()

            optimizer.step()
            optimizer.zero_grad()
            
            # if epoch >= 1:
            #     logits = outputs.logits
            #     raw_y = torch.nn.functional.softmax(logits)
            #     tokens = torch.argmax(raw_y,dim=-1)
            #     print("CLS: ", raw_y[0][0])
            #     print("SEP: ", raw_y[0][-1])
            #     print("first token: ",raw_y[0][1])
            #     print("Masked tokens: ", batch['input_ids'][0])
            #     print("Labels: ", batch['labels'][0])
            #     print("Predict: ", tokens[0])
                
            #     print("Masked tokens: ", batch['input_ids'][1])
            #     print("Labels: ", batch['labels'][1])
            #     print("Predict: ", tokens[1])
                
    train_metric, eval_metric, test_metric = [],[],[]
    for i in range(len(ts)):
        train_org, train_generated = _split_data(_generate(model,
                                                           train_qds,
                                                           id=f"ts{i}",
                                                           horizon=int(train_qds.get_unbatched(f"ts{i}").length()-qmc.window_length)),
                                                qmc)
        eval_org, eval_generated = _split_data(_generate(model,
                                                         eval_qds,
                                                         id=f"ts{i}",
                                                         horizon=int(eval_qds.get_unbatched(f"ts{i}").length()-qmc.window_length)),
                                                qmc)
        test_org, test_generated = _split_data(_generate(model,
                                                         test_qds,
                                                         id=f"ts{i}",
                                                         horizon=int(test_qds.get_unbatched(f"ts{i}").length()-qmc.window_length)),
                                                qmc)
        _plot(ts,train_qds,test_qds,eval_qds,
              train_generated,train_org,
              eval_generated,eval_org,
              test_generated,test_org)
        
        train_true =_get_original_ts(train_qds,f"ts{i}")
        train_metric.append(QMetric(train_true, train_generated))
        eval_true =_get_original_ts(eval_qds,f"ts{i}")
        eval_metric.append(QMetric(eval_true, eval_generated))
        test_true =_get_original_ts(test_qds,f"ts{i}")
        test_metric.append(QMetric(test_true, test_generated))
        
        # print(f"TS: {i}")
        # print("Train split:", train_metric[i])
        # print("Eval split:", eval_metric[i])
        # print("Test split:", test_metric[i])
        # print("="*20)
    
    metric ={
        **{"Overall (avg on test splits)": _get_avg_metrics(test_metric)},
        **{ts_i: {
            "train": train_metric[i].__dict__,
            "eval": eval_metric[i].__dict__,
            "test": test_metric[i].__dict__
            } for ts_i in range(len(ts))}
    }
    with open('./plots/metrics.json', 'w') as f:
        json.dump(metric, f, indent=4)
         
    ############################
    exit()
    qmc.train(train_qds, eval_qds)
    
    train_metric, eval_metric, test_metric = [],[],[]
    for i in range(len(ts)):
        train_org, train_generated = _split_data(qmc.generate(train_qds,
                                                   id=f"ts{i}",
                                                   horizon=int(train_qds.get_unbatched(f"ts{i}").length()-(qmc.window_length-qmc.num_last_unmasked))),
                                        qmc)
        eval_org, eval_generated = _split_data(qmc.generate(eval_qds,
                                                            id=f"ts{i}",
                                                            horizon=int(eval_qds.get_unbatched(f"ts{i}").length()-(qmc.window_length-qmc.num_last_unmasked))),
                                               qmc)
        test_org, test_generated = _split_data(qmc.generate(test_qds,
                                                            id=f"ts{i}",
                                                            horizon=int(test_qds.get_unbatched(f"ts{i}").length()-(qmc.window_length-qmc.num_last_unmasked))),
                                               qmc)
        _plot(ts,train_qds,test_qds,eval_qds,
              train_generated,train_org,
              eval_generated,eval_org,
              test_generated,test_org)
        
        train_true =_get_original_ts(train_qds,f"ts{i}")
        train_metric.append(QMetric(train_true, train_generated))
        eval_true =_get_original_ts(eval_qds,f"ts{i}")
        eval_metric.append(QMetric(eval_true, eval_generated))
        test_true =_get_original_ts(test_qds,f"ts{i}")
        test_metric.append(QMetric(test_true, test_generated))
        
        # print(f"TS: {i}")
        # print("Train split:", train_metric[i])
        # print("Eval split:", eval_metric[i])
        # print("Test split:", test_metric[i])
        # print("="*20)
    
    metric ={
        **{"Overall (avg on test splits)": _get_avg_metrics(test_metric)},
        **{ts_i: {
            "train": train_metric[i].__dict__,
            "eval": eval_metric[i].__dict__,
            "test": test_metric[i].__dict__
            } for ts_i in range(len(ts))}
    }
    with open('./plots/metrics.json', 'w') as f:
        json.dump(metric, f, indent=4)
        
       
    #[{ts_i: metric.__dict__} for ts_i, metric in enumerate(train_metric)]
    # print("--------")
    # num_samples = 100
    # # all
    # train_random_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    # for _ in range(num_samples):
    #     train_random_plot.plot(qmc.generate(train_qds, 
    #                                         id="sin", 
    #                                         stochastic=True, 
    #                   ()                     horizon=int(train_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked))), 
    #                                         alpha=1/num_samples)
    # train_random_plot.save("random_train.png")
    # # eval
    # eval_random_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    # for _ in range(num_samples):
    #     eval_random_plot.plot(qmc.generate(eval_qds, 
    #                                        id="sin", 
    #                                        horizon=int(eval_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked)), 
    #                                        stochastic=True), 
    #                                        alpha=1/num_samples)
    # eval_random_plot.save("random_eval.png")

    # cos_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    # cos_plot.plot(tz, label="true")
    # cos_plot.plot(qmc.generate(QDataset(tz, split="train"), id="cos"))
    # cos_plot.save("cos_test.png")