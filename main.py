import json
import numpy as np
import torch
from model import BertModel, TransformerModel, GPTModel
from model import QModelContainer
from plot import Plotter
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from dataset import QDatasetForHuggingFaceModels
from metric import QMetric
from transformers import BertTokenizer


def _split_data(generated, qmc):
    original_from_generated = QTimeSeries(TimeSeries(generated.ts.x[:qmc._global_window_length],
                                                generated.ts.y[:qmc._global_window_length],
                                                generated.id()),
                                    generated.tokens[:qmc._global_window_length],
                                    generated.tokens_y[:qmc._global_window_length])
    generated.tokens = generated.tokens[qmc._global_window_length:]
    generated.ts.x = np.array([generated.ts.x[0] + qmc._global_window_length])
    generated.tokens_y = generated.tokens_y[qmc._global_window_length:]
    return original_from_generated, generated


def _get_original_ts(qds,ts_id):
    """
    Return original ts without first window
    """
    unbatched_ts = qds.get_unbatched(ts_id, _quantized=True)
    return QTimeSeries(ts=qds.get_unbatched(ts_id),
                       bin_idx=np.asarray(unbatched_ts.tokens[qmc._global_window_length:]),
                       bin_val=np.asarray(unbatched_ts.tokens_y[qmc._global_window_length:]))
    

def _plot(ts,train_qds, test_qds, eval_qds,
          train_generated, train_org,
          eval_generated, eval_org,
          test_generated, test_org):
    plot = Plotter(TimeSeriesQuantizer(), "plots/")
    plot.plot(ts[i], label="true")
    plot.plot(train_qds.get_unbatched(f"ts{i}"), label="train")
    plot.plot(QDatasetForHuggingFaceModels(ts, split="eval", batch=False).raw_data[i].ts, label="eval") #error with eval_qds
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
    x = np.arange(366)
    ts  = [TimeSeries(
                    x,
                    np.loadtxt(f"./data/basic_cyclic/cyclic{i}.csv",delimiter =","),
                    f"ts{i}"
            ) for i in range(10)
    ]
    train_qds = QDatasetForHuggingFaceModels(ts, split="train", batch=True)
    eval_qds = QDatasetForHuggingFaceModels(ts, split="eval", batch=True)
    test_qds = QDatasetForHuggingFaceModels(ts, split="test", batch=True)
    
    quant = TimeSeriesQuantizer()
    trans = GPTModel(quant.special_tokens, len(quant.bins_indices))
    qmc = QModelContainer(trans, quant)
    
    qmc.train(train_qds, eval_qds)
    train_metric, eval_metric, test_metric = [],[],[]
    for i in range(len(ts)):
        train_org, train_generated = _split_data(qmc.generate(train_qds,
                                                   id=f"ts{i}",
                                                   horizon=int(train_qds.get_unbatched(f"ts{i}").length()-qmc._global_window_length)),
                                        qmc)
        eval_org, eval_generated = _split_data(qmc.generate(eval_qds,
                                                            id=f"ts{i}",
                                                            horizon=int(eval_qds.get_unbatched(f"ts{i}").length()-qmc._global_window_length)),
                                               qmc)
        test_org, test_generated = _split_data(qmc.generate(test_qds,
                                                            id=f"ts{i}",
                                                            horizon=int(test_qds.get_unbatched(f"ts{i}").length()-qmc._global_window_length)),
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
    exit()