import json
import numpy as np
from model import TransformerModel
from model import QModelContainer
from plot import Plotter
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from dataset import QDataset
from metric import QMetric


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
        'mae': np.array([m.soft_accuracy for m in metrics]).mean()
    }
    

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
                    np.loadtxt(f"./data/synthetic/synthetic{i}.csv",delimiter =","),
                    f"ts{i}"
            ) for i in range(10)
    ]
    
    all_qds = QDataset(ts, batch=True) 
    train_qds = QDataset(ts, split="train", batch=True, soft_labels=soft_labels)
    eval_qds = QDataset(ts, split="eval", batch=True, soft_labels=soft_labels)
    test_qds = QDataset(ts, split="test", batch=True)

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
        # print("============")
    
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