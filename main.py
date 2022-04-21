import os
import json
import numpy as np
from decorators import get_global_args
from _factory import QFactory, TransformerFactory, BertFactory, GPTFactory
from plot import Plotter
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from metric import QMetric

@get_global_args
def process_input(_global_input_dir, _global_model, _global_window_length, **kwargs) -> QFactory:
    ts = []
    for i,file in enumerate(os.listdir(_global_input_dir)):
        if file == 'plots' or file == "params.json":
            continue
        y = np.loadtxt(os.path.join(_global_input_dir, file), delimiter=",")
        x = np.arange(len(y))
        ts.append(TimeSeries(x, y, f"ts{i}")) 
    factories = {
        "transformer": TransformerFactory(ts),
        "bert": BertFactory(ts),
        "gpt": GPTFactory(ts)
    }
    return factories[_global_model],  _global_window_length


def main(factory: QFactory, window_length: int):
    train_qds = factory.get_dataset(split="train", batch=True)
    eval_qds = factory.get_dataset(split="eval", batch=True)
    test_qds = factory.get_dataset(split="test", batch=True)
    qmc = factory.get_container()
    
    qmc.train(train_qds, eval_qds)
    
    train_metric, eval_metric, test_metric = [],[],[]
    for i, ts in enumerate(train_qds.raw_unbatched_data):
        train_org, train_generated = _split_data(qmc.generate(train_qds,
                                                   id=ts.id(),
                                                   horizon=int(train_qds.get_unbatched(ts.id()).length()-window_length)),
                                        window_length)
        eval_org, eval_generated = _split_data(qmc.generate(eval_qds,
                                                            id=ts.id(),
                                                            horizon=int(eval_qds.get_unbatched(ts.id()).length()-window_length)),
                                               window_length)
        test_org, test_generated = _split_data(qmc.generate(test_qds,
                                                            id=ts.id(),
                                                            horizon=int(test_qds.get_unbatched(ts.id()).length()-window_length)),
                                              window_length)
        _plot(ts,train_qds,test_qds,eval_qds,
              train_generated,train_org,
              eval_generated,eval_org,
              test_generated,test_org)
        
        train_true = _get_original_ts(train_qds, ts.id(), window_length)
        train_metric.append(QMetric(train_true, train_generated))
        eval_true =_get_original_ts(eval_qds, ts.id(), window_length)
        eval_metric.append(QMetric(eval_true, eval_generated))
        test_true =_get_original_ts(test_qds, ts.id(), window_length)
        test_metric.append(QMetric(test_true, test_generated))    
    metric ={
        **{"Overall (avg on test splits)": _get_avg_metrics(test_metric)},
        **{ts_i: {
            "train": train_metric[i].__dict__,
            "eval": eval_metric[i].__dict__,
            "test": test_metric[i].__dict__
            } for ts_i in range(len(train_metric))}
    }
    with open('./plots/metrics.json', 'w') as f:
        json.dump(metric, f, indent=4)
    

def _split_data(generated, window_length):
    original_from_generated = QTimeSeries(TimeSeries(generated.ts.x[:window_length],
                                                generated.ts.y[:window_length],
                                                generated.id()),
                                    generated.tokens[:window_length],
                                    generated.tokens_y[:window_length])
    generated.tokens = generated.tokens[window_length:]
    generated.ts.x = np.array([generated.ts.x[0] + window_length])
    generated.tokens_y = generated.tokens_y[window_length:]
    return original_from_generated, generated


def _get_original_ts(qds, ts_id, window_length):
    """
    Return original ts without first window
    """
    unbatched_ts = qds.get_unbatched(ts_id, _quantized=True)
    return QTimeSeries(ts=qds.get_unbatched(ts_id),
                       bin_idx=np.asarray(unbatched_ts.tokens[window_length:]),
                       bin_val=np.asarray(unbatched_ts.tokens_y[window_length:]))
    

def _plot(ts,train_qds, test_qds, eval_qds,
          train_generated, train_org,
          eval_generated, eval_org,
          test_generated, test_org):
    plot = Plotter(TimeSeriesQuantizer(), "plots/")
    plot.plot(ts, label="true")
    plot.plot(train_qds.get_unbatched(ts.id()), label="train")
    plot.plot(eval_qds.get_unbatched(ts.id()), label="eval")
    #plot.plot(QDatasetForHuggingFaceModels(ts, split="eval", batch=False).raw_data[i].ts, label="eval") #error with eval_qds
    plot.plot(test_qds.get_unbatched(ts.id()),  label="test")
    plot.plot(train_org, mfc='blue', mec='blue', c="blue")
    plot.plot(train_generated)
    plot.plot(eval_org, mfc='blue', mec='blue', c="blue")
    plot.plot(eval_generated)
    plot.plot(test_org, mfc='blue', mec='blue', c="blue")
    plot.plot(test_generated)
    
    plot.save(f"{ts.id()}.png")
    
def _get_avg_metrics(metrics):
    return {
        'accuracy': np.array([m.accuracy for m in metrics]).mean(),
        'soft_accuracy': np.array([m.soft_accuracy for m in metrics]).mean(),
        'mae': np.array([m.mae for m in metrics]).mean()
    }


if __name__ == "__main__":
    factory, window_length = process_input()
    main(factory, window_length)
    exit()

    
   
    
    
   