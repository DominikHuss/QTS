from operator import ge
import os
import json
import numpy as np
from decorators import get_global_args
from _factory import FACTORIES
from plot import Plotter
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from metric import QMetric


@get_global_args
def main(_global_input_dir,
         _global_model,
         _global_cuda,
         **kwargs):
    print("Main script start")
    print("="*20)
    print("Model: ", _global_model)
    print("Data: ", _global_input_dir)
    print("Cuda: ", _global_cuda)
    print("="*20)
    
    train_qds, eval_qds, test_qds, qmc = _process_input(_global_input_dir, _global_model)
    #print(vars(qmc.model))
    print("Training start")
    qmc.train(train_qds, eval_qds)
    print("Training end")
    print("="*20)
    print("Plotting")
    train_metric, eval_metric, test_metric = [],[],[]
    for id_ ,(train_ts, eval_ts, test_ts) in enumerate(zip(train_qds.raw_unbatched_data,
                                          eval_qds.raw_unbatched_data,
                                          test_qds.raw_unbatched_data)):
        train_first_window, train_org = _split_original_ts(train_qds, id_)
        eval_first_window, eval_org = _split_original_ts(eval_qds, id_)
        test_first_window, test_org =_split_original_ts(test_qds, id_)
        
        fix_horizon = 1 if _global_model == "transformer" else 0
        train_generated = _get_only_generated(qmc, len(train_org.tokens) + fix_horizon, train_qds, id_)
        eval_generated = _get_only_generated(qmc, len(eval_org.tokens) + fix_horizon, eval_qds, id_)
        test_generated = _get_only_generated(qmc, len(test_org.tokens) + fix_horizon, test_qds, id_)
        
       
        if id_  % 10 == 0:
            print(f"Plot {id_} ts")
            _plot(train_ts, eval_ts, test_ts,
                  train_generated,train_first_window,
                  eval_generated,eval_first_window,
                  test_generated,test_first_window)
        train_metric.append(QMetric(train_org, train_generated))
        eval_metric.append(QMetric(eval_org, eval_generated))
        test_metric.append(QMetric(test_org, test_generated))    
    
    metric ={
        **{"Overall (avg on test splits)": _get_avg_metrics(test_metric)},
        **{i: {
            "train": train_metric[i].__dict__,
            "eval": eval_metric[i].__dict__,
            "test": test_metric[i].__dict__
            } for i in range(len(train_metric))}
    }
    print("="*20)
    print("Performance: ",metric['Overall (avg on test splits)'])
    with open('./plots/metrics.json', 'w') as f:
        json.dump(metric, f, indent=4)
    print("="*20)
    print("Main script end")


def _process_input(input_dir, model): 
    ts = []
    i = 0 #don't use enumerate!
    for file in os.listdir(input_dir):
        if file == 'plots' or file == "params.json":
            continue
        y = np.loadtxt(os.path.join(input_dir, file), delimiter=",")
        x = np.arange(len(y))
        ts.append(TimeSeries(x, y, f"{i}"))
        i += 1 
    factory = FACTORIES[model](ts)
    train_qds = factory.get_dataset(split="train", batch=True)
    eval_qds = factory.get_dataset(split="eval", batch=True)
    test_qds = factory.get_dataset(split="test", batch=True)
    qmc = factory.get_container()
    
    return train_qds, eval_qds, test_qds, qmc

def _get_only_generated(qmc,
                        horizon,
                        qds,
                        id_):
    generated = qmc.generate(qds,
                        id=str(id_),
                        horizon=int(horizon))
    generated.tokens = generated.tokens[qmc._global_window_length:]
    generated.tokens_y = generated.tokens_y[qmc._global_window_length:]
    generated.ts.x = np.arange(generated.ts.x[-1]+1, generated.ts.x[-1]+1 + len(generated.tokens))
    
    return  generated


def _split_original_ts(qds,id_):
    (first_window, _),_, _ = qds.get_batched(str(id_),_all=True)
    rest = qds.get_unbatched(str(id_), _quantized=True)
    rest.tokens = rest.tokens[first_window.length():]
    rest.tokens_y = rest.tokens_y[first_window.length():]
    rest.ts.x = rest.ts.x[first_window.length():]
    rest.ts.y = rest.ts.y[first_window.length():]
    return first_window, rest
    

def _plot(train_ts, eval_ts, test_ts,
          train_generated, train_first_window,
          eval_generated, eval_first_window,
          test_generated, test_first_window):
    plot = Plotter(TimeSeriesQuantizer(), "plots/")
    plot.plot(train_ts, label="train")
    plot.plot(eval_ts, label="eval")
    plot.plot(test_ts,  label="test")
    plot.plot(train_first_window, mfc='blue', mec='blue', c="blue")
    plot.plot(train_generated)
    plot.plot(eval_first_window, mfc='blue', mec='blue', c="blue")
    plot.plot(eval_generated)
    plot.plot(test_first_window, mfc='blue', mec='blue', c="blue")
    plot.plot(test_generated)
    
    plot.save(f"{train_ts.id()}.png")
    
def _get_avg_metrics(metrics):
    return {
        'accuracy': np.array([m.accuracy for m in metrics]).mean(),
        'soft_accuracy': np.array([m.soft_accuracy for m in metrics]).mean(),
        'mae': np.array([m.mae for m in metrics]).mean()
    }


if __name__ == "__main__":
    main()
    exit()

    
   
    
    
   