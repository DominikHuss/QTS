import matplotlib.pyplot as plt
import numpy as np
import torch
from model import TransformerModel
from model import QModelContainer
from plot import Plotter
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from dataset import QDataset


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
    if qmc._global_SMOKE_TEST:
        exit()

    
    for i in range(len(ts)):
        plot = Plotter(TimeSeriesQuantizer(), "plots/")
        plot.plot(ts[i], label="true")
        plot.plot(train_qds.get_unbatched(f"ts{i}"), label="train")
        plot.plot(QDataset(ts, split="eval", batch=False).raw_data[i].ts, label="eval")
        plot.plot(qmc.generate(train_qds, id=f"ts{i}", horizon=int(train_qds.get_unbatched(f"ts{i}").length()-(qmc.window_length-qmc.num_last_unmasked))))
        plot.save(f"train{i}.png")
        
        eval_plot = Plotter(TimeSeriesQuantizer(), "plots/")
        eval_plot.plot(all_qds.get_unbatched(f"ts{i}"), label="true")
        eval_plot.plot(train_qds.get_unbatched(f"ts{i}"), label="train")
        eval_plot.plot(eval_qds.get_unbatched(f"ts{i}"),  label="eval")
        eval_plot.plot(qmc.generate(eval_qds, id=f"ts{i}", horizon=int(eval_qds.get_unbatched(f"ts{i}").length()-(qmc.window_length-qmc.num_last_unmasked))))
        eval_plot.save(f"eval{i}.png")
        
        test_plot = Plotter(TimeSeriesQuantizer(), "plots/")
        test_plot.plot(all_qds.get_unbatched(f"ts{i}"), label="true")
        test_plot.plot(train_qds.get_unbatched(f"ts{i}"), label="train")
        test_plot.plot(eval_qds.get_unbatched(f"ts{i}"),  label="eval")
        test_plot.plot(test_qds.get_unbatched(f"ts{i}"),  label="test")
        test_plot.plot(qmc.generate(test_qds, id=f"ts{i}", horizon=int(test_qds.get_unbatched(f"ts{i}").length()-(qmc.window_length-qmc.num_last_unmasked))))
        test_plot.save(f"test{i}.png")

    # print("--------")
    # num_samples = 100
    # # all
    # train_random_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    # for _ in range(num_samples):
    #     train_random_plot.plot(qmc.generate(train_qds, 
    #                                         id="sin", 
    #                                         stochastic=True, 
    #                                         horizon=int(train_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked))), 
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