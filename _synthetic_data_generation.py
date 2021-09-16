import numpy as np
from numpy.core.fromnumeric import var

def random_peaks(length, peak_mean, peak_variance, peak_frequency):
    zeros = np.zeros(length)
    peaks_indices = np.random.binomial(n=1, p=peak_frequency, size=zeros.size)
    peak_values = np.abs(np.random.normal(loc=peak_mean, scale=peak_variance, size=zeros.size))
    peaks = peaks_indices*peak_values
    return peaks

def random(length, mean, variance):
    return np.random.normal(loc=mean, scale=variance, size=length)


if __name__ == "__main__":
    from preprocessing import TimeSeries, TimeSeriesQuantizer
    from dataset import QDataset
    from plot import Plotter

    ds = QDataset(random_peaks(100, 1, 2, 0.2))

    plot = Plotter(TimeSeriesQuantizer(), "plots/")
    
    qts = ds.get_batched(id="0")
    ts = ds.get_unbatched(id="0")
    plot.plot(ts, label="cont.")
    plot.plot(qts, label="quant.")
    plot.save("synth.png")
    print(qts.tokens)