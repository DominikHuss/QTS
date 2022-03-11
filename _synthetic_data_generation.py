import numpy as np
import pandas as pd
from numpy.core.fromnumeric import var
from timeseries_generator import LinearTrend, Generator, WhiteNoise, HolidayFactor, WeekdayFactor,RandomFeatureFactor,SinusoidalFactor

def generate_univariate_ts(start: str = "01-01-2018",
                        end: str = "01-01-2019",
                        rand_lt: bool = False,
                        lt_coef:  float = None,
                        lt_offset: float = None,
                        rand_wn: bool = False, 
                        wn_std: float = None,
                        rand_holidays: bool = False,
                        holidays: dict = None,
                        rand_week_days: bool = False,
                        week_days: dict = None,
                        seed: int = None
) -> np.array:
    if seed: np.seed(seed)
    if rand_lt:
        lt_coef = np.random.normal()
        lt_offset = np.random.normal()
    if rand_wn: wn_std = np.random.normal(scale = 0.005)
    if rand_holidays: holidays = {
        "Christmas_day": np.random.normal(10),
        "Easter Monday": np.random.normal(6),
        "All Saints' Day": np.random.normal(3)
    }
    if rand_week_days: week_days = {4: np.random.normal(1.1),
                    5: np.random.normal(1.4),
                    6: np.random.normal(1.2)
    }
    
    lt = LinearTrend(coef= lt_coef,
                    offset= lt_offset,
                    col_name="lt_trend"
    ) if lt_coef and lt_offset else None
    wn = WhiteNoise(stdev_factor= wn_std) if wn_std else None
    hd = HolidayFactor(holiday_factor = 2.,
        special_holiday_factors = holidays
    ) if holidays else None
    wd = WeekdayFactor(col_name="week",
                    factor_values= week_days 
    )   if week_days else None
    # rff = RandomFeatureFactor(
    #     feature_values = ["val1",'val2','val3'],
    #     feature = "f1",
    #     min_factor_value = 0.1,
    #     max_factor_value = 3,
    #     col_name = "rff"
    # )
    sin = SinusoidalFactor(
        feature="sin_feature",
        col_name="sin_factor",
        feature_values={
            "val_sin1": {
                "wavelength": 365.,
                "amplitude": 0.2,
                "phase": 365/4,
                "mean": 1.
            },
            "val_sin2": {
                "wavelength": 365.,
                "amplitude": 0.2,
                "phase": 0.,
                "mean": 1.
            }
        }
    )
    
    features_dict = {
        "country": ["Netherlands", "Italy", "Romania"],
        "sin_feature":  ["val_sin1",'val_sin2']
    }
    factors = [*filter(lambda f: f is not None, [lt,wn,hd,wd])]
    generator = Generator(factors={*factors},
                          features= features_dict,
                          date_range=pd.date_range(start=start, end = end))
    
    test = generator.generate()
    return np.array(test['value'])
    
    
def random_peaks(length, peak_mean, peak_variance, peak_frequency):
    zeros = np.zeros(length)
    peaks_indices = np.random.binomial(n=1, p=peak_frequency, size=zeros.size)
    peak_values = np.abs(np.random.normal(loc=peak_mean, scale=peak_variance, size=zeros.size))
    peaks = peaks_indices*peak_values
    return peaks

def random(length, mean, variance):
    return np.random.normal(loc=mean, scale=variance, size=length)


if __name__ == "__main__":
    from preprocessing import  TimeSeriesQuantizer
    from dataset import QDataset
    from plot import Plotter

    example_ts = generate_univariate_ts(rand_lt = False,
                        lt_coef = None,
                        lt_offset = None,
                        rand_wn = True, 
                        wn_std = None,
                        rand_holidays = True,
                        holidays = None,
                        rand_week_days = False,
                        week_days = None,
                        seed = None
    )
    
    ds = QDataset(example_ts)
    plot = Plotter(TimeSeriesQuantizer(), "plots/")
    
    qts = ds.get_batched(id="0")
    ts = ds.get_unbatched(id="0")
    plot.plot(ts, label="cont.")
    plot.plot(qts, label="quant.")
    plot.save("synth.png")
    print(qts.tokens)