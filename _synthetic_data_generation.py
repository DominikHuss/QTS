import json
import numpy as np
import pandas as pd
from timeseries_generator import LinearTrend, Generator, WhiteNoise, HolidayFactor, WeekdayFactor,SinusoidalFactor, RandomFeatureFactor

from dataset import QDatasetBase


def generate_univariate_ts(start: str = "01-01-2018",
                        end: str = "01-01-2021",
                        lt_coef:  float = None,
                        lt_offset: float = None,
                        wn_std: float = None,
                        holidays: dict = None,
                        week_days: dict = None,
                        sin_config: dict = None
                     
) -> np.array:
    lt = LinearTrend(coef= lt_coef,
                    offset= lt_offset,
                    col_name="lt_trend"
    ) if lt_coef and lt_offset else None
    wn = WhiteNoise(stdev_factor= wn_std) if wn_std else None
    hd = HolidayFactor(holiday_factor = 1.2,
        special_holiday_factors = holidays
    ) if holidays else None
    wd = WeekdayFactor(col_name="week",
                    factor_values= week_days 
    )   if week_days else None
    rff = RandomFeatureFactor(
        feature="random_feature",
        feature_values=["val"],
        min_factor_value=1,
        max_factor_value=10
    )
    sin = SinusoidalFactor(
            feature="sin_feature",
            col_name="sin_factor",
            feature_values={
                "val_sin1": {
                    "wavelength": sin_config['wavelength'],
                    "amplitude": sin_config['amplitude'],
                    "phase": sin_config['phase'],
                    "mean": sin_config['mean']
                }    
            }
    ) if sin_config  else None
    
    features_dict = {
        "country": ["Italy"],
        "sin_feature":  ["val_sin1"],
        "random_feature": ["val"]
    }
    factors = [*filter(lambda f: f is not None, [lt,wn,rff,sin,hd,wd])]
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
    
    gen_params = {
            # "lt_coef": 1.0,
            # "lt_offset": 0.0000,
            "wn_std": 0.005,
            # "week_days": {0: 0.99,
            #     1: 0.98,
            #     2: 0.99,
            #     3: 1.0,
            #     4: 1.01,
            #     5: 1.02,
            #     6: 1.01
            # },
            # "holidays": {
            #     "Christmas_day": abs(np.random.normal()),
            #     "Easter Monday":abs(np.random.normal()),
            #     "All Saints' Day": abs(np.random.normal()),
            #     "International Workers' Day": abs(np.random.normal()),
            #     "Liberation Day": abs(np.random.normal())
            # },
            "sin_config": {"wavelength":90,
                    "amplitude": 0.02,
                    "phase": 0,
                    "mean": 0
            }
        }
    gen_params2 = {
            "wn_std": 0.02,
            "sin_config": {"wavelength": 60,
                    "amplitude": 1,
                    "phase": 0,
                    "mean": 2
            }
        }
    gen_params3 = {
            "wn_std": 0.02,
            "sin_config": {"wavelength": 30,
                    "amplitude": 1,
                    "phase": 90,
                    "mean": 2
            }
        }
    # params = {"sin1":gen_params,
    #           "sin2":gen_params2,
            #   "sin3":gen_params3}
    with open(f"./data/basic/params.json","w") as fp:
        json.dump(gen_params,fp,indent=4)
    
    for i in range(10):
        example_ts = generate_univariate_ts(**gen_params)
        # example_ts2 = generate_univariate_ts(**gen_params2)
        # example_ts3 = generate_univariate_ts(**gen_params3)
        # example_ts += example_ts2 + example_ts3
        # shift = np.zeros(example_ts.shape)
        # period = len(shift)//6
        # shift[period:2*period] = 10
        # shift[3*period:4*period] = 10
        # shift[5*period:] = 10
        # example_ts -= shift
        np.savetxt(f"./data/basic/ts{i}.csv", example_ts, delimiter=",")
        
        ds = QDatasetBase(example_ts)
        plot = Plotter(TimeSeriesQuantizer(), "./data/basic/plots/")
        
        qts = ds.get_batched(id= "0")
        ts = ds.get_unbatched(id= "0")
        plot.plot(ts, label="cont.")
        plot.plot(qts, label="quant.")
        plot.save(f"ts{i}.png")