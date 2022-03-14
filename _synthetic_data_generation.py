import json
import numpy as np
import pandas as pd
from timeseries_generator import LinearTrend, Generator, WhiteNoise, HolidayFactor, WeekdayFactor,SinusoidalFactor, RandomFeatureFactor


def generate_univariate_ts(start: str = "01-01-2018",
                        end: str = "01-01-2019",
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
    hd = HolidayFactor(holiday_factor = 2.,
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
        
    for i in range(10):
        gen_params = {
            "wn_std": 0.15,
            "week_days": {0: 0.89,
                1: 0.92,
                2: 0.95,
                3: 0.91,
                4: 1.11,
                5: 1.15,
                6: 1.2
            },
            "holidays": {
                "Christmas_day": np.random.normal(6),
                "Easter Monday": np.random.normal(6),
                "All Saints' Day": np.random.normal(6),
                "International Workers' Day":np.random.normal(6),
                "Liberation Day":np.random.normal(6)
            },
            "sin_config": {"wavelength": 366,
                    "amplitude": 0.2,
                    "phase": 0,
                    "mean": 0.5
            }
        }
        with open(f"./data/synthetic/synthetic_params{i}.json","w") as fp:
            json.dump(gen_params,fp)
        
        example_ts = generate_univariate_ts(**gen_params)
        np.savetxt(f"./data/synthetic/synthetic{i}.csv", example_ts, delimiter=",")
        
        ds = QDataset(example_ts)
        plot = Plotter(TimeSeriesQuantizer(), "./data/synthetic/plots/")
        
        qts = ds.get_batched(id= "0")
        ts = ds.get_unbatched(id= "0")
        plot.plot(ts, label="cont.")
        plot.plot(qts, label="quant.")
        plot.save(f"synthetic{i}.png")