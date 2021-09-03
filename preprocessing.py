# This __init__ is currently here, but will be removed once all of this is properly packed into a package 
from __init__ import *
from decorators import parse_args

import torch
import torch.nn.functional as F

from typing import Union, Iterable, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()


class TimeSeries():
    @typechecked
    def __init__(self, 
                 x: TensorType["T", "x_features"], 
                 y: TensorType["T", 1]) -> None:
        self.x = x
        self.y = y

    def add_to_plot(self, ax, *args, **kwargs):
        pass

class QTimeSeries(TimeSeries):
    def __init__(self, *args) -> None:
        super().__init__(*args)

@parse_args(args_prefix="qtz")
class TimeSeriesQuantizer():

    @typechecked
    def quantize(self, time_series: Union[TimeSeries, List[TimeSeries]]) -> Union[QTimeSeries, List[QTimeSeries]]:
        if type(time_series) == type([]):
            time_series = [time_series]
        
        for ts in time_series:
            pass

    def _build(self):
        bins_edges = torch.linspace(self.l_bound, self.u_bound, self.num_bins+1)
        bins_values = torch.tensor([0.5*bins_edges[i] + 0.5*bins_edges[i+1] for i in range(len(bins_edges)-1)] + [0.0, self.])
        bin_indices = torch.arange(self.num_bins+1+self.num_special_bins)
        print(bins_edges)
        print(bins_values)
        print(bin_indices)

if __name__ == '__main__':
    tsq = TimeSeriesQuantizer()