from dataset import QDataset
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args

import os

import matplotlib.pyplot as plt
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)

import warnings
from typing import Union, Iterable, List, Literal
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

class Plotter():
    @typechecked
    def __init__(self,
                 quantizer: TimeSeriesQuantizer,
                 save_location: str,
                 *,
                 _unscale: bool = False) -> None:
        # TODO: No support for plotting unscaled data
        if _unscale:
            raise Exception("Plotter does not yet allow for plotting original-space data!")

        self.quantizer = quantizer
        self.save_location = save_location

        # TODO: currently added support for one plot per Plotter instance
        self.num_axis = 1

        self.fig, self.ax = plt.subplots(self.num_axis, dpi=800)
        self._prepare_grid()


    def _prepare_grid(self):
        self.ax.set_ylim(min(self.quantizer.bins_values), max(self.quantizer.bins_values))

        self.ax.xaxis.set_major_locator(MultipleLocator(10))
        self.ax.xaxis.set_minor_locator(AutoMinorLocator(10))

        self.ax.yaxis.set_major_locator(MultipleLocator(1))
        self.ax.yaxis.set_minor_locator(AutoMinorLocator(self.quantizer.num_bins))

        self.ax.grid(b=True, which='major', axis='x', color='#CCCCCC', linestyle='--')
        self.ax.grid(b=True, which='major', axis='y', color='#CCCCCC', linestyle='-')
        self.ax.grid(b=True, which='minor', color='#CCCCCC', linestyle=':')

    @typechecked
    def plot(self,
             ts: Union[TimeSeries, QTimeSeries],
             *args,
             **kwargs) -> None:
        ts.add_to_plot(self.ax, *args, **kwargs)

    @typechecked
    def save(self,
             filename: str) -> None:
        self.fig.legend()
        self.fig.savefig(os.path.join(self.save_location, filename))