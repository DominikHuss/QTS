# This __init__ is currently here, but will be removed once all of this is properly packed into a package 
from __init__ import *
from typing import Union, List

import numpy as np
from torchtyping import patch_typeguard
from typeguard import typechecked

from decorators import parse_args
patch_typeguard()

@parse_args(args_prefix="ts")
class TimeSeries():
    """
    Class representing time series.
    :param np.ndarray x: Indexes of time series
    :param np.ndarray y: Values of time series
    :param str id: Id of time series
    :param np.number min_y: Minimal value of time series
    :param np.number max_y: Maximum value of time series  
    """
    @typechecked
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 id: str,
                 *,
                 min_y: np.number = None,
                 max_y: np.number = None) -> None:

        assert len(x.shape) == 1
        assert len(y.shape) == 1

        self.x = x
        self.y = y
        self._id = id

        self.min_y = min_y if min_y is not None else np.min(y)
        self.max_y = max_y if max_y is not None else np.max(y)

    def add_to_plot(self, ax, *args, **kwargs):
        # TODO: No support for plotting unscaled data
        ax.plot(self.x, self.normalize(), *args, **kwargs)

    @typechecked
    def id(self) -> str:
        return self._id

    def normalize(self):
        return (self.y-self.min_y)/(self.max_y-self.min_y)

    def unnormalize(self, norm_y):
        return norm_y*(self.max_y-self.min_y) + self.min_y

    def length(self, split="none"):
        return self.splits_len[split]

    def get(self, split="none"):
        m = self.splits_masks[split]
        return self.x[m], self.y[m]

    def _build(self):
        train_mask = np.zeros_like(self.y, dtype=bool)
        eval_mask = np.zeros_like(self.y, dtype=bool)
        test_mask = np.zeros_like(self.y, dtype=bool)
        none_mask = np.zeros_like(self.y, dtype=bool)
        train_split = int(self.train_split*len(train_mask))
        eval_split = int(self.eval_split*len(eval_mask)) + train_split

        train_mask[:train_split] = 1
        if self.train_split + self.eval_split != 1:
            eval_mask[train_split:eval_split] = 1
            test_mask[eval_split:] = 1
        else:
            eval_mask[train_split:] = 1
        none_mask[:] = 1
        self.splits_masks = {"train": train_mask,
                             "eval": eval_mask,
                             "test": test_mask,
                             "none": none_mask}
        self.splits_len = {"train": np.sum(train_mask),
                           "eval": np.sum(eval_mask),
                           "test": np.sum(test_mask),
                           "none": np.sum(none_mask)}
        self.splits_starts = {"train": 0,
                              "eval": train_split,
                              "test": eval_split,
                              "none": 0}
        assert np.sum(self.splits_masks["train"] & self.splits_masks["eval"] & self.splits_masks["test"]) == 0
        if self.splits_len["eval"] != 0 and self.splits_starts["eval"] != 0:
            assert self.splits_masks["eval"][self.splits_starts["eval"]] == 1 and self.splits_masks["eval"][self.splits_starts["eval"]-1] == 0
        if self.splits_len["test"] != 0 and self.splits_starts["test"] != 0:
            assert self.splits_masks["test"][self.splits_starts["test"]] == 1 and self.splits_masks["test"][self.splits_starts["test"]-1] == 0
        

class QTimeSeries():
    """
    Class representing quantized time series
    
    :param TimeSeries ts: Original time series
    :param np.ndarray bin_idx: Quantized buckets (tokens) from time series
    :param np.ndarray bin_val:  Values of buckets (tokens)
    """
    @typechecked
    def __init__(self, 
                 ts: TimeSeries, 
                 bin_idx: np.ndarray, 
                 bin_val: np.ndarray) -> None:
        super().__init__()

        assert len(bin_idx.shape) == 1
        assert len(bin_val.shape) == 1

        self.ts = ts
        self.tokens = bin_idx
        self.tokens_y = bin_val

    def unnormalize(self):
        return self.ts.unnormalize(self.tokens_y)

    def add_to_plot(self, ax, *args, **kwargs):
        kwargs['mfc'] = kwargs['mfc'] if 'mfc' in kwargs else 'black'
        kwargs['mec'] = kwargs['mec'] if 'mec' in kwargs else 'black'
        kwargs['c'] = kwargs['c'] if 'c' in kwargs else 'black'
        
        ax.plot(range(self.ts.x[0], self.ts.x[0]+self.tokens_y.shape[0]), self.tokens_y, marker="|", ms=1, mew=1, ls="--", lw=0.06, **kwargs)

    def get(self, split="none"):
        m = self.ts.splits_masks[split]
        return self.tokens[m], self.tokens_y[m], self.ts.x[m], self.ts.y[m]

    def length(self, split="none"):
        return self.ts.splits_len[split]

    @typechecked
    def id(self) -> str:
        return self.ts.id()


@parse_args(args_prefix="qtz")
class TimeSeriesQuantizer():
    """
    Class representing quantizer.
    """
    @typechecked
    def quantize(self, 
                 _time_series: Union[TimeSeries, List[TimeSeries]],
                 *,
                 batch: bool = False) -> Union[QTimeSeries, List[QTimeSeries]]:
        """
        Quantized time series to quantized version.
        :param Union[TimeSeries, List[TimeSeries]] _time_series: Single time series or list of time series.
        :param bool batch: Boolean value. If True quantizer batch given time series. Defaults to False.
        :return: Quantized time series or list of time series; batched or not.
        :rtype: Union[QTimeSeries, List[QTimeSeries]] 
        """
        if isinstance(_time_series, list):
            time_series = _time_series
        else:
            time_series = [_time_series]

        time_series = self.batch(time_series) if batch else time_series

        bin_edges_strided = np.lib.stride_tricks.sliding_window_view(self.bins_edges, 2)
        qts = []
        for ts in time_series:
            y = ts.y
            y_norm = ts.normalize()
          
            zero_bin_assignments = (y == 0)[np.newaxis, :]
            l_bin_assignments = (y < self.l_bound)[np.newaxis, :] if self.l_bound \
                else np.zeros(y.shape, dtype = bool)[np.newaxis, :] 
            u_bin_assignments = (y > self.u_bound)[np.newaxis, :] if self.u_bound \
                else np.zeros(y.shape, dtype = bool)[np.newaxis, :]
            not_num_special_assignments = ~np.logical_or(zero_bin_assignments,
                                                np.logical_or(l_bin_assignments,
                                                              u_bin_assignments))
            others_assignments = np.zeros((len(self.special_tokens)-3,y.shape[0],), dtype = bool) #why -3? --> 3 num special tokens! 
            bin_assignments = np.logical_and(y_norm>bin_edges_strided[:, 0][np.newaxis, :].T, 
                                             y_norm<=bin_edges_strided[:, 1][np.newaxis, :].T)
            bin_assignments = not_num_special_assignments * bin_assignments
            all_bin_assignments = np.concatenate((bin_assignments,
                                                  zero_bin_assignments,
                                                  l_bin_assignments,
                                                  u_bin_assignments,
                                                  others_assignments))
            bin_idx = self.bins_indices@all_bin_assignments
            bin_val = np.take(self.bins_values, bin_idx) 
            if (not isinstance(_time_series, list)) and (not batch):
                return QTimeSeries(ts, bin_idx, bin_val)
            else:
                qts.append(QTimeSeries(ts, bin_idx, bin_val))
        return qts

    @typechecked
    def batch(self,
              _time_series: List[TimeSeries],
              *,
              _padded: bool = True) -> List[TimeSeries]:
        """
        Batch given time series.
        :param List[TimeSeries]: List of time series
        :param bool _padded: Boolean value. If true perform right padding to window_length.
        :return: Batched time series.
        :rtype:  List[TimeSeries]
        """
        time_series = []
        for ts in _time_series:
            if _padded and (n_padding := self._global_window_length - ts.length()) > 0: #:= works only with version python 3.8+ 
                ts.x = np.concatenate((ts.x, np.full(n_padding, self.special_tokens['pad'])))
                ts.y = np.concatenate((ts.y, np.full(n_padding, self.special_tokens['pad'])))
            y_batched = np.lib.stride_tricks.sliding_window_view(ts.y, self._global_window_length)
            x_batched = np.lib.stride_tricks.sliding_window_view(ts.x, self._global_window_length)
            for i in range(y_batched.shape[0]):
                time_series.append(TimeSeries(x_batched[i], y_batched[i], id=ts._id, min_y=ts.min_y, max_y=ts.max_y))
        return time_series

    def _build(self): 
        default_special_tokens ={
            "zero": self.num_bins,
            "low": self.num_bins + 1, #lower anomaly
            "upp": self.num_bins + 2, #upper anomaly
            "cls": self.num_bins + 3, 
            "bos": self.num_bins + 4, 
            "sep": self.num_bins + 5,
            "eos": self.num_bins + 6,
            "mask": self.num_bins + 7,
            "pad": self.num_bins + 8
        }
        additional_special_tokens = ({token: self.num_bins
                                             + len(self.default_special_tokens) + idx #maybe +1?
                                             for idx, token in enumerate(self.additional_special_bins)}
                                        if self.additional_special_bins else None)
        self.special_tokens = {**default_special_tokens, **additional_special_tokens} if additional_special_tokens else default_special_tokens
        self.bins_edges = np.linspace(0, 1, self.num_bins+1) #ts is always has normalized values in range [0,1]
        self.bins_values = np.array([0.5*self.bins_edges[i] + 0.5*self.bins_edges[i+1] for i in range(len(self.bins_edges)-1)] + [0.0, self.l_value, self.u_value])
        self.bins_indices = np.arange(self.num_bins+len(self.special_tokens)) 