# This __init__ is currently here, but will be removed once all of this is properly packed into a package 
from __init__ import *
from decorators import parse_args

import numpy as np

from typing import Union, Iterable, List
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

@parse_args(args_prefix="ts")
class TimeSeries():
    @typechecked
    def __init__(self, 
                 x: np.ndarray, 
                 y: np.ndarray,
                 *,
                 min_y: np.number = None,
                 max_y: np.number = None) -> None:

        assert len(x.shape) == 1
        assert len(y.shape) == 1

        self.x = x
        self.y = y

        self.min_y = min_y if min_y is not None else np.min(y)
        self.max_y = max_y if max_y is not None else np.max(y)

    def add_to_plot(self, ax, *args, **kwargs):
        pass

    def normalize(self):
        return (self.y-self.min_y)/(self.max_y-self.min_y)

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
        #print(self.splits_masks)
        #print(self.splits_starts)
        #print(self.splits_len)
        assert np.sum(self.splits_masks["train"] & self.splits_masks["eval"] & self.splits_masks["test"]) == 0
        if self.splits_len["eval"] != 0 and self.splits_starts["eval"] != 0:
            assert self.splits_masks["eval"][self.splits_starts["eval"]] == 1 and self.splits_masks["eval"][self.splits_starts["eval"]-1] == 0
        if self.splits_len["test"] != 0 and self.splits_starts["test"] != 0:
            assert self.splits_masks["test"][self.splits_starts["test"]] == 1 and self.splits_masks["test"][self.splits_starts["test"]-1] == 0
        

class QTimeSeries():
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

    def get(self, split="none"):
        m = self.ts.splits_masks[split]
        return self.tokens[m], self.tokens_y[m], self.ts.x[m], self.ts.y[m]

    def length(self, split="none"):
        return self.ts.splits_len[split]


@parse_args(args_prefix="qtz")
class TimeSeriesQuantizer():
    @typechecked
    def quantize(self, 
                 _time_series: Union[TimeSeries, List[TimeSeries]],
                 *,
                 batch: bool = False) -> Union[QTimeSeries, List[QTimeSeries]]:
        if isinstance(_time_series, list):
            time_series = _time_series
        else:
            time_series = [_time_series]

        time_series = self.batch(time_series) if batch else time_series

        bin_edges_strided = np.lib.stride_tricks.sliding_window_view(self.bins_edges, 2)
        qts = []
        for ts in time_series:
            y = ts.normalize()
            #print(y)
            bin_assignments = np.logical_and(y>bin_edges_strided[:, 0][np.newaxis, :].T, 
                                             y<=bin_edges_strided[:, 1][np.newaxis, :].T)
            zero_bin_assignments = (y==0)[np.newaxis, :]
            l_bin_assignments = (y<self.bins_edges[0])[np.newaxis, :]
            u_bin_assignments = (y>self.bins_edges[-1])[np.newaxis, :]
            all_bin_assignments = np.concatenate((bin_assignments, zero_bin_assignments, l_bin_assignments, u_bin_assignments))
            bin_idx = self.bins_indices@all_bin_assignments
            #bin_val = self.bins_values@all_bin_assignments
            bin_val = np.take(self.bins_values, bin_idx)

            if (not isinstance(_time_series, list)) and (not batch):
                return QTimeSeries(ts, bin_idx, bin_val)
            else:
                #print(bin_idx)
                qts.append(QTimeSeries(ts, bin_idx, bin_val))
        return qts

    def batch(self,
              _time_series: List[TimeSeries]):
        time_series = []
        for ts in _time_series:
            y_batched = np.lib.stride_tricks.sliding_window_view(ts.y, self.window_length)
            x_batched = np.lib.stride_tricks.sliding_window_view(ts.x, self.window_length)
            #print(y_batched)
            for i in range(y_batched.shape[0]):
                time_series.append(TimeSeries(x_batched[i], y_batched[i], min_y=ts.min_y, max_y=ts.max_y))
        return time_series


    def _build(self):
        self.bins_edges = np.linspace(self.l_bound, self.u_bound, self.num_bins+1)
        self.bins_values = np.array([0.5*self.bins_edges[i] + 0.5*self.bins_edges[i+1] for i in range(len(self.bins_edges)-1)] + [0.0, self.l_value, self.u_value])
        self.bins_indices = np.arange(self.num_bins+self.num_special_bins)
        #print(self.bins_edges)
        #print(self.bins_values)
        #print(self.bins_indices)


if __name__ == '__main__':
    
    x = np.arange(15)
    y = np.arange(15)
    tsq = TimeSeriesQuantizer()
    print(tsq.quantize(TimeSeries(x,y)).get())
    qts = tsq.quantize(TimeSeries(x,y), batch=True)
    for q in qts:
        print(q.get())
    #t = qts.get()
    #print(t[0].shape)
    #print(qts.length())
    #t2 = qts.get("train")
    #print(t2[0].shape)
    #print(qts.length("train"))