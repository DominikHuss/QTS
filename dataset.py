import warnings
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Tuple, Union, Iterable, List, Literal, Optional
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

@parse_args(args_prefix="qds")
class QDataset(Dataset):
    @typechecked
    def __init__(self, 
                 _data: Union[TimeSeries, List[TimeSeries], QTimeSeries, List[QTimeSeries]],
                 split: Literal["train", "eval", "test", "none"] = "none",
                 *,
                 batch: bool = False) -> None:

        if isinstance(_data, list):
            _data_list = _data
        else:
            _data_list = [_data]

        if type(_data_list[0]).__name__ == TimeSeries.__name__:
            _data_list = [TimeSeries(*d.get(split), id=d._id, min_y=d.min_y, max_y=d.max_y) for d in _data_list]
            self.raw_unbatched_data = _data_list
            self.raw_data =  TimeSeriesQuantizer().quantize(_data_list, batch=batch)
        else:
            self.raw_unbatched_data = None
            self.raw_data = _data_list

        self.split = split
        self.batch = batch

    def get_unbatched(self,
                      id: str,
                      *,
                      _quantized: bool = False) -> Optional[TimeSeries]:
        if self.raw_unbatched_data is None:
            warnings.warn("Initializing a QDataset with QTimeSeries holds not enough information to dissambiguate whether data are batched already. Returning None")
            return None

        if _quantized and not self.batch:
            ts_data = self.raw_data
        elif _quantized and self.batch:
            warnings.warn("Requested quantized unbatched data, but instance was initialized with batch=True. Returning None")
            return None
        else:
            ts_data = self.raw_unbatched_data

        for ts in ts_data:
            if ts.id() == id:
                return ts
        warnings.warn(f"TimeSeries with id={id} was not found in the QDataset. Returning None")
        return None



    @typechecked
    def get(self,
            id: str) -> Tuple[Tuple[Optional[QTimeSeries], Optional[TensorType[-1]]], int, int]:
        length = 0
        start = -1
        qts = None
        prepped_y = None
        for i in range(len(self.raw_data)):
            if self.raw_data[i].id() == id:
                length += 1
                if start == -1:
                    start = i
                    qts = self.raw_data[i]
                    prepped_y = self.data[i]["y"]
            else:
                if start != -1:
                    break
        if prepped_y is None:
            warnings.warn(f"No mini-batches of a QTimeSeries with matching id='{id}' were found in the QDataset! Returning None")
        return (qts, prepped_y), start, length
            

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        y = self.data[idx]["y"]
        y_hat = self.data[idx]["y_hat"]
        mask = self.data[idx]["mask"]
        qts_idx = torch.tensor([idx])
        return {"y": y, "y_hat": y_hat, "mask": mask, "idx": qts_idx} 

    def _get_y(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        tokens = tokens[:-1] if self.objective == "ar" else tokens
        return torch.from_numpy(tokens)

    def _get_y_hat(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        tokens = tokens[1:] if self.objective == "ar" else tokens
        return torch.from_numpy(tokens)

    def _get_mask(self, idx):
        l = self.raw_data[idx].length(self.split if self.inner_split else "none") - 1
        mask = torch.zeros((l), dtype=torch.bool)
        if self.objective == "ar":
            mask[-self.num_last_unmasked:] = 1
        else:
            raise NotImplementedError("MLM isn't implemented yet.")
        return mask

    def _build(self):
        self.data = {idx: {"y": self._get_y(idx),
                           "y_hat": self._get_y_hat(idx),
                           "mask": self._get_mask(idx)} for idx in range(len(self.raw_data))}

if __name__ == "__main__":
    import numpy as np
    x = np.arange(15)
    y = np.arange(15)
    ts = TimeSeries(x,y)
    qds = QDataset(ts, batch=True)
    out = qds[0]
    print(out["y"].shape)
    print(out["y_hat"].shape)
    print(out["mask"].shape)
    print(out["i"].shape)
    dl = DataLoader(qds, batch_size=4, shuffle=False)
    for b in dl:
        print(b)