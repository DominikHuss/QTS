import warnings
import copy
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from typing import Dict, Tuple, Union, Iterable, List, Literal, Optional
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

@parse_args(args_prefix="qds")
class QDataset(Dataset):
    @typechecked
    def __init__(self, 
                 _data: Union[np.ndarray, List[np.ndarray], TimeSeries, List[TimeSeries], QTimeSeries, List[QTimeSeries]],
                 split: Literal["train", "eval", "test", "none"] = "none",
                 *,
                 batch: bool = False,
                 soft_labels: bool = False,
                 random_shifts: bool = False) -> None:
        self.tsq = TimeSeriesQuantizer()

        if isinstance(_data, list):
            _data_list = _data
        else:
            _data_list = [_data]

        if isinstance(_data_list[0], np.ndarray):
            _data_list = [TimeSeries(x=np.arange(len(d)), y=d, id=f"{i}") for i, d in enumerate(_data_list)]

        if type(_data_list[0]).__name__ == TimeSeries.__name__:
            _data_list = [TimeSeries(*d.get(split), id=d._id, min_y=d.min_y, max_y=d.max_y) for d in _data_list]
            self.raw_unbatched_data = _data_list
            self.raw_data =  self.tsq.quantize(_data_list, batch=batch)
        else:
            self.raw_unbatched_data = None
            self.raw_data = _data_list

        self.split = split
        self.batch = batch
        self.soft_labels = soft_labels
        self.random_shifts = random_shifts
        
        self.mlm_masked_probability = 0.20 #TO DO: add as args
        self.mlm_non_masked_value = -100 #TO DO: add as args
        self.mlm_masked_token_prob = 0.8 #TO DO: add as args
        self.mlm_random_token_prob = 0.1 #TO DO: add as args
        assert self.mlm_masked_token_prob + self.mlm_random_token_prob <= 1
        if self.mlm_masked_token_prob + self.mlm_random_token_prob == 1:
            warnings.warn("Probability of masked inputs with token [MASK] and probability of random replace equals 1" +
                      "None of the inputs tokens will be unchanged. It will cause invalid training model!") 
    
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
            warnings.warn("Only works for window_step = 1")
            tokens,tokens_y = None, None 
            for qts in self.raw_data:
                if qts.id() == id:
                    if tokens is None and tokens_y is None:
                        tokens = qts.tokens.tolist()
                        tokens_y =  qts.tokens_y.tolist()
                    else: 
                        tokens.append(qts.tokens[-1])
                        tokens_y.append(qts.tokens_y[-1])
            return QTimeSeries(ts=self.get_unbatched(id),
                       bin_idx=np.asarray(tokens),
                       bin_val=np.asarray(tokens_y))
            #Added get_unbatched quantized data, so commented below:
            # warnings.warn("Requested quantized unbatched data, but instance was initialized with batch=True. Returning None")
            # return None
        else:
            ts_data = self.raw_unbatched_data

        for ts in ts_data:
            if ts.id() == id:
                return copy.copy(ts)
        warnings.warn(f"TimeSeries with id={id} was not found in the QDataset. Returning None")
        return None

    @typechecked
    def get_batched(self,
                    id: str,
                    *,
                    _all=False) -> Union[Tuple[Tuple[Optional[QTimeSeries], 
                                                     Optional[TensorType[-1]]], 
                                                     int, 
                                                     int], 
                                         QTimeSeries]:
        length = 0
        start = -1
        qts = None
        prepped_y = None
        for i in range(len(self.raw_data)):
            if self.raw_data[i].id() == id:
                if not _all:
                    return copy.copy(self.raw_data[i])
                length += 1
                if start == -1:
                    start = i
                    qts = copy.copy(self.raw_data[i])
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
        if self.objective == "ar":
            y = self.data[idx]["y"]
            if self.random_shifts:
                y = self._random_shifts(y)
            y_hat = self.data[idx]["y_hat"]
            y_hat_probs = self.data[idx]["y_hat_probs"]
            mask = self.data[idx]["mask"]
            qts_idx = torch.tensor([idx]) 
            return {"y": y, "y_hat": y_hat, "y_hat_probs": y_hat_probs, "mask": mask, "idx": qts_idx} 
        if self.objective == "mlm":
            return self._get_mlm_data(idx)
       
    def _random_shifts(self,y:torch.Tensor):
        shifts_idx = torch.tensor([0.1, 0.8, 0.1]).multinomial(num_samples=y.shape[0], replacement=True)
        shifts = torch.tensor([-1, 0, 1])[shifts_idx]
        specials_mask = y>=self.tsq.num_bins
        y = F.relu(y+~specials_mask*shifts)
        specials_mask_check = y>=self.tsq.num_bins
        y[torch.logical_xor(specials_mask, specials_mask_check)] -= 1
        return y
    
    def _get_mlm_data(self, idx=None):
        """
        Warning: masked also special token (eg. zero, l/u_anomaly, potential <sep> (sentence A <sep> sentence B))
        For HG models. Masking mlm.probability of tokens. Inner split:
            - from mlm_masked_token_prob probability masking as [MASK]
            - from mlm_random_token_prob probability masking as random token (sampling from num_tokens only)
            - rest of tokens are notchanged
        """
        tokens, labels = self.data[idx]['y'].clone(), self.data[idx]['y'].clone()
        special_tokens_matrix = tokens >= self.tsq.num_bins  
        probability_matrix = torch.full(tokens.shape, self.mlm_masked_probability)
        probability_matrix.masked_fill_(special_tokens_matrix, value=0.0)
        masked_idx = torch.bernoulli(probability_matrix).bool()
        labels[~masked_idx] = self.mlm_non_masked_value
        masked_tokens_idx = torch.bernoulli(torch.full(tokens.shape, self.mlm_masked_token_prob)).bool() & masked_idx
        tokens[masked_tokens_idx] = self.tsq.special_tokens['mask']
        random_tokens_idx = torch.bernoulli(torch.full(tokens.shape, self.mlm_random_token_prob)).bool() & masked_idx & ~masked_tokens_idx
        random_tokens = torch.randint(self.tsq.num_bins, tokens.shape, dtype=torch.long)
        tokens[random_tokens_idx] = random_tokens[random_tokens_idx]
        # tokens = torch.cat((torch.tensor([self.tsq.special_tokens['cls']]),
        #                     tokens,
        #                     torch.tensor([self.tsq.special_tokens['sep']])))
        # labels = torch.cat((torch.tensor([self.mlm_non_masked_value]),
        #                     labels,
        #                     torch.tensor([self.mlm_non_masked_value])))
        return {"input_ids": tokens,
                "labels": labels}
    
    def _get_y(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        if self.objective == "ar":
            tokens = tokens[:-1]
        else:
            tokens = np.concatenate(([self.tsq.special_tokens['cls']],
                                      tokens,
                                      [self.tsq.special_tokens['sep']]))
        return torch.from_numpy(tokens).to(torch.long)

    def _get_y_hat(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        if self.objective == "ar": 
            tokens = tokens[1:]
        return torch.from_numpy(tokens).to(torch.long)

    def _get_y_hat_probs(self, idx):
        tokens = self._get_y_hat(idx)
        tsq = TimeSeriesQuantizer()
        num_all_bins = tsq.bins_indices.shape[0]
        num_bins = tsq.num_bins
        #num_special_bins = tsq.num_special_bins

        num_tokens = tokens.shape[0]

        t = torch.zeros((num_tokens, num_all_bins)).scatter(-1, tokens.unsqueeze(-1), 1)
        normal_t = t[...,:num_bins]
        special_t = t[...,num_bins:]

        kernel = torch.tensor([[[0.05, 0.1, 1, 0.1, 0.05]]]).repeat(num_tokens, 1, 1)

        if self.soft_labels:
            normal_t = F.conv1d(normal_t.unsqueeze(0), kernel, groups=num_tokens, padding="same")
            t = torch.cat((normal_t.squeeze(0), special_t), dim=-1)
            t = t/t.sum(dim=-1, keepdim=True)
        return t

    def _get_mask(self, idx):
        l = self.raw_data[idx].length(self.split if self.inner_split else "none") - 1
        mask = torch.zeros((l), dtype=torch.bool)
        if self.objective == "ar":
            mask[-self.num_last_unmasked:] = 1
        else:
            raise NotImplementedError("MLM isn't implemented yet.")
        return mask

    def _build(self):
        if self.objective == 'ar':
            self.data = {idx: {"y": self._get_y(idx),
                               "y_hat": self._get_y_hat(idx),
                               "mask": self._get_mask(idx),
                               "y_hat_probs": self._get_y_hat_probs(idx)} for idx in range(len(self.raw_data))} 
        elif self.objective == 'mlm':
            self.data = {idx: {"y":self._get_y(idx)} for idx in range(len(self.raw_data))}

if __name__ == "__main__":
    import numpy as np
    x = np.arange(15)
    y = np.arange(15)
    ts = TimeSeries(x,y, id="test")
    qds = QDataset(ts, batch=True, soft_labels=False)
    qds.random_shifts = True
    for i in range(1):
        qds[i]
    exit()
    out = qds[0]
    print(out["y"].shape)
    print(out["y_hat"].shape)
    print(out["mask"].shape)
    print(out["i"].shape)
    dl = DataLoader(qds, batch_size=4, shuffle=False)
    for b in dl:
        print(b)