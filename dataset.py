import warnings
import copy
from typing import Tuple, Union, List, Literal, Optional
import abc

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args
patch_typeguard()


class QDataset(abc.ABC, Dataset):
    @abc.abstractmethod
    def get_unbatched(self, *args, **kwargs) -> Optional[TimeSeries]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def get_batched(self, *args, **kwargs) -> Union[Tuple[Tuple[Optional[QTimeSeries], 
                                                     Optional[TensorType[-1]]], 
                                                     int, 
                                                     int], 
                                         QTimeSeries]:
        raise NotImplementedError()
    
    @abc.abstractmethod
    def __len__(self):
        raise NotImplementedError()
    
        
class QDatasetBase(QDataset):
    @typechecked
    def __init__(self, 
                 _data: Union[np.ndarray, List[np.ndarray], TimeSeries, List[TimeSeries], QTimeSeries, List[QTimeSeries]],
                 split: Literal["train", "eval", "test", "none"] = "none",
                 *,
                 batch: bool = False) -> None:
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
        self.mlm_masked_probability = 0.15 #TO DO: add as args add as param
        self.mlm_non_masked_value = -100 #TO DO: add as args add as param
        self.mlm_masked_token_prob = 0.8 #TO DO: add as args add as param
        self.mlm_random_token_prob = 0.1 #TO DO: add as args add as param
        assert self.mlm_masked_token_prob + self.mlm_random_token_prob <= 1
        if self.mlm_masked_token_prob + self.mlm_random_token_prob == 1:
            warnings.warn("Probability of masked inputs with token [MASK] and probability of random replace equals 1" +
                      "None of the inputs tokens will be unchanged. It will cause invalid training model!") 
        self.data = None
    
    @typechecked
    def get_unbatched(self,
                      id: str,
                      *,
                      _quantized: bool = False) -> Optional[Union[TimeSeries, QTimeSeries]]:
        if self.raw_unbatched_data is None:
            warnings.warn("Initializing a QDataset with QTimeSeries holds not enough information to dissambiguate whether data are batched already. Returning None")
            return None

        if _quantized and not self.batch:
            ts_data = self.raw_data
        elif _quantized and self.batch:
            #warnings.warn("Only works for window_step = 1")
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
                    qts = copy.deepcopy(self.raw_data[i])
                    prepped_y = self.data[i]["y"] #Exception?
            else:
                if start != -1:
                    break
        if prepped_y is None:
            warnings.warn(f"No mini-batches of a QTimeSeries with matching id='{id}' were found in the QDataset! Returning None")
        if isinstance(prepped_y, np.ndarray):
            prepped_y = torch.from_numpy(prepped_y).to(device=self._global_cuda)
        return (qts, prepped_y), start, length
            
    def __len__(self):
        return len(self.data)




@parse_args(args_prefix="qds")
class QDatasetForTransformerARModel(QDatasetBase):
    @typechecked
    def __init__(self, 
                 _data: Union[np.ndarray, List[np.ndarray], TimeSeries, List[TimeSeries], QTimeSeries, List[QTimeSeries]],
                 split: Literal["train", "eval", "test", "none"] = "none",
                 *,
                 batch: bool = False,
                 soft_labels: bool = False,
                 random_shifts: bool = False) -> None:
        super().__init__(_data, split, batch= batch)
        self.soft_labels = soft_labels
        self.random_shifts = random_shifts  
        
    def __getitem__(self, idx):
        y = self.data[idx]["y"]
        if self.random_shifts:
            y = self._random_shifts(y)
        y_hat = self.data[idx]["y_hat"]
        y_hat_probs = self.data[idx]["y_hat_probs"]
        mask = self.data[idx]["mask"]
        qts_idx = torch.tensor([idx]) 
        return {"y": y, "y_hat": y_hat, "y_hat_probs": y_hat_probs, "mask": mask, "idx": qts_idx}
    
    def _random_shifts(self,y:torch.Tensor):
        shifts_idx = torch.tensor([0.1, 0.8, 0.1]).multinomial(num_samples=y.shape[0], replacement=True)
        shifts = torch.tensor([-1, 0, 1])[shifts_idx]
        specials_mask = y>=self.tsq.num_bins
        y = F.relu(y+~specials_mask*shifts)
        specials_mask_check = y>=self.tsq.num_bins
        y[torch.logical_xor(specials_mask, specials_mask_check)] -= 1
        return y
              
    def _get_y(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        tokens = tokens[:-1]
        return torch.from_numpy(tokens).to(device=torch.device(self._global_cuda))

    def _get_y_hat(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        tokens = tokens[1:]
        return torch.from_numpy(tokens).to(device=torch.device(self._global_cuda))

    def _get_y_hat_probs(self, idx):
        tokens = self._get_y_hat(idx)
        tsq = TimeSeriesQuantizer()
        num_all_bins = tsq.bins_indices.shape[0]
        num_bins = tsq.num_bins
        num_tokens = tokens.shape[0]

        t = torch.zeros((num_tokens, num_all_bins),device=torch.device(self._global_cuda)).scatter(-1, tokens.unsqueeze(-1), 1)
        normal_t = t[...,:num_bins]
        special_t = t[...,num_bins:]

        kernel = torch.tensor([[[0.05, 0.1, 1, 0.1, 0.05]]]).repeat(num_tokens, 1, 1)

        if self.soft_labels:
            normal_t = F.conv1d(normal_t.unsqueeze(0), kernel, groups=num_tokens, padding="same")
            t = torch.cat((normal_t.squeeze(0), special_t), dim=-1)
            t = t/t.sum(dim=-1, keepdim=True)
        return t.to(device=torch.device(self._global_cuda))

    def _get_mask(self, idx):
        l = self.raw_data[idx].length(self.split if self.inner_split else "none") - 1
        mask = torch.zeros((l), dtype=torch.bool)
        mask[-self.num_last_unmasked:] = 1
        return mask.to(device=torch.device(self._global_cuda))

    def _build(self):
        self.data = {idx: {"y": self._get_y(idx),
                               "y_hat": self._get_y_hat(idx),
                               "mask": self._get_mask(idx),
                               "y_hat_probs": self._get_y_hat_probs(idx)} for idx in range(len(self.raw_data))} 


@parse_args(args_prefix="qds")
class QDatasetForTransformerMLMModel(QDatasetBase):
    @typechecked
    def __init__(self, 
                 _data: Union[np.ndarray, List[np.ndarray], TimeSeries, List[TimeSeries], QTimeSeries, List[QTimeSeries]],
                 split: Literal["train", "eval", "test", "none"] = "none",
                 *,
                 batch: bool = False,
                 soft_labels: bool = False,
                 random_shifts: bool = False) -> None:
        super().__init__(_data, split, batch= batch)
        self.soft_labels = soft_labels
        self.random_shifts = random_shifts  
        
    def __getitem__(self, idx):
        return {"y": self._get_mlm(idx),
                "y_hat_probs": self.data[idx]["y_hat_probs"],
                "mask": self.mask,
                "mask_attention": self.mask_attention,
                "idx": torch.tensor([idx])}
    
    def _get_mlm(self, idx):
        y = self.data[idx]['y'].clone()
        masked_tokens_idx = torch.bernoulli(torch.full(y.shape, self.mlm_masked_token_prob)).bool().to(device=torch.device(self._global_cuda)) & self.mask
        y[masked_tokens_idx] = self.tsq.special_tokens['mask']
        random_tokens_idx = torch.bernoulli(torch.full(y.shape, self.mlm_random_token_prob)).bool().to(device=torch.device(self._global_cuda)) & self.mask & ~masked_tokens_idx
        random_tokens = torch.randint(self.tsq.num_bins, y.shape, dtype=torch.long, device=torch.device(self._global_cuda))
        y[random_tokens_idx] = random_tokens[random_tokens_idx]
        return y
    
    def _get_y(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        return torch.from_numpy(tokens).to(device=torch.device(self._global_cuda))
    
    def _get_y_hat_probs(self, idx):
        tokens = self._get_y(idx)
        tsq = TimeSeriesQuantizer()
        num_all_bins = tsq.bins_indices.shape[0]
        num_bins = tsq.num_bins
        num_tokens = tokens.shape[0]

        t = torch.zeros((num_tokens, num_all_bins),device=torch.device(self._global_cuda)).scatter(-1, tokens.unsqueeze(-1), 1)
        normal_t = t[...,:num_bins]
        special_t = t[...,num_bins:]

        kernel = torch.tensor([[[0.05, 0.1, 1, 0.1, 0.05]]]).repeat(num_tokens, 1, 1)

        if self.soft_labels:
            normal_t = F.conv1d(normal_t.unsqueeze(0), kernel, groups=num_tokens, padding="same")
            t = torch.cat((normal_t.squeeze(0), special_t), dim=-1)
            t = t/t.sum(dim=-1, keepdim=True)
        return t.to(device=torch.device(self._global_cuda))
    
    def __get_mask(self):
        probability_matrix = torch.full((self._global_window_length,),
                                        self.mlm_masked_probability,
                                        device=torch.device(self._global_cuda))
        repeat = True
        while repeat: 
            mask =torch.bernoulli(probability_matrix).bool().to(device=torch.device(self._global_cuda))
            if mask.any():
                repeat = False
        return mask
    def __get_mask_attention(self):
        mask_attention = torch.zeros((self.mask.shape[-1], self.mask.shape[-1]),
                                dtype=torch.bool,
                                device=torch.device(self._global_cuda)) # SxS
        masked_positions = (self.mask == True).nonzero(as_tuple=True)[0]
        for pos in masked_positions:
            mask_attention[:,pos] = True
            mask_attention[pos, pos] = False
        return mask_attention

    
    def _build(self):
        self.data = {idx:{"y": self._get_y(idx),
                          "y_hat_probs": self._get_y_hat_probs(idx)
                        } for idx in range(len(self.raw_data))}
        
        self.mask = self.__get_mask()
        self.mask_attention = self.__get_mask_attention() 

@parse_args(args_prefix="qds")
class QDatasetForHuggingFaceModels(QDatasetBase):
    @typechecked
    def __init__(self, 
                 _data: Union[np.ndarray, List[np.ndarray], TimeSeries, List[TimeSeries], QTimeSeries, List[QTimeSeries]],
                 split: Literal["train", "eval", "test", "none"] = "none",
                 *,
                 batch: bool = False) -> None:
        super().__init__(_data, split, batch= batch)
       
    def __getitem__(self, idx):
        if self.objective == "mlm":
           tokens, labels = self._get_mlm(idx)
           true = torch.from_numpy(self.data[idx]['y'])
           return (torch.from_numpy(tokens).to(device=self._global_cuda), torch.from_numpy(labels).to(device=self._global_cuda), true)
        elif self.objective == "ar":
            return self._get_ar(idx)
    
    def _get_mlm(self, idx):
        tokens, labels = self.data[idx]['y'].copy(), self.data[idx]['y'].copy()
        special_tokens_matrix = (tokens >= self.tsq.num_bins)
        probability_matrix = np.ma.masked_array(np.full(tokens.shape, self.mlm_masked_probability), special_tokens_matrix)
        probability_matrix = probability_matrix.filled(fill_value=0.0)
        masked_idx = np.random.binomial(n=1,p=probability_matrix).astype(bool)
        labels[~masked_idx] = self.mlm_non_masked_value
        masked_tokens_idx = np.random.binomial(n=1,p=np.full(tokens.shape, self.mlm_masked_token_prob)).astype(bool) & masked_idx
        tokens[masked_tokens_idx] = self.tsq.special_tokens['mask']
        random_tokens_idx = np.random.binomial(n=1, p=np.full(tokens.shape, self.mlm_random_token_prob)).astype(bool) & masked_idx & ~masked_tokens_idx
        random_tokens = np.random.randint(0, self.tsq.num_bins, tokens.shape)
        tokens[random_tokens_idx] = random_tokens[random_tokens_idx]
        return (tokens, labels)
    
    def _get_ar(self, idx):
        return (self.data[idx]['y'].clone(), self.data[idx]['y'].clone())
                
    
    def _get_y(self, idx):
        tokens, _, _, _ = self.raw_data[idx].get(self.split if self.inner_split else "none")
        if self._global_model == "bert":
            tokens = np.concatenate(([self.tsq.special_tokens['cls']],
                                        tokens,
                                        [self.tsq.special_tokens['sep']]))
        return tokens#torch.from_numpy(tokens).to(device=torch.device(self._global_cuda))
    
    def _build(self):
        self.data = {idx: {"y":self._get_y(idx)} for idx in range(len(self.raw_data))}
        
        
"""
1)
def __get_mask_attention(self):
mask_attention = torch.zeros((self.mask.shape[-1], self.mask.shape[-1]),
                                dtype=torch.bool,
                                device=torch.device(self._global_cuda)) # SxS
masked_positions = (self.mask == True).nonzero(as_tuple=True)[0]
for pos in masked_positions:
    mask_attention[:,pos] = True
    mask_attention[pos, pos] = False
return mask_attention
2)
 return self.mask.repeat(self.mask.shape[-1], 1)


3)tokens, labels = self.data[idx]['y'].clone(), self.data[idx]['y'].clone()
        special_tokens_matrix = tokens >= self.tsq.num_bins  
        probability_matrix = torch.full(tokens.shape, self.mlm_masked_probability,device=torch.device(self._global_cuda))
        probability_matrix.masked_fill_(special_tokens_matrix, value=0.0)
        masked_idx = torch.bernoulli(probability_matrix).bool().to(device=torch.device(self._global_cuda))
        labels[~masked_idx] = self.mlm_non_masked_value
        masked_tokens_idx = torch.bernoulli(torch.full(tokens.shape, self.mlm_masked_token_prob)).bool().to(device=torch.device(self._global_cuda)) & masked_idx
        tokens[masked_tokens_idx] = self.tsq.special_tokens['mask']
        random_tokens_idx = torch.bernoulli(torch.full(tokens.shape, self.mlm_random_token_prob)).bool().to(device=torch.device(self._global_cuda)) & masked_idx & ~masked_tokens_idx
        random_tokens = torch.randint(self.tsq.num_bins, tokens.shape, dtype=torch.long, device=torch.device(self._global_cuda))
        tokens[random_tokens_idx] = random_tokens[random_tokens_idx]
        return (tokens, labels)
"""