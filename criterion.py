from plot import Plotter
from dataset import QDataset
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import warnings
from typing import Union, Iterable, List, Literal, Optional
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()



def SoftCrossEntropyLoss(y_hat, labels):
        logprobs = torch.nn.functional.log_softmax(y_hat, dim=-1)
        return -(labels*logprobs).sum()/y_hat.shape[0]


if __name__ == "__main__":
    torch.manual_seed(2020)

    criterion = SoftCrossEntropyLoss
    ref = nn.CrossEntropyLoss()

    y = torch.rand((4, 2, 10))
    # target values are "soft" probabilities that sum to one (for each sample in batch)
    target = torch.nn.functional.softmax(torch.randn((4, 2, 10)), dim=-1)
    _, target_cat = target.max(dim=-1)
    target_onehot = torch.zeros_like(target).scatter(-1, target_cat.unsqueeze(-1), 1)

    print(target)
    print(target_cat)
    print(target_onehot)

    print(criterion(y, target))
    print(y.view(y.shape[0], -1).shape)
    print(target_cat.shape)
    #print(ref(y.view(y.shape[0], -1), target_cat.view(-1)))
    print(criterion(y, target_onehot))