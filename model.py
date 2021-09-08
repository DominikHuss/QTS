from datetime import time
from torch.utils.data import dataloader
from dataset import QDataset
from torch import optim
from torch.utils.data.dataloader import DataLoader
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F

import warnings
from typing import Union, Iterable, List, Literal
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

class QModel(abc.ABC):
    def __init__(self) -> None:
        super().__init__()
        self.was_trained = False

    @abc.abstractmethod
    def train_one_epoch(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def evaluate(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def load(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def save(self, *args, **kwargs):
        raise NotImplementedError()

    @abc.abstractmethod
    def _reset(self, *args, **kwargs):
        raise NotImplementedError()

@parse_args(args_prefix="ngram")
class NgramModel(nn.Module, QModel):
    def train_one_epoch(self, *args, **kwargs):
        return True

    def evaluate(self, *args, **kwargs):
        return True

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    def _reset(self, *args, **kwargs):
        return True

    def _build(self):
        print(self.a)
        self.l = nn.Linear(1,1)

@parse_args(args_prefix="trans")
class TransformerModel(nn.Module, QModel):
    @typechecked
    def train_one_epoch(self, epoch, train_dataloader: DataLoader):
        epoch_loss = 0
        self.train()
        for batch in train_dataloader:
            print(batch)
            self.optimizer.zero_grad()
            y_hat = self.forward(batch["y"])

            true = batch["y_hat"][batch["mask"]]
            pred = y_hat[batch["mask"]]
            #print(true.shape)
            #print(pred.shape)
            loss = self.criterion(pred, true)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        print(f"Epoch {epoch}: Loss = {epoch_loss}")

    @typechecked
    def predict(self, 
                y: Union[TensorType[-1], TensorType[-1, -1]],
                *,
                stochastic: bool = False) -> TensorType:
        self.eval()
        if not stochastic:
            _y = y.unsqueeze(0) if len(y.shape) == 1 else y
            _y = F.softmax(self.forward(_y), dim=-1).argmax(dim=-1)
            _y = _y.squeeze(0) if len(y.shape) == 1 else _y
            return _y
        else:
            if y.shape[0] != 1 and len(y.shape) == 2:
                raise Exception("Cannot process batched data!")
            _y = y.unsqueeze(0) if len(y.shape) == 1 else y
            _y = torch.multinomial(F.softmax(self.forward(_y), dim=-1).squeeze(0), 1).squeeze(-1)
            _y = _y.unsqueeze(0) if len(y.shape) == 2 else _y
            return _y


    def evaluate(self, 
                 eval_dataloader: DataLoader,
                 *,
                 epoch: int = -1):
        self.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            y_hat = self.forward(batch["y"])
            true = batch["y_hat"][batch["mask"]]
            pred = y_hat[batch["mask"]]
            loss = self.criterion(pred, true)
            eval_loss += loss
        if epoch >= 0:
            print(f"EVAL -- Epoch {epoch}: Loss = {eval_loss}")
        else:
            print(f"EVAL -- Loss = {eval_loss}")

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    @typechecked
    def generate(self, 
                 time_series: Union[TensorType[-1], TensorType[-1, -1]], 
                 horizon=50,
                 *,
                 stochastic: bool = False) -> TensorType:
        if len(time_series.shape) == 2:
            pass
        elif len(time_series.shape) == 1:
            time_series = time_series.unsqueeze(0)
        else:
            raise Exception("Wrong dimensions")

        self.eval()
        full_time_series = time_series.clone()
        for _ in range(horizon):
            time_series = torch.cat((time_series[:, 1:], self.predict(time_series, stochastic=stochastic)[:, -1:]), dim=-1)
            #print(time_series)
            full_time_series = torch.cat((full_time_series, time_series[:, -1:]), dim=-1)
        full_time_series = full_time_series.squeeze(0)
        #print(full_time_series)
        return full_time_series


    def forward(self, y):
        vac = ~torch.tril(torch.ones(y.shape[1], y.shape[1])).type(torch.BoolTensor)

        o = self.module["emb"](y)
        o = self.module["trans"](o, mask=vac)
        o = self.module["out_proj"](o)
        return o

    def _reset(self, *args, **kwargs):
        self.was_trained = False
        self._build()


    def _build(self):
        embedding = nn.Embedding(self.num_embedding, 
                                 self.embedding_dim*self.att_num_heads)
        transformer_encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=self.embedding_dim*self.att_num_heads, 
                                                                               nhead=self.att_num_heads, 
                                                                               dim_feedforward=self.att_feedforward_dim, 
                                                                               dropout=self.dropout, 
                                                                               batch_first=True), 
                                                    num_layers=self.att_num_layers, 
                                                    norm=nn.LayerNorm(normalized_shape=self.embedding_dim*self.att_num_heads))
        output_proj = nn.Linear(in_features=self.embedding_dim*self.att_num_heads, 
                                out_features=self.num_embedding)
        self.module = nn.ModuleDict({"emb": embedding,
                                      "trans": transformer_encoder,
                                      "out_proj": output_proj})
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = nn.CrossEntropyLoss()

@parse_args(args_prefix="qmc")
class QModelContainer():
    def __init__(self, model: QModel):
        super().__init__()
        self.model = model

    @typechecked
    def train(self,
              train_dataset: QDataset,
              eval_dataset: QDataset = None,
              *,
              _reset: bool = False):
        if _reset: self.model._reset()

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=self.shuffle) if eval_dataset is not None else train_dataloader
        for epoch in range(self.num_epochs):
            self.model.train_one_epoch(epoch, train_dataloader)
            if epoch % self.eval_epoch == 0:
                self.model.evaluate(eval_dataloader, epoch=epoch)
        self.model.was_trained = True

    @typechecked
    def generate(self,
                 y: Union[TensorType[-1], TensorType[-1, -1]],
                 stochastic: bool = False):
        if not self.model.was_trained:
            warnings.warn("The model was not trained, or was reset!")
        y_hat = self.model.generate(y, stochastic=stochastic)
        return y_hat

    def _build(self):
        pass


if __name__ == "__main__":
    import numpy as np
    x = np.arange(30)
    y = np.sin(np.arange(30))+2
    #print(y)

    ts = TimeSeries(x,y)
    #train_qds = QDataset(ts, batch=True)
    train_qds = QDataset(ts, split="train", batch=True)
    #eval_qds = QDataset(ts, split="eval", batch=True)
    #test_qds = QDataset(ts, split="test", batch=True)

    trans = TransformerModel()
    qmc = QModelContainer(trans)
    
    qmc.train(train_qds)
    print(qmc.generate(train_qds[0]["y"]))
    print(torch.tensor(QDataset(ts, split="train", batch=False).raw_data[0].tokens))
    print(torch.tensor(QDataset(ts, batch=False).raw_data[0].tokens))
    #print(y)
