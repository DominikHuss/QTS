from plot import Plotter
from _synthetic_data_generation import random_peaks, random
from dataset import QDataset
from criterion import SoftCrossEntropyLoss
from positional_encoding import PositionalEncoding
from preprocessing import TimeSeriesQuantizer, TimeSeries, QTimeSeries
from decorators import parse_args

import abc

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

import warnings
from typing import Union, Iterable, List, Literal, Optional
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
            #print(batch)
            self.optimizer.zero_grad()
            y_hat = self.forward(batch["y"])

            #true = batch["y_hat"][batch["mask"]]
            true = batch["y_hat_probs"][batch["mask"]]
            pred = y_hat[batch["mask"]]
            #print(true.shape)
            #print(pred.shape)
            loss = self.criterion(pred, true)
            loss.backward()
            self.optimizer.step()
            epoch_loss += loss
        print(f"Epoch {epoch}: Loss = {epoch_loss/len(train_dataloader)}")

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
                 epoch: int = -1,
                 _dataset: str = "EVAL"):
        self.eval()
        eval_loss = 0
        for batch in eval_dataloader:
            y_hat = self.forward(batch["y"])
            #true = batch["y_hat"][batch["mask"]]
            true = batch["y_hat_probs"][batch["mask"]]
            pred = y_hat[batch["mask"]]
            loss = self.criterion(pred, true)
            eval_loss += loss
        if epoch >= 0:
            print(f"{_dataset} -- Epoch {epoch}: Loss = {eval_loss/len(eval_dataloader)}")
        else:
            print(f"{_dataset} -- Loss = {eval_loss/eval_dataloader}")

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    @typechecked
    def generate(self, 
                 time_series: Union[TensorType[-1], TensorType[-1, -1]], 
                 horizon=150,
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
        #o = self.module["pos"](o)
        o = self.module["trans"](o, mask=vac)
        o = self.module["out_proj"](o)
        return o

    def _reset(self, *args, **kwargs):
        self.was_trained = False
        self._build()


    def _build(self):
        embedding = nn.Embedding(self.num_embedding, 
                                 self.embedding_dim*self.att_num_heads)
        pos = PositionalEncoding(d_model=self.embedding_dim*self.att_num_heads,
                                 dropout=self.pos_dropout,
                                 max_len=self.pos_max_len)
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
                                     "pos": pos,
                                     "trans": transformer_encoder,
                                     "out_proj": output_proj})
        self.optimizer = torch.optim.Adam(self.module.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.criterion = SoftCrossEntropyLoss
        #w = torch.ones(self.num_embedding)
        #w[-3] = 0.1
        #self.criterion = nn.CrossEntropyLoss(weight=w)

@parse_args(args_prefix="qmc")
class QModelContainer():
    def __init__(self, 
                 model: QModel,
                 quantizer: TimeSeriesQuantizer):
        super().__init__()
        self.model = model
        self.quantizer = quantizer

    @typechecked
    def train(self,
              train_dataset: QDataset,
              eval_dataset: QDataset = None,
              *,
              _reset: bool = False):
        if _reset: self.model._reset()

        train_dataset.random_shifts = True if self.random_shifts else False
        #train_dataset.soft_labels = True if self.soft_labels else False

        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        eval_dataloader = DataLoader(eval_dataset, batch_size=self.batch_size, shuffle=self.shuffle) if eval_dataset is not None else train_dataloader
        for epoch in range(self.num_epochs):
            self.model.train_one_epoch(epoch, train_dataloader)
            if epoch % self.eval_epoch == 0:
                self.model.evaluate(eval_dataloader, epoch=epoch)
        self.model.was_trained = True

    @typechecked
    def test(self,
             test_dataset: QDataset):
        test_dataloader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=self.shuffle)
        self.model.evaluate(test_dataloader, _dataset="TEST")

    @typechecked
    def generate(self,
                 y: Union[QDataset, QTimeSeries, TensorType[-1], TensorType[-1, -1]],
                 horizon: int = None,
                 *,
                 window_length: int = None,
                 stochastic: bool = False,
                 id: str = None) -> Optional[Union[QTimeSeries, TensorType[-1], TensorType[-1, -1]]]:
        if not self.model.was_trained:
            warnings.warn("The model was not trained, or was reset!")

        if window_length is None:
            window_length = self.window_length - self.num_last_unmasked


        if isinstance(y, torch.Tensor):
            if y.shape[-1] < window_length:
                warnings.warn("When using a torch.Tensor as an input make sure that it is of sufficient length w.r.t. window length! Nothing was generated, and None was returned!")
                return None
            if horizon is None:
                raise Exception("Horizon not provided when supplying torch.Tensor!")
            _y = y[:window_length]
        elif type(y).__name__ == QTimeSeries.__name__:
            if y.tokens.shape[-1] < window_length:
                warnings.warn("When using a QTimeSeries as an input make sure that it is of sufficient length w.r.t. window length! Nothing was generated, and None was returned!")
                return None
            if horizon is None:
                raise Exception("Horizon not provided when supplying QTimeSeries!")
            qts = y
            _y = torch.tensor(y.tokens)
            _y = _y[:window_length]
        elif type(y).__name__ == QDataset.__name__:

            if id is None:
                raise Exception("When used with a QDataset id parameter must be set!")

            (qts, _y), _, _ = y.get_batched(id, _all=True)
            if _y is None:
                warnings.warn("Nothing was generated.")
                return None

            if horizon is None:
                horizon = int(y.get_unbatched(id).length()-window_length)
            _y = _y[:window_length]
        y_hat = self.model.generate(_y, horizon=horizon, stochastic=stochastic)

        if isinstance(y, torch.Tensor):
            return y_hat
        
        # TODO: return QTimeSeries
        qts.tokens_y = torch.take(torch.from_numpy(self.quantizer.bins_values), y_hat).numpy()
        qts.tokens = y_hat.numpy()
        #print(qts.tokens)
        #print(qts.tokens_y)
        return qts


    def _build(self):
        pass


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import numpy as np

    np.random.seed(0)

    trans = TransformerModel()
    quant = TimeSeriesQuantizer()
    qmc = QModelContainer(trans, quant)
    #exit()

    l=300
    soft_labels = qmc.soft_labels

    x = np.arange(l)

    #y = np.sin(np.arange(l))
    #y = np.concatenate((np.sin(np.arange(l//2)), np.sin(np.arange(l//2, l))-1))
    #y = random(l, 1, 1)
    y = random_peaks(l, 1, 1, 0.4)
    z = np.cos(np.arange(l))

    plot = Plotter(TimeSeriesQuantizer(), "plots/")

    ts = TimeSeries(x,y, id="sin")
    tz = TimeSeries(x,z, id="cos")
    
    #ts = [ts, tz]
    all_qds = QDataset(ts, batch=True)
    train_qds = QDataset(ts, split="train", batch=True, soft_labels=soft_labels)
    eval_qds = QDataset(ts, split="eval", batch=True, soft_labels=soft_labels)
    #test_qds = QDataset(ts, split="test", batch=True)
    print(train_qds.raw_data[0].tokens)
    print(train_qds.raw_data[0].tokens_y)
    print(train_qds.raw_data[0].unnormalize())
    print(train_qds.raw_data[0].ts.y)

    qmc.train(train_qds, eval_qds)
    if qmc._global_SMOKE_TEST:
        exit()

    print(qmc.generate(train_qds, id="sin"))
    plot.plot(ts, label="true")
    plot.plot(train_qds.get_unbatched("sin"), label="train")
    plot.plot(QDataset(ts, split="eval", batch=False).raw_data[0].ts, label="eval")
    plot.plot(qmc.generate(train_qds, id="sin", horizon=int(train_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked))))
    plot.save("train.png")
    print(qmc.generate(train_qds, id="cos"))
    print(qmc.generate(train_qds.raw_data[0], horizon=150))
    print(qmc.generate(train_qds[0]["y"], horizon=150))
    #print(qmc.generate(torch.tensor([5,9,9])))
    #print(qmc.generate(torch.tensor([[5,9,9]])))

    print(qmc.generate(QDataset(ts)[0]["y"], window_length=7, horizon=150))
    print(torch.tensor(QDataset(ts, split="train", batch=False).raw_data[0].tokens))
    print(torch.tensor(QDataset(ts, batch=False).raw_data[0].tokens))
    for t in QDataset(ts, batch=False).raw_data:
        print(t.tokens)

    eval_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    eval_plot.plot(all_qds.get_unbatched("sin"), label="true")
    eval_plot.plot(train_qds.get_unbatched("sin"), label="train")
    eval_plot.plot(eval_qds.get_unbatched("sin"),  label="eval")
    eval_plot.plot(qmc.generate(eval_qds, id="sin", horizon=int(eval_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked))))
    eval_plot.save("eval.png")

    print("--------")
    num_samples = 100
    # all
    train_random_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    for _ in range(num_samples):
        train_random_plot.plot(qmc.generate(train_qds, 
                                            id="sin", 
                                            stochastic=True, 
                                            horizon=int(train_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked))), 
                                            alpha=1/num_samples)
    train_random_plot.save("random_train.png")
    # eval
    eval_random_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    for _ in range(num_samples):
        eval_random_plot.plot(qmc.generate(eval_qds, 
                                           id="sin", 
                                           horizon=int(eval_qds.get_unbatched("sin").length()-(qmc.window_length-qmc.num_last_unmasked)), 
                                           stochastic=True), 
                                           alpha=1/num_samples)
    eval_random_plot.save("random_eval.png")

    cos_plot = Plotter(TimeSeriesQuantizer(), "plots/")
    cos_plot.plot(tz, label="true")
    cos_plot.plot(qmc.generate(QDataset(tz, split="train"), id="cos"))
    cos_plot.save("cos_test.png")
