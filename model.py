import abc
import warnings
from typing import Union, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
from transformers import BertForMaskedLM, BertConfig
from transformers import GPT2Config, GPT2LMHeadModel

from dataset import QDataset
from criterion import SoftCrossEntropyLoss
from positional_encoding import PositionalEncoding
from preprocessing import TimeSeriesQuantizer, QTimeSeries
from decorators import parse_args
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
            self.optimizer.zero_grad()
            y_hat = self.forward(batch["y"])
            true = batch["y_hat_probs"][batch["mask"]]
            pred = y_hat[batch["mask"]]
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
        vac = ~torch.tril(torch.ones(y.shape[1], y.shape[1])).type(torch.BoolTensor).to(device=torch.device(self._global_cuda))

        o = self.module["emb"](y)
        o = self.module["pos"](o)
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
        if self._global_cuda == "cuda":
            self.cuda()


@parse_args(args_prefix="trans")
class BertModel(nn.Module, QModel):
    def __init__(self,
                 cls_token:int,
                 sep_token:int,
                 mask_token:int,
                 pad_token:int,
                 vocab_size:int,
                 *args, **kwargs) -> None:
        super().__init__()
        #TODO: add special tokens validations
        self.cls_token = torch.tensor(cls_token).unsqueeze(0)
        self.sep_token = torch.tensor(sep_token).unsqueeze(0)
        self.mask_token = torch.tensor(mask_token).unsqueeze(0)
        self.pad_token = torch.tensor(pad_token).unsqueeze(0)
        self.vocab_size = vocab_size   
    
    @typechecked
    def train_one_epoch(self, epoch, train_dataloader: DataLoader):
        epoch_loss = 0
        self.train()
        for (tokens, labels) in train_dataloader:
            loss = self.model(input_ids=tokens, labels=labels).loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss
        print(f"Epoch {epoch}: Loss = {epoch_loss/len(train_dataloader)}")

    def evaluate(self, 
                 eval_dataloader: DataLoader,
                 *,
                 epoch: int = -1,
                 _dataset: str = "EVAL"):
        self.eval()
        eval_loss = 0
        for (tokens, labels) in eval_dataloader:
            loss = self.model(input_ids=tokens, labels=labels).loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            eval_loss += loss
        if epoch >= 0:
            print(f"{_dataset} -- Epoch {epoch}: Loss = {eval_loss/len(eval_dataloader)}")
        else:
            print(f"{_dataset} -- Loss = {eval_loss/eval_dataloader}")

    @typechecked
    def predict(self, y: Union[TensorType[-1], TensorType[-1, -1]]) -> TensorType:
        self.eval()
        return F.softmax(self.model(input_ids=y.unsqueeze(0)).logits[0][-2], dim=-1).argmax(dim=-1)
        
    @typechecked
    def generate(self, 
                 time_series: Union[TensorType[-1], TensorType[-1, -1]],
                 horizon: int,
                 *,
                 stochastic) -> TensorType: 
        self.eval()
        _time_series = time_series[1:-1]
        full_time_series = _time_series.clone()
        for _ in range(horizon):
            _time_series = torch.cat((self.cls_token,
                                     _time_series,
                                     self.mask_token,
                                     self.sep_token))
            outputs = self.model(input_ids=_time_series.unsqueeze(0))
            predicted_token = outputs.logits[0][-2]
            predicted_token = torch.nn.functional.softmax(predicted_token)
            predicted_token = torch.argmax(predicted_token)
            
            #predicted_token = self.predict(_time_series)# get [MASK] prediction
            _time_series = torch.cat((_time_series[2:-2], predicted_token.unsqueeze(0)))
            full_time_series = torch.cat((full_time_series, predicted_token.unsqueeze(0)), dim=-1)
            
        return full_time_series

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    def _reset(self, *args, **kwargs):
        self.was_trained = False
        self._build()

    def _build(self):
        max_length = self._global_window_length + 3 #[CLS] tokens *[MASK] [SEP] #*only when generating
        config = BertConfig(vocab_size=self.vocab_size,
                            max_length=max_length,
                            hidden_size=self.embedding_dim,
                            num_hidden_layers=self.att_num_layers,
                            num_attention_heads=self.att_num_heads,
                            intermediate_size=self.att_feedforward_dim,
                            hidden_dropout_prob=self.dropout,
                            attention_probs_dropout_prob=self.dropout,
                            max_position_embeddings=max_length,
                            mask_token_id=self.mask_token.item(),
                            cls_token_id =self.cls_token.item(),
                            sep_token_id=self.sep_token.item(),
                            pad_token_id=self.pad_token.item())
        self.model =  BertForMaskedLM(config)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=5e-5, weight_decay=self.weight_decay)
        self.cls_token = self.cls_token.to(device=torch.device(self._global_cuda))
        self.sep_token = self.sep_token.to(device=torch.device(self._global_cuda))
        self.mask_token = self.mask_token.to(device=torch.device(self._global_cuda))
        self.pad_token = self.pad_token.to(device=torch.device(self._global_cuda))
        if self._global_cuda == "cuda":
            self.cuda()


@parse_args(args_prefix="trans")
class GPTModel(nn.Module, QModel):
    def __init__(self,
                mask_token:int,
                vocab_size: int,
                *args, **kwargs) -> None:
        super().__init__()
        self.mask_token = torch.tensor(mask_token).unsqueeze(0)
        self.vocab_size = vocab_size   
    
    @typechecked
    def train_one_epoch(self, epoch, train_dataloader: DataLoader):
        epoch_loss = 0
        self.train()
        for (tokens, labels) in train_dataloader:
            loss = self.model(input_ids=tokens, labels=labels).loss
            loss.backward()

            self.optimizer.step()
            self.optimizer.zero_grad()
            epoch_loss += loss
        print(f"Epoch {epoch}: Loss = {epoch_loss/len(train_dataloader)}")

    def evaluate(self, 
                 eval_dataloader: DataLoader,
                 *,
                 epoch: int = -1,
                 _dataset: str = "EVAL"):
        self.eval()
        eval_loss = 0
        for (tokens, labels) in eval_dataloader:
            loss = self.model(input_ids=tokens, labels=labels).loss
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()
            eval_loss += loss
        if epoch >= 0:
            print(f"{_dataset} -- Epoch {epoch}: Loss = {eval_loss/len(eval_dataloader)}")
        else:
            print(f"{_dataset} -- Loss = {eval_loss/eval_dataloader}")

    @typechecked
    def predict(self, y: Union[TensorType[-1], TensorType[-1, -1]]) -> TensorType:
        self.eval()
        return F.softmax(self.model(input_ids=y.unsqueeze(0)).logits[0][-1],dim=-1).argmax(dim=-1)
        #return F.softmax(self.model(input_ids=y.unsqueeze(0)).logits.squeeze(0), dim=-1).argmax(dim=-1)
        
    @typechecked
    def generate(self, 
                 time_series: Union[TensorType[-1], TensorType[-1, -1]],
                 horizon: int,
                 *,
                 stochastic) -> TensorType: 
        self.eval()
        _time_series = time_series.clone()
        full_time_series = _time_series.clone()
        for _ in range(horizon):
            _time_series = torch.cat((_time_series,
                                     self.mask_token))
            #predicted_token = self.predict(_time_series)
            outputs = self.model(input_ids=_time_series.unsqueeze(0))
            predicted_token = outputs.logits[0][-1]
            predicted_token = torch.nn.functional.softmax(predicted_token)
            predicted_token = torch.argmax(predicted_token)
            _time_series = torch.cat((_time_series[1:-1], predicted_token.unsqueeze(0)))
            full_time_series = torch.cat((full_time_series, predicted_token.unsqueeze(0)), dim=-1)
            
        return full_time_series

    def load(self, *args, **kwargs):
        return True

    def save(self, *args, **kwargs):
        return True

    def _reset(self, *args, **kwargs):
        self.was_trained = False
        self._build()

    def _build(self):
        max_length = self._global_window_length + 1 #tokens *[MASK] #*only when generating
        config = GPT2Config(vocab_size=self.vocab_size,
                            n_positions=max_length,
                            n_embd=self.embedding_dim,
                            n_layer=self.att_num_layers,
                            n_head=self.att_num_heads,
                            resid_pdrop=self.dropout,
                            attn_pdrop=self.dropout,
                            predict_special_tokens=False,
                            max_position_embeddings=max_length,
                            mask_token_id=self.mask_token.item(),
                            )
        self.model =  GPT2LMHeadModel(config)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.mask_token = self.mask_token.to(device=torch.device(self._global_cuda))
        if self._global_cuda == "cuda":
            self.cuda()


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
        train_dataset.soft_labels = True if self.soft_labels else False

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
            if type(self.model).__name__ == TransformerModel.__name__:#isinstance(self.model, type(TransformerModel)): why doesn't work properly?
                window_length = self._global_window_length - self.num_last_unmasked
            
            elif type(self.model).__name__ == BertModel.__name__:
                window_length = self._global_window_length + 2 #[CLS] tokens [SEP]
            else:
                window_length = self._global_window_length
        
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
        elif issubclass(type(y), QDataset):
            if id is None:
                raise Exception("When used with a QDataset id parameter must be set!")
            (qts, _y), _, _ = y.get_batched(id, _all=True)
            if _y is None:
                warnings.warn("Nothing was generated.")
                return None
            if horizon is None:
                horizon = int(y.get_unbatched(id).length()-window_length)
            _y = _y[:window_length]
        y_hat = self.model.generate(_y, horizon=horizon, stochastic=stochastic).to(device=torch.device(self._global_cuda))

        if isinstance(y, torch.Tensor):
            return y_hat
        qts.tokens_y = torch.take(torch.from_numpy(
            self.quantizer.bins_values).to(device=torch.device(self._global_cuda)),
            y_hat).cpu().numpy()
        qts.tokens = y_hat.cpu().numpy()
        return qts

    def _build(self):
        pass