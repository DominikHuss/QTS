import abc
from preprocessing import TimeSeriesQuantizer
from dataset import QDataset, QDatasetForTransformerARModel, QDatasetForTransformerMLMModel, QDatasetForBertModel, QDatasetForGPTModel
from model import TransformerModel, BertModel, GPTModel, QModelContainer

class QFactory(abc.ABC):
    def __init__(self, ts) -> None:
        self.ts = ts
    @abc.abstractmethod
    def get_dataset(self, *args, **kwargs) -> QDataset:
        """Get Qdataset"""
        
    @abc.abstractmethod
    def get_container(self, *args, **kwargs) -> QModelContainer:
        """Get QModelContainer"""


class TransformerARFactory(QFactory):
    def get_dataset(self, split:str, batch:bool) -> QDataset:
        return QDatasetForTransformerARModel(self.ts, split=split, batch=batch)
    
    def get_container(self) -> QModelContainer:
        quant = TimeSeriesQuantizer()
        model = TransformerModel(objective="ar")
        return QModelContainer(model=model, quantizer=quant)

class TransformerMLMFactory(QFactory):
    def get_dataset(self, split:str, batch:bool) -> QDataset:
        return QDatasetForTransformerMLMModel(self.ts, split=split, batch=batch)
    
    def get_container(self) -> QModelContainer:
        quant = TimeSeriesQuantizer()
        model = TransformerModel(objective="mlm", mask_token=quant.special_tokens['mask'])
        return QModelContainer(model=model, quantizer=quant)


class BertFactory(QFactory):
    def get_dataset(self, split:str, batch:bool) -> QDataset:
        return QDatasetForBertModel(self.ts, split=split, batch=batch)
    
    def get_container(self) -> QModelContainer:
        quant = TimeSeriesQuantizer()
        model = BertModel(cls_token=quant.special_tokens['cls'],
                          sep_token=quant.special_tokens['sep'],
                          mask_token=quant.special_tokens['mask'],
                          pad_token=quant.special_tokens['pad'],
                          vocab_size=len(quant.bins_indices))
        return QModelContainer(model=model, quantizer=quant)
   
class GPTFactory(QFactory):
    def get_dataset(self, split:str, batch:bool) -> QDataset:
        return QDatasetForGPTModel(self.ts, split=split, batch=batch)
    
    def get_container(self) -> QModelContainer:
        quant = TimeSeriesQuantizer()
        model = GPTModel(vocab_size=len(quant.bins_indices))
        return QModelContainer(model=model, quantizer=quant)

    
FACTORIES ={"transformer_ar": TransformerARFactory,
            "transformer_mlm": TransformerMLMFactory,
            "bert": BertFactory,
            "gpt": GPTFactory
}