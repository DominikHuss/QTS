import abc
from preprocessing import TimeSeriesQuantizer
from dataset import QDataset, QDatasetForTransformerModels, QDatasetForHuggingFaceModels
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


class TransformerFactory(QFactory):
    def get_dataset(self, split:str, batch:bool) -> QDataset:
        return QDatasetForTransformerModels(self.ts, split=split, batch=batch)
    
    def get_container(self) -> QModelContainer:
        quant = TimeSeriesQuantizer()
        model = TransformerModel()
        return QModelContainer(model=model, quantizer=quant)


class BertFactory(QFactory):
    def get_dataset(self, split:str, batch:bool) -> QDataset:
        return QDatasetForHuggingFaceModels(self.ts, split=split, batch=batch)
    
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
        return QDatasetForHuggingFaceModels(self.ts, split=split, batch=batch)
    
    def get_container(self) -> QModelContainer:
        quant = TimeSeriesQuantizer()
        model = GPTModel(mask_token=quant.special_tokens['mask'],
                         vocab_size=len(quant.bins_indices))
        return QModelContainer(model=model, quantizer=quant)

    
FACTORIES ={"transformer": TransformerFactory,
            "bert": BertFactory,
            "gpt": GPTFactory
}