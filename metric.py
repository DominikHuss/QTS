import numpy as np
from typing import Tuple, Union
from functools import reduce
from preprocessing import QTimeSeries
from typeguard import typechecked


ASSERTION_ERROR = "Original and generated data must have to equal shape"

class QMetric():
    @typechecked
    def __init__(self,
                original:QTimeSeries,
                generated:QTimeSeries,
                soft_acc_threshold: int = 1) -> None:
        if (type(original).__name__ == QTimeSeries.__name__ 
                and type(generated).__name__ == QTimeSeries.__name__):
            self.__validate(original,generated)
        else:
            raise ValueError('Original and generated must be QTimeSeries type!')
        self.soft_acc_threshold = soft_acc_threshold
        self.accuracy = self.calc_accuracy(original, generated)
        self.soft_accuracy = self.calc_soft_accuracy(original, generated,self.soft_acc_threshold)
        self.mae = self.calc_mae(original.tokens_y,generated.tokens_y)
    
    @staticmethod
    def __validate(original: Union[QTimeSeries, np.ndarray],
                   generated: Union[QTimeSeries, np.ndarray]
    ) -> None:
        if (type(original).__name__ == QTimeSeries.__name__ 
                and type(generated).__name__ == QTimeSeries.__name__):
            assert original.tokens.shape == generated.tokens.shape, ASSERTION_ERROR 
            assert original.tokens_y.shape == generated.tokens_y.shape, ASSERTION_ERROR
        elif (isinstance(original, np.ndarray)
                and isinstance(generated, np. ndarray)):
                assert original.shape == generated.shape, ASSERTION_ERROR
                assert original.shape == generated.shape, ASSERTION_ERROR
        else:
            raise ValueError("Original and generated must be the same type!") 
    
    @staticmethod
    @typechecked
    def calc_accuracy(original: Union[QTimeSeries, np.ndarray],
                      generated: Union[QTimeSeries, np.ndarray]
    ) -> float:
        __class__.__validate(original, generated)
        if type(original).__name__ == QTimeSeries.__name__:
            original, generated = original.tokens, generated.tokens
        return (original == generated).mean()         
    
    @staticmethod
    @typechecked
    def calc_soft_accuracy(original: Union[QTimeSeries, np.ndarray],
                           generated: Union[QTimeSeries, np.ndarray],
                           threshold: int = 1
    ) -> float:
        __class__.__validate(original, generated)
        if type(original).__name__ == QTimeSeries.__name__:
            original, generated = original.tokens, generated.tokens
        return (np.abs(original - generated) <= threshold).mean()
      
    @staticmethod
    @typechecked
    def calc_mae(original: Union[QTimeSeries, np.ndarray],
                generated: Union[QTimeSeries, np.ndarray]
    ) -> float:
        __class__.__validate(original, generated)
        if type(original).__name__ == QTimeSeries.__name__:
            original, generated = original.tokens_y, generated.tokens_y 
        return np.abs(original - generated).mean()
    
    def __str__(self) -> str:
        return str({
            key: f'{val:.2f}' for key, val in zip(self.__dict__.keys(), self.__dict__.values())
        })