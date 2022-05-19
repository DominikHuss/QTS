import torch
import torch.nn as nn
import math
from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()


class PositionalEncoding(nn.Module):
    """
    Class representing postional encoding using by `TransformerAR` and `TransformerMLM` model.
        It is default implementation of positional encoding from official pytorch tutorial.
    :param int d_model: Dimension of embeddings"
    :param int max_len: Length of positional encoding, It should be equal to window length.
    """
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x.permute(1,0,2)
        x = x + self.pe[:x.size(0)]
        return x.permute(1,0,2)