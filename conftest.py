import pytest
import numpy as np

from preprocessing import TimeSeries, TimeSeriesQuantizer

@pytest.fixture
def get_ndarray():
    return np.array([-200, -100, 0, 100, 200])

@pytest.fixture
def get_ts(get_ndarray):
    return TimeSeries(np.arange(len(get_ndarray)), get_ndarray, "0")

@pytest.fixture
def get_quantizer():
    return TimeSeriesQuantizer()

@pytest.fixture
def get_qts(get_quantizer, get_ts):
    quant = get_quantizer
    ts = get_ts
    return quant.quantize(ts)
   