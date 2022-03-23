import pytest
import numpy as np
from .. import TimeSeries
from .. import TimeSeriesQuantizer


def test_timeseries():
    x = np.array([0,1,2])
    y = np.array([11,22,33])
    id = "ts_test"
    
    ts = TimeSeries(x,y,id)
    train = ts.get("train")[1]
    _eval = ts.get("eval")[1]
    test = ts.get("test")[1]
    _all = ts.get()[1]
    
    np.testing.assert_array_equal(train, np.array([11,22]))
    np.testing.assert_array_equal(_eval, np.array([]))
    np.testing.assert_array_equal(test, np.array([33]))
    np.testing.assert_array_equal(_all, y)



@pytest.mark.parametrize("expected_tokens, l_bound,u_bound",[
    (np.array([0,9,20,10,10,10,19]),None,None),
    (np.array([21,9,20,10,10,10,22]),-1000,1000)
])
def test_quantizer(expected_tokens,l_bound,u_bound):
    y = np.array([-100000,-11,0,11,22,33,100000])
    x = np.arange(y.shape[-1])
    id = "ts_test"
    ts = TimeSeries(x,y,id)
    
    quantizer = TimeSeriesQuantizer()
    quantizer.l_bound = l_bound
    quantizer.u_bound = u_bound
    qts = quantizer.quantize(ts)
    tokens = qts.tokens
    
    np.testing.assert_array_equal(tokens,expected_tokens,"Quantization bug!")
 
    
def test_batching_quantizer():
    y = np.array([-100000,-11,0,11,22,33,100000])
    x = np.arange(y.shape[-1])
    id = "ts_test"
    ts = [TimeSeries(x,y,id)]
    
    quantizer = TimeSeriesQuantizer()
    quantizer.window_length = 2
    ts_batched = quantizer.batch(ts)
    x0_batched = ts_batched[0].x
    y0_batched = ts_batched[0].y
    
    np.testing.assert_array_equal(x0_batched, np.array([0,1]),"Batching bug!")
    np.testing.assert_array_equal(y0_batched, np.array([-100000,-11]),"Batching bug!")