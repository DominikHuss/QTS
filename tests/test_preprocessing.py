import pytest
import numpy as np

from preprocessing import TimeSeries
from preprocessing import TimeSeriesQuantizer


def test_timeseries(get_ndarray, get_ts):
    # x = np.array([0,1,2])
    # y = np.array([11,22,33])
    # id = "ts_test"
    
    ts = get_ts#TimeSeries(x,y,id)
    train = ts.get("train")[1]
    _eval = ts.get("eval")[1]
    test = ts.get("test")[1]
    _all = ts.get()[1]
    
    np.testing.assert_array_equal(train, np.array([-200, -100, 0]))
    np.testing.assert_array_equal(_eval, np.array([100]))
    np.testing.assert_array_equal(test, np.array([200]))
    np.testing.assert_array_equal(_all, get_ndarray)



@pytest.mark.parametrize("expected_tokens, l_bound,u_bound",[
    (np.array([0,1,5,3,4]),None,None),
    (np.array([0,1,5,3,4]),-1000,1000),
    (np.array([0,1,5,3,4]),-200,200),
    (np.array([6,1,5,3,7]),-150,150)])
def test_quantizer(expected_tokens,l_bound,u_bound):
    y = np.array([-200, -100, 0, 100, 200])#np.array([-100,-11,0,11,22,33,100])
    x = np.arange(y.shape[-1])
    id = "ts_test"
    ts = TimeSeries(x,y,id)
    
    quantizer = TimeSeriesQuantizer()
    quantizer.l_bound = l_bound
    quantizer.u_bound = u_bound
    qts = quantizer.quantize(ts)
    tokens = qts.tokens
    
    np.testing.assert_array_equal(tokens,expected_tokens,f"Quantization bug!")
 
    
def test_batching_quantizer():
    y = np.array([-100000,-11,0,11,22,33,100000])
    x = np.arange(y.shape[-1])
    id = "ts_test"
    ts = [TimeSeries(x,y,id)]
    
    quantizer = TimeSeriesQuantizer()
    ts_batched = quantizer.batch(ts)
    x0_batched = ts_batched[0].x
    y0_batched = ts_batched[0].y
    
    np.testing.assert_array_equal(x0_batched, np.array([0,1]),"Batching bug!")
    np.testing.assert_array_equal(y0_batched, np.array([-100000,-11]),"Batching bug!")