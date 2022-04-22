import pytest
import numpy as np
import torch
from preprocessing import TimeSeries
from preprocessing import TimeSeriesQuantizer
from dataset import QDataset


# NDARRAY = np.arange(TS_LENGTH)
# TS =  TimeSeries(np.arange(TS_LENGTH),NDARRAY,"0")
# QUANTIZER =  TimeSeriesQuantizer()
# QTS = QUANTIZER.quantize(TS)
# ts1 = TimeSeries(np.arange(TS_LENGTH),np.arange(TS_LENGTH),"0")
# ts2 = TimeSeries(np.arange(TS_LENGTH,TS_LENGTH + 11),np.arange(TS_LENGTH,TS_LENGTH + 11),"1")
# TS_LIST = [ts1,ts2]


# @pytest.mark.parametrize("data",[
#     NDARRAY,
#     [NDARRAY],
#     TS,
#     [TS],
#     QTS,
#     [QTS]
# ])

def test_QDataset(get_qts):
    data = get_qts
    print(data)
    # expected_y = torch.tensor(QTS.tokens[:-1])
    # expected_y_hat = torch.tensor(QTS.tokens[1:])
    # expected_y_hat_probs = torch.nn.functional.one_hot(expected_y_hat,len(QUANTIZER.bins_indices))
    # expected_mask = torch.zeros((TS_LENGTH-1), dtype=torch.bool)
    # expected_mask[-1] = 1 #only works for "ar"!
    # qds = QDataset(data, batch=False, soft_labels=False)
    
    # assert (qds[0]['y'] == expected_y).all()
    # assert (qds[0]['y_hat'] == expected_y_hat ).all()
    # assert (qds[0]['y_hat_probs'] == expected_y_hat_probs).all()
    # assert (qds[0]['mask'] == expected_mask).all()

@pytest.mark.parametrize("batch",[False,True])
def test_get_unbatched_QDataset(batch):
    qds = QDataset(TS_LIST, batch=batch, soft_labels=False)
    unbatched_ts1 = qds.get_unbatched("0") 
    unbatched_ts2 = qds.get_unbatched("1")
    
    np.testing.assert_array_equal(unbatched_ts1.x, ts1.x )
    np.testing.assert_array_equal(unbatched_ts1.y, ts1.y)
    
    np.testing.assert_array_equal(unbatched_ts2.x, ts2.x )
    np.testing.assert_array_equal(unbatched_ts2.y, ts2.y)
    
    
def test_get_batched_QDataset():
    """
    return only first batch of given ts(id) -> strange?
    """
    qts_list = QUANTIZER.quantize(TS_LIST)
    qds = QDataset(TS_LIST, batch=True, soft_labels=False)
    first_batch_ts0 = qds.get_batched("0") 
    (first_batch_ts1, prepped_ts1), start, length = qds.get_batched("1",_all=True)
    
    np.testing.assert_array_equal(first_batch_ts0.tokens, qts_list[0].tokens[:WINDOW_WIDTH])
    np.testing.assert_array_equal(first_batch_ts0.tokens_y, qts_list[0].tokens_y[:WINDOW_WIDTH])
    
    np.testing.assert_array_equal(first_batch_ts1.tokens, qts_list[1].tokens[:WINDOW_WIDTH])
    np.testing.assert_array_equal(first_batch_ts1.tokens_y, qts_list[1].tokens_y[:WINDOW_WIDTH])
    assert (prepped_ts1 == torch.tensor(qts_list[1].tokens[:WINDOW_WIDTH - 1])).all()
    assert qds.raw_data[start].id() == "1"
    assert length == WINDOW_WIDTH

def test_dataloader():
    qds = QDataset(TS_LIST, batch=True, soft_labels=False)
    expected_batch = {
           'y': torch.stack((qds[0]['y'],qds[1]['y'])),
           'y_hat': torch.stack((qds[0]['y_hat'],qds[1]['y_hat'])),
           'y_hat_probs': torch.stack((qds[0]['y_hat_probs'],qds[1]['y_hat_probs'])),
           'mask': torch.stack((qds[0]['mask'],qds[1]['mask'])) 
    }
    
    dl = torch.utils.data.DataLoader(qds, batch_size=2, shuffle=False)
    for batch in dl:
        assert (batch['y'] == expected_batch['y']).all()
        assert (batch['y_hat'] == expected_batch['y_hat']).all()
        assert (batch['y_hat_probs'] == expected_batch['y_hat_probs']).all()
        assert (batch['mask'] == expected_batch['mask']).all()
        break  

def test_random_shifts():
    """
    WARNING: results may be nondeterministic
    for seed 44 shifts should be [0,0,-1,0]
    """
    torch.manual_seed(SEED)
    expected_shifts = torch.tensor([0,0,-1,0])
    g = torch.Generator()
    g.manual_seed(SEED)
    qds = QDataset(TS_LIST, batch=True, soft_labels=False, random_shifts=True)
   
    for idx, batch in enumerate(torch.utils.data.DataLoader(qds,
                                    batch_size=1,
                                    shuffle=False,
                                    generator=g
    )):
        assert (batch['y'] == qds[idx]['y'] + expected_shifts).all()
        break
    
def test_soft_label():
        pass 

   