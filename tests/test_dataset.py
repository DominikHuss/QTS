import pytest
import numpy as np
import torch
import torch.nn.functional as F
from dataset import QDatasetBase, QDatasetForTransformerARModel
from dataset import QDatasetForTransformerMLMModel, QDatasetForBertModel
from dataset import QDatasetForGPTModel
from conftest import _ID


@pytest.mark.parametrize("batch, _quantized",[
    (False, False),
    (False, True),
    (True, False),
    (True, True)
])
def test_get_unbatched_QDataset(get_ts, get_qts, batch, _quantized):
    ts = get_ts
    qts = get_qts
    qds = QDatasetBase(ts, batch=batch)
    unbatched_ts = qds.get_unbatched(_ID, _quantized=_quantized) 
    if _quantized:
        np.testing.assert_array_equal(unbatched_ts.tokens, qts.tokens)
        np.testing.assert_array_equal(unbatched_ts.tokens_y, qts.tokens_y)
    else:
        np.testing.assert_array_equal(unbatched_ts.x, ts.x )
        np.testing.assert_array_equal(unbatched_ts.y, ts.y)

        
def test_get_batched_QDataset(get_ts, get_qts, get_args):
    ts = get_ts
    qts = get_qts
    qds = QDatasetBase(ts, batch=True)
    window_length = get_args['window_length']
    
    first_batch_0 = qds.get_batched(_ID) 
    (first_batch_1, prepped_ts), start, length = qds.get_batched("0",_all=True)
    
    np.testing.assert_array_equal(first_batch_0.tokens, qts.tokens[:window_length])
    np.testing.assert_array_equal(first_batch_0.tokens_y, qts.tokens_y[:window_length])
    
    np.testing.assert_array_equal(first_batch_1.tokens, qts.tokens[:window_length])
    np.testing.assert_array_equal(first_batch_1.tokens_y, qts.tokens_y[:window_length])
    assert (prepped_ts == None)
    assert qds.raw_data[start].id() == _ID
    assert length == len(qds)

@pytest.mark.parametrize("batch",[False,True])
def test_QDatasetForTransformerARModel(get_ts, get_qts, get_quantizer, get_args, batch):
    ts = get_ts
    qts= get_qts
    quantizer = get_quantizer
    window_length = get_args['window_length']
    expected_y = torch.tensor(qts.tokens[:window_length])[:-1] if batch else torch.tensor(qts.tokens[:-1])
    expected_y_hat = torch.tensor(qts.tokens[:window_length])[1:] if batch else torch.tensor(qts.tokens[1:])
    expected_y_hat_probs = F.one_hot(expected_y_hat,len(quantizer.bins_indices))
    expected_mask = (torch.zeros(window_length - 1, dtype=torch.bool) if batch
                        else torch.zeros(len(qts.tokens[1:]), dtype=torch.bool)) 
    expected_mask[-1] = 1 
    
    qds = QDatasetForTransformerARModel(ts, batch=batch)
    
    assert (qds[0]['y'] == expected_y).all()
    assert (qds[0]['y_hat'] == expected_y_hat ).all()
    assert (qds[0]['y_hat_probs'] == expected_y_hat_probs).all()
    assert (qds[0]['mask'] == expected_mask).all()


@pytest.mark.parametrize("batch",[False,True])
def test_QDatasetForTransformerMLModel(get_ts, get_qts, get_quantizer, get_args, batch):
    ts = get_ts
    qts= get_qts
    quantizer = get_quantizer
    window_length = get_args['window_length']
    expected_y =  torch.tensor(qts.tokens[:window_length]) if batch else torch.tensor(qts.tokens)
    expected_y_hat = torch.tensor(qts.tokens[:window_length]) if batch else torch.tensor(qts.tokens)
    expected_y_hat_probs = F.one_hot(expected_y_hat,len(quantizer.bins_indices))
   
    qds = QDatasetForTransformerMLMModel(ts, batch=batch)
    
    assert (qds[0]['y'] == expected_y).all()
    assert (qds[0]['y_hat_probs'] == expected_y_hat_probs).all()


@pytest.mark.parametrize("batch",[False,True])
def test_QDatasetForBertModel(get_ts, get_qts, get_quantizer, get_args, batch):
    """
    Not implemented -- it's difficult to generate deterministic output without code changes.
    """
    ...
    
@pytest.mark.parametrize("batch",[False,True])
def test_QDatasetForGPTModel(get_ts, get_qts, get_args, batch):
    ts = get_ts
    qts= get_qts
    window_length = get_args['window_length']
    expected_input_ids =  torch.tensor(qts.tokens[:window_length]) if batch else torch.tensor(qts.tokens)
    expected_labels = torch.tensor(qts.tokens[:window_length]) if batch else torch.tensor(qts.tokens)
   

    qds = QDatasetForGPTModel(ts, batch=batch)
  
    assert (qds[0]['input_ids'] == expected_input_ids).all()
    assert (qds[0]['labels'] == expected_labels).all()

def test_random_shifts():
   "TODO: implementing"
   ...
def test_soft_label():
    "TODO: implementing" 

   