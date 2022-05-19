import warnings
import os
import json
import sys
import torch
from typing import Dict, List
from argparse import ArgumentParser, Namespace


def validate_args(args: Namespace) -> Namespace:
    if not (args.trans_num_embedding == args.qtz_num_bins + args.qtz_special_bins):
        warnings.warn(f"Arg `trans_num_embedding` must equal to sum of args: `qtz_num_bins` and 'qtz_special_bins'." +
                      f"Force change value of `trans_num_embedding` to `qtz_num_bins` + 'qtz_special_bins'")
        args.trans_num_embedding = args.qtz_num_bins + args.qtz_special_bins
    if args.trans_pos_max_len < args.window_length:
        warnings.warn(f"Arg `trans_pos_max_len` must be greater or equal to `window_length` ." +
            f"Force change value of `trans_pos_max_len` to `window_length`")
        args.trans_pos_max_len = args.window_length
    if args.qmc_num_last_unmasked == args.qds_num_last_unmasked:
        warnings.warn("Args `qmc_num_last_unmasked` and `qds_num_last_unmasked` must be equal" +
                      "Force change value of qmc_num_last_unmasked to `qds_num_last_unmasked`")
        args.qmc_num_last_unmasked = args.qds_num_last_unmasked
    
    assert args.qds_mlm_mask_token_prob + args.qds_mlm_random_token_prob <= 1, "Sum of args `mlm_masked_token_prob` and `mlm_random_token_prob` must be less or equal to 1.0 "
    if args.qds_mlm_mask_token_prob + args.qds_mlm_random_token_prob == 1:
        warnings.warn("Probability of masking tokens and probability of random replacing tokens equals to 1" +
                      "None of the inputs tokens will be unchanged. It can cause inappropriate learning and in future model malfunction") 

    if args.qmc_save_model_path is not None and args.qmc_save_model_path == args.qmc_load_model_path:
        warnings.warn("Save path and load path model are the same.")
    
    if args.SMOKE_TEST:
        args.qmc_num_epochs = 1

    return args

def save_args(args: Namespace,
              path: str) -> None:
    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

def load_args(path: str) -> Namespace:
    with open(os.path.join(path, 'params.json')) as f:
        return Namespace(**json.load(f))


class ArgumentHandler():
    args = None
    _processed_args = dict()
    _prefix_registry = set()

    @staticmethod
    def set_args(args):
        ArgumentHandler.args = args

    @staticmethod
    def _register_prefix(prefix):
        ArgumentHandler._process_args(prefix)
        ArgumentHandler._prefix_registry.add(prefix)

    @staticmethod
    def _process_args(_prefix):
        if _prefix in ArgumentHandler._prefix_registry:
            # Prefix already processed by a different decorated class
            pass
        else:
            if _prefix == "_global":
                ArgumentHandler._processed_args["_global"] = {f"_global_{k}": v for k,v in vars(ArgumentHandler.args).items() if not any([k.startswith(f"{_p}_") for _p in ArgumentHandler._prefix_registry])}
                if ArgumentHandler.args.cuda and torch.cuda.is_available():
                    ArgumentHandler._processed_args["_global"]["_global_cuda"] = "cuda"
                else:
                    ArgumentHandler._processed_args["_global"]["_global_cuda"] = "cpu"

            else:
                ArgumentHandler._processed_args[_prefix] = {k[len(_prefix)+1:]: v for k,v in vars(ArgumentHandler.args).items() if k.startswith(f"{_prefix}_")}


    @staticmethod
    def get_global_args() -> Dict:
        return ArgumentHandler._processed_args["_global"]

    @staticmethod
    def get_args(prefix) -> Dict:
        return ArgumentHandler._processed_args[prefix]


    def __new__(cls, *args, **kwargs):
        raise TypeError("abstract class may not be instantiated")

# TODO: validate args; compute dependent arguments e.g. ngram_num_embedding = qtz_num_bins + qtz_num_special_bins, smoke_test and num_epochs
def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--ARGS-FILE", type=str, default=None)
    parser.add_argument("--input-dir", type=str, default="data/basic")
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--SMOKE-TEST", action="store_true")
    parser.add_argument("--window-length", type=int, default=20)
    parser.add_argument("--model", choices=['transformer_ar', 'transformer_mlm', 'bert', "gpt"], default="transformer_ar")
    parser.add_argument("--qtz-l-bound", type=float, default= None)
    parser.add_argument("--qtz-u-bound", type=float, default= None)
    parser.add_argument("--qtz-num-bins", type=int, default=20)
    parser.add_argument("--qtz-special-bins", type=int, default=9)
    parser.add_argument("--qtz-additional-special-bins", type= List[str], default=None)
    parser.add_argument("--qtz-l-value", type=int, default=-1)
    parser.add_argument("--qtz-u-value", type=int, default=2)
    parser.add_argument("--ts-train-split", type=float, default=0.7)
    parser.add_argument("--ts-eval-split", type=float, default=0.2)
    parser.add_argument("--qds-num-last-unmasked", type=int, default=1) # values different than 1 break the training loop (needs ugly reshapes)
    parser.add_argument("--qds-inner-split", action="store_true")
    parser.add_argument("--qds-mlm-mask-prob", type=float, default=0.15)
    parser.add_argument("--qds-mlm-non-masked-value", type=int, default=-100)
    parser.add_argument("--qds-mlm-mask-token-prob", type=float, default=0.8)
    parser.add_argument("--qds-mlm-random-token-prob", type=float, default=0.1)
    parser.add_argument("--trans-num-embedding", type=int, default=13)
    parser.add_argument("--trans-att-num-heads", type=int, default=4)
    parser.add_argument("--trans-att-feedforward-dim", type=int, default=64)
    parser.add_argument("--trans-dropout", type=float, default=0.1)
    parser.add_argument("--trans-pos-dropout", type=float, default=0.1)
    parser.add_argument("--trans-pos-max-len", type=int, default=20)
    parser.add_argument("--trans-att-num-layers", type=int, default=4)
    parser.add_argument("--trans-embedding-dim", type=int, default=64)
    parser.add_argument("--trans-lr", type=float, default=1e-4)
    parser.add_argument("--trans-weight-decay", type=float, default=1e-7)
    parser.add_argument("--qmc-save-model-path", type=str, default=None) #assert save == load as warning
    parser.add_argument("--qmc-load-model-path", type=str, default=None)
    parser.add_argument("--qmc-checkpoints-epochs", type=int, default=10)
    parser.add_argument("--qmc-from-saved-model", type=bool, default=False)
    parser.add_argument("--qmc-num-epochs", type=int, default=100)
    parser.add_argument("--qmc-batch-size", type=int, default=128)
    parser.add_argument("--qmc-shuffle", type=bool, default=False)
    parser.add_argument("--qmc-eval-epoch", type=int, default=10)
    parser.add_argument("--qmc-num-last-unmasked", type=int, default=1)
    parser.add_argument("--qmc-random-shifts", type=bool, default=False)
    parser.add_argument("--qmc-soft-labels", type=bool, default=False)
    
    args = parser.parse_args(["--ARGS-FILE", "tests"]) if "pytest" in sys.argv else parser.parse_args()
    if args.ARGS_FILE is not None:
        args_dict = vars(args)
        args_dict.update(vars(load_args(args.ARGS_FILE)))
        args = Namespace(**args_dict)
    
    args = validate_args(args)
    save_args(args, "plots/")
    ArgumentHandler.set_args(args)