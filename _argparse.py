from typing import Dict
from argparse import ArgumentParser, Namespace
import warnings

import torch

import os
import json

def validate_args(args: Namespace) -> Namespace:
    args.ngram_num_embedding = args.qtz_num_bins + args.qtz_num_special_bins
    args.qmc_window_length = args.qtz_window_length
    args.qmc_num_last_unmasked = args.qds_num_last_unmasked

    if args.SMOKE_TEST:
        args.qmc_num_epochs = 1

    return args

def save_args(args: Namespace,
              path: str) -> None:
    with open(os.path.join(path, 'params.json'), 'w') as f:
        json.dump(vars(args), f)

def load_args(path: str) -> Namespace:
    with open(os.path.join(path, 'params.json')) as f:
        args = Namespace(**json.load(f))
    return args

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
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--SMOKE-TEST", action="store_true")
    parser.add_argument("--ARGS-FILE", type=str, default=None)
    parser.add_argument("--qtz-l-bound", type=float, default=0.0)
    parser.add_argument("--qtz-u-bound", type=float, default=1.0)
    parser.add_argument("--qtz-num-bins", type=int, default=10)
    parser.add_argument("--qtz-num-special-bins", type=int, default=3)
    parser.add_argument("--qtz-l-value", type=int, default=-1)
    parser.add_argument("--qtz-u-value", type=int, default=2)
    parser.add_argument("--qtz-window-length", type=int, default=8)
    parser.add_argument("--ts-train-split", type=float, default=0.7)
    parser.add_argument("--ts-eval-split", type=float, default=0.15)
    parser.add_argument("--qds-num-last-unmasked", type=int, default=1) # values different than 1 break the training loop (needs ugly reshapes)
    parser.add_argument("--qds-objective", choices=["ar", "mlm"], default="ar")
    parser.add_argument("--qds-inner-split", action="store_true")
    parser.add_argument("--trans-num-embedding", type=int, default=13)
    parser.add_argument("--trans-att-num-heads", type=int, default=2)
    parser.add_argument("--trans-att-feedforward-dim", type=int, default=64)
    parser.add_argument("--trans-dropout", type=float, default=0.1)
    parser.add_argument("--trans-pos-dropout", type=float, default=0.1)
    parser.add_argument("--trans-pos-max-len", type=int, default=5000)
    parser.add_argument("--trans-att-num-layers", type=int, default=2)
    parser.add_argument("--trans-embedding-dim", type=int, default=16)
    parser.add_argument("--trans-lr", type=float, default=3e-4)
    parser.add_argument("--trans-weight-decay", type=float, default=1e-5)
    parser.add_argument("--qmc-num-epochs", type=int, default=1000)
    parser.add_argument("--qmc-batch-size", type=int, default=10)
    parser.add_argument("--qmc-shuffle", type=bool, default=False)
    parser.add_argument("--qmc-eval-epoch", type=int, default=10)
    parser.add_argument("--qmc-window-length", type=int, default=8) # assert equals to qtz
    parser.add_argument("--qmc-num-last-unmasked", type=int, default=1) # assert equals to qds
    parser.add_argument("--qmc-random-shifts", type=bool, default=True)
    parser.add_argument("--qmc-soft-labels", type=bool, default=True)
    args = parser.parse_args()
    if args.ARGS_FILE is not None:
        args = load_args("plots/")
    args = validate_args(args)
    save_args(args, "plots/")
    ArgumentHandler.set_args(args)