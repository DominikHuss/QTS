from typing import Dict
from argparse import ArgumentParser

import torch

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
    parser.add_argument("--qtz-l-bound", type=float, default=0.0)
    parser.add_argument("--qtz-u-bound", type=float, default=1.0)
    parser.add_argument("--qtz-num-bins", type=int, default=10)
    parser.add_argument("--qtz-num-special-bins", type=int, default=3)
    parser.add_argument("--qtz-l-value", type=int, default=-1)
    parser.add_argument("--qtz-u-value", type=int, default=2)
    parser.add_argument("--qtz-window-length", type=int, default=4)
    parser.add_argument("--ts-train-split", type=float, default=0.7)
    parser.add_argument("--ts-eval-split", type=float, default=0.15)
    parser.add_argument("--qds-num-last-unmasked", type=int, default=1)
    parser.add_argument("--qds-objective", choices=["ar", "mlm"], default="ar")
    parser.add_argument("--trans-num-embedding", type=int, default=13)
    parser.add_argument("--trans-att-num-heads", type=int, default=2)
    parser.add_argument("--trans-att-feedforward-dim", type=int, default=64)
    parser.add_argument("--trans-dropout", type=float, default=0.1)
    parser.add_argument("--trans-att-num-layers", type=int, default=2)
    parser.add_argument("--trans-embedding-dim", type=int, default=16)
    parser.add_argument("--trans-lr", type=float, default=1e-3)
    parser.add_argument("--trans-weight-decay", type=float, default=1e-5)
    parser.add_argument("--qmc-num-epochs", type=int, default=100)
    parser.add_argument("--qmc-batch-size", type=int, default=4)
    parser.add_argument("--qmc-shuffle", type=bool, default=False)
    args = parser.parse_args()
    ArgumentHandler.set_args(args)