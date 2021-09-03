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

def _parse_arguments():
    parser = ArgumentParser()
    parser.add_argument("--cuda", action="store_true")
    parser.add_argument("--qtz-l-bound", type=float, default=0.0)
    parser.add_argument("--qtz-u-bound", type=float, default=1.0)
    parser.add_argument("--qtz-num-bins", type=int, default=10)
    parser.add_argument("--qtz-num-special-bins", type=int, default=3)
    args = parser.parse_args()
    ArgumentHandler.set_args(args)