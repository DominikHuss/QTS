import functools
import inspect

# TODO: remove this import once packaged
from _argparse import ArgumentHandler

def parse_args(args_prefix):
    def decorator_parse_args(cls):
        ArgumentHandler._register_prefix(args_prefix)
        @functools.wraps(cls)
        def wrapper_parse_args(*args, **kwargs):
            ArgumentHandler._register_prefix("_global")
            c = cls(*args, **kwargs)
            cls_args = {**ArgumentHandler.get_args(args_prefix), **ArgumentHandler.get_global_args()}
            for k, v in cls_args.items():
                setattr(c, k, v)
            c._build()
            return c
        return wrapper_parse_args
    return decorator_parse_args

