"""Exact numerical checkpoint comparison; no optimizer/model activation authority."""

import math
from typing import Any


def exact_tensor_tree(actual: Any, expected: Any, *, path: str = "root") -> None:
    import torch

    if isinstance(expected, torch.Tensor):
        if (
            not isinstance(actual, torch.Tensor)
            or actual.dtype != expected.dtype
            or actual.shape != expected.shape
            or not torch.isfinite(actual).all()
            or not torch.isfinite(expected).all()
            or actual.detach().cpu().contiguous().numpy().tobytes()
            != expected.detach().cpu().contiguous().numpy().tobytes()
        ):
            raise ValueError("checkpoint tensor bytes differ: " + path)
    elif isinstance(expected, dict):
        if not isinstance(actual, dict) or set(actual) != set(expected):
            raise ValueError("checkpoint mapping keys differ: " + path)
        for key, value in expected.items():
            exact_tensor_tree(actual[key], value, path=path + "." + str(key))
    elif type(expected) in (list, tuple):
        if type(actual) is not type(expected) or len(actual) != len(expected):
            raise ValueError("checkpoint sequence differs: " + path)
        for i, value in enumerate(expected):
            exact_tensor_tree(actual[i], value, path=path + "[" + str(i) + "]")
    elif type(expected) in (str, bool, int, float, type(None)):
        if (
            type(actual) is not type(expected)
            or actual != expected
            or (type(expected) is float and not math.isfinite(expected))
        ):
            raise ValueError("checkpoint scalar differs: " + path)
    else:
        raise ValueError("unsupported numerical checkpoint object: " + path)
