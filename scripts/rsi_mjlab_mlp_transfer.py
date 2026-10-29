"""Narrow MLP weight expansion for a separately verified observation prefix.

Research-only: callers must independently establish identical semantics and
ordering of every prefix dimension, action space and model architecture. This
does not migrate LSTMs or optimizers and does not certify a trained policy.
"""

import torch


def expand_mlp_state(source, destination, *, old_features, new_features):
    """Copy the source function while zeroing weights for new inputs.

    `old_features` and `new_features` are dimension-level identifiers, not
    merely observation-term names. New normalizer dimensions use neutral
    destination initialization; the copied sample count will make subsequent
    running statistics on those dimensions initially conservative.
    """
    old_features, new_features = tuple(old_features), tuple(new_features)
    if (
        not old_features
        or len(set(new_features)) != len(new_features)
        or new_features[: len(old_features)] != old_features
        or len(new_features) <= len(old_features)
        or source.keys() != destination.keys()
    ):
        raise ValueError("explicit unique strict observation prefix and equal state keys required")
    n_old, n_new = len(old_features), len(new_features)
    first = "mlp.0.weight"
    if first not in source or source[first].ndim != 2 or destination[first].ndim != 2:
        raise ValueError("expected MLP first-layer weight required")
    if (
        source[first].shape != (destination[first].shape[0], n_old)
        or destination[first].shape[1] != n_new
    ):
        raise ValueError("first-layer feature dimensions differ")
    out = {}
    for key, src in source.items():
        dst = destination[key]
        if (
            not isinstance(src, torch.Tensor)
            or not isinstance(dst, torch.Tensor)
            or src.dtype != dst.dtype
        ):
            raise ValueError("same tensor state schema required")
        if key == first:
            out[key] = torch.zeros_like(dst)
            out[key][:, :n_old].copy_(src)
        elif key in ("obs_normalizer._mean", "obs_normalizer._var", "obs_normalizer._std"):
            if src.shape != (1, n_old) or dst.shape != (1, n_new):
                raise ValueError("normalizer feature dimensions differ")
            neutral = 0 if key.endswith("_mean") else 1
            if not torch.all(dst[:, n_old:] == neutral):
                raise ValueError("new normalizer features require neutral initialization")
            out[key] = dst.clone()
            out[key][:, :n_old].copy_(src)
        else:
            if src.shape != dst.shape:
                raise ValueError(f"incompatible tensor {key}")
            out[key] = src.clone()
    return out
