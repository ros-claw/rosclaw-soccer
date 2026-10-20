"""Gaussian probability MASS of the actual FSQ32 symbol, not raw-action density.

Integrates over each monotone quantizer bin. The actual sampler remains the
existing float32 normal -> tanh -> round network; this is its ideal continuous
Gaussian bin model (finite RNG/rounding approximations are not changed).
"""

from typing import Any


def fsq_log_mass(mean: Any, std: Any, token: Any) -> Any:
    import torch

    if (
        mean.shape != std.shape
        or mean.shape != token.shape
        or mean.dtype != torch.float32
        or std.dtype != torch.float32
        or token.dtype != torch.float32
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or not torch.isfinite(token).all()
        or (std <= 0).any()
        or (token < -1).any()
        or (token > 0.9375).any()
        or not torch.equal(token * 16, torch.round(token * 16))
    ):
        raise ValueError("finite equal-shaped Gaussian and original FSQ32 symbols required")
    k = token.double() * 16
    lowest, highest = k == -16, k == 15
    # Finite placeholders for infinite end bins avoid 0*inf backward NaNs.
    lo = (
        torch.atanh(torch.where(lowest, torch.zeros_like(k), k / 15.515501022338867))
        - 0.03223699703812599
    )
    hi = (
        torch.atanh(torch.where(highest, torch.zeros_like(k), (k + 1) / 15.515501022338867))
        - 0.03223699703812599
    )
    zl, zh = (lo - mean.double()) / std.double(), (hi - mean.double()) / std.double()
    upper_tail = ((zl > 0) & ~lowest) | highest
    big = torch.where(upper_tail, torch.special.log_ndtr(-zl), torch.special.log_ndtr(zh))
    small = torch.where(upper_tail, torch.special.log_ndtr(-zh), torch.special.log_ndtr(zl))
    small = torch.where(lowest | highest, torch.full_like(small, -torch.inf), small)
    result = big + torch.log(-torch.expm1(small - big))
    if not torch.isfinite(result).all() or (result > 1e-10).any():
        raise ValueError("nonfinite or invalid FSQ symbol mass")
    return result.to(torch.float32)
