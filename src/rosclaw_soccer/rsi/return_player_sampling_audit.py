"""Independent full-history probability check for simulated A-side samples."""

import math

import numpy as np
import torch

from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp


def audit_return_player_sampling(trace, scale):
    """Reject incomplete, noncontiguous or incorrectly scored behavior rows."""
    mask = np.asarray(trace["sampling_active"])
    if mask.ndim != 1 or mask.dtype != np.bool_:
        raise ValueError("one boolean sampling marker per actual control required")
    indices = np.flatnonzero(mask)
    if len(indices) > 300 or (
        len(indices)
        and not np.array_equal(indices, np.arange(indices[0], indices[0] + len(indices)))
    ):
        raise ValueError("one bounded consecutive sampling window required")
    if not len(indices):
        return {"samples": 0, "max_logp_error": 0.0}
    names = ("sampling_mean", "sampling_action", "sampling_reference")
    arrays = [np.asarray(trace[name])[mask] for name in names]
    if any(a.shape != (len(indices), 29) or a.dtype != np.float32 for a in arrays):
        raise ValueError("actual float32 sampling history required")
    saved = np.asarray(trace["sampling_logp"])[mask]
    scale = np.asarray(scale)
    if saved.shape != (len(indices),) or not np.isfinite(saved).all():
        raise ValueError("finite behavior density per sampled control required")
    if scale.shape != (29,) or scale.dtype != np.float32:
        raise ValueError("exact float32 action scale required")
    with torch.no_grad():
        actual = history_motor_logp(
            "box-truncated-ar1",
            *[torch.from_numpy(a.copy()) for a in arrays],
            torch.from_numpy(scale.copy()),
            torch.full((29,), math.log(0.02), dtype=torch.float32),
            rho=0.9,
        ).numpy()
    error = float(np.max(np.abs(actual - saved)))
    if not np.isfinite(error) or error > 1e-4:
        raise ValueError("full-history conditional behavior density mismatch")
    return {"samples": len(indices), "max_logp_error": error}
