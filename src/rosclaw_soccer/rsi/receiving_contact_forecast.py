"""Causal SIM contact-forecast labels with explicit end-of-episode censoring."""

import numpy as np


def future_first_contact_labels(force: np.ndarray, *, horizon: int = 5) -> np.ndarray:
    """0=no contact, 1=foot only, 2=includes nonfoot; excludes censored tails.

    Index t uses actual subsequent native force in controls [t, t+horizon).
    This output is a supervised label and must never become an input feature.
    """
    if (
        force.ndim != 4
        or force.shape[1:] != (10, 2, 3)
        or force.dtype != np.float64
        or not np.isfinite(force).all()
        or (force < 0).any()
        or type(horizon) is not int
        or not 1 <= horizon <= 20
        or len(force) < horizon
    ):
        raise ValueError("finite measured native force and valid prediction horizon required")
    out = np.zeros(len(force) - horizon + 1, dtype=np.int8)
    for t in range(len(out)):
        window = force[t : t + horizon, :, 1, :]
        events = np.argwhere((window > 1).any(axis=2))
        if len(events):
            control, substep = events[0]
            out[t] = 2 if window[control, substep, 2] > 1 else 1
    return out
