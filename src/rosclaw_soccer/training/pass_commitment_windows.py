"""Enumerate observed tactical commitment windows without inferring contact."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True, slots=True)
class PassCommitmentWindow:
    sender_code: int
    receiver_code: int
    first_frame: int
    last_frame: int
    first_observed_sec: float
    last_observed_sec: float
    next_observation_sec: float | None
    right_censored: bool


def enumerate_pass_commitment_windows(
    *,
    control_time_sec: NDArray[Any],
    source_codes: NDArray[Any],
    receiver_codes: NDArray[Any],
    agent_count: int,
    control_period_sec: float = 0.02,
) -> tuple[PassCommitmentWindow, ...]:
    """List every maximal contiguous visible source/receiver pair.

    A zero/zero frame means no observed commitment; it breaks the window.
    Unknown, partial or out-of-range identities are rejected rather than
    guessed. Start and end times are sampled observation boundaries, not the
    actual tactical transition time. A final window is right-censored. This
    function says nothing about contacts, intent causality, fitness or safety;
    callers must authenticate the input and inspect physical events separately.
    """
    t, src, dst = map(np.asarray, (control_time_sec, source_codes, receiver_codes))
    if (
        t.ndim != 1
        or not 1 <= len(t) <= 1001
        or t.dtype.kind not in "fiu"
        or not np.isfinite(t).all()
        or (t < 0).any()
        or (t > 20.000001).any()
        or src.shape != t.shape
        or dst.shape != t.shape
        or src.dtype.kind not in "iu"
        or dst.dtype.kind not in "iu"
        or type(agent_count) is not int
        or not 2 <= agent_count <= 64
        or type(control_period_sec) not in (float, int)
        or not math.isfinite(control_period_sec)
        or not 0.001 <= control_period_sec <= 0.2
        or not np.allclose(np.diff(t), control_period_sec, rtol=0.0, atol=1e-8)
        or ((src < 0) | (src > agent_count) | (dst < 0) | (dst > agent_count)).any()
        or ((src == 0) != (dst == 0)).any()
        or ((src == dst) & (src != 0)).any()
    ):
        raise ValueError("complete ordered bounded commitment snapshots required")
    windows: list[PassCommitmentWindow] = []
    start = 0
    while start < len(t):
        sender, receiver = int(src[start]), int(dst[start])
        end = start + 1
        while end < len(t) and src[end] == sender and dst[end] == receiver:
            end += 1
        if sender:
            windows.append(
                PassCommitmentWindow(
                    sender_code=sender,
                    receiver_code=receiver,
                    first_frame=start,
                    last_frame=end - 1,
                    first_observed_sec=float(t[start]),
                    last_observed_sec=float(t[end - 1]),
                    next_observation_sec=float(t[end]) if end < len(t) else None,
                    right_censored=end == len(t),
                )
            )
        start = end
    return tuple(windows)
