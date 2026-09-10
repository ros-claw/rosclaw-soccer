"""First-departure learning credit, not possession or motor authorization.

A later chase-and-rekick must not turn an inadequate first pass into a success.
The caller supplies measured contacts and keeps body/ball safety gates separate.
Torch remains optional until this pure tensor operation is explicitly called.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class FirstLaunchCredit:
    departed: Any
    closed: Any
    departure_time_sec: Any
    eligible_contact: Any
    late_source_contact: Any


def advance_first_launch_credit(
    *,
    departed: Any,
    closed: Any,
    departure_time_sec: Any,
    foot_contact: Any,
    forward_speed_mps: Any,
    previous_elapsed_sec: float,
    elapsed_sec: float,
) -> FirstLaunchCredit:
    """Credit preparation and the first uninterrupted departure contact only.

    Measured forward ball speed >=0.4 m/s during foot contact starts departure.
    The contact window then
    closes on separation or after 0.12 s, never renewing on subsequent touches.
    Navigation/recovery selection is the caller's job; no motion is requested.
    Inputs are not mutated. The 2 ms observation stream must remain fresh.
    """
    import torch

    if (
        any(
            type(t) not in (int, float) or not math.isfinite(t) or not 0 <= t <= 60
            for t in (previous_elapsed_sec, elapsed_sec)
        )
        or not 0 < elapsed_sec - previous_elapsed_sec <= 0.0041
        or not isinstance(forward_speed_mps, torch.Tensor)
        or forward_speed_mps.ndim != 1
        or not 1 <= len(forward_speed_mps) <= 4096
        or forward_speed_mps.dtype not in (torch.float32, torch.float64)
        or forward_speed_mps.requires_grad
        or not bool(torch.isfinite(forward_speed_mps).all())
        or bool((forward_speed_mps.abs() > 100).any())
    ):
        raise ValueError("fresh finite measured first-launch observations required")
    if any(
        not isinstance(t, torch.Tensor)
        or t.shape != forward_speed_mps.shape
        or t.device != forward_speed_mps.device
        or t.dtype != torch.bool
        for t in (departed, closed, foot_contact)
    ):
        raise ValueError("aligned boolean first-launch history required")
    if (
        not isinstance(departure_time_sec, torch.Tensor)
        or departure_time_sec.shape != departed.shape
        or departure_time_sec.device != departed.device
        or departure_time_sec.dtype not in (torch.float32, torch.float64)
        or departure_time_sec.requires_grad
        or not bool(torch.isfinite(departure_time_sec).all())
        or bool((departed != (departure_time_sec >= 0)).any())
        or bool(((~departed) & (departure_time_sec != -1)).any())
        or bool((closed & ~departed).any())
        or bool((departure_time_sec > previous_elapsed_sec + 1e-6).any())
    ):
        raise ValueError("first-launch timestamp disagrees with bounded history")
    start = ~departed & foot_contact & (forward_speed_mps >= 0.4)
    time = torch.where(start, torch.full_like(departure_time_sec, elapsed_sec), departure_time_sec)
    next_departed = departed | start
    next_closed = closed | (departed & (~foot_contact | (elapsed_sec - time > 0.12 + 1e-6)))
    return FirstLaunchCredit(
        departed=next_departed,
        closed=next_closed,
        departure_time_sec=time,
        eligible_contact=foot_contact & ~next_closed,
        late_source_contact=foot_contact & next_closed,
    )
