"""Causal, SIM-only research signal; not a collision guard or motor command.

This reports an observation for controlled experiments. It does not alter the
ball, actor, controller, official exam or authorization state. The geometry
must come from the same current physical state, not an offline future replay.
"""

import math
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class KneeSignalState:
    """Exactly one observation per contiguous 50 Hz control boundary."""

    next_control: int = 0
    previous_gaps_m: tuple[float, float] | None = None
    previous_shrinking: tuple[bool, bool] = (False, False)
    foot_contact_seen: bool = False


def precontact_knee_signal(
    state: KneeSignalState,
    *,
    control: int,
    knee_ball_surface_gaps_m: tuple[float, float],
    actual_foot_contact: bool,
) -> tuple[KneeSignalState, tuple[bool, bool]]:
    """Signal a persistent near-and-closing pair before observed foot contact.

    Both same-side gaps must be shrinking on two consecutive transitions at or
    below 0.06 m. Contact is latched, not inferred from a future event. A
    negative geometric gap is admissible and does not imply contact force.
    Returns a new state; invalid input never mutates the caller's old state.
    """
    if (
        type(state) is not KneeSignalState
        or type(control) is not int
        or control != state.next_control
        or type(actual_foot_contact) is not bool
        or type(knee_ball_surface_gaps_m) is not tuple
        or len(knee_ball_surface_gaps_m) != 2
        or any(
            type(x) not in (float, int) or not math.isfinite(x) for x in knee_ball_surface_gaps_m
        )
        or type(state.foot_contact_seen) is not bool
        or type(state.previous_shrinking) is not tuple
        or len(state.previous_shrinking) != 2
        or any(type(x) is not bool for x in state.previous_shrinking)
    ):
        raise ValueError("contiguous controls and finite explicit physical observations required")
    previous = state.previous_gaps_m
    if previous is not None and (
        type(previous) is not tuple
        or len(previous) != 2
        or any(type(x) not in (float, int) or not math.isfinite(x) for x in previous)
    ):
        raise ValueError("finite previous physical gaps required")
    gaps = tuple(float(x) for x in knee_ball_surface_gaps_m)
    shrinking = tuple(
        previous is not None and gaps[i] <= 0.06 and gaps[i] < previous[i] for i in range(2)
    )
    seen = state.foot_contact_seen or actual_foot_contact
    signal = tuple(not seen and shrinking[i] and state.previous_shrinking[i] for i in range(2))
    return (
        KneeSignalState(control + 1, gaps, shrinking, seen),
        signal,
    )
