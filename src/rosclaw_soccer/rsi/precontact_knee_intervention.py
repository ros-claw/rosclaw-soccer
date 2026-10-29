"""Explicit SIM research intervention, not a validated or hardware guard."""

from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.precontact_knee_signal import KneeSignalState, precontact_knee_signal

KNEES = tuple(G1_DDS_JOINT_NAMES.index(f"{side}_knee_joint") for side in ("left", "right"))


@dataclass(frozen=True, slots=True)
class KneeInterventionState:
    signal: KneeSignalState = KneeSignalState()
    active_sides: tuple[bool, bool] = (False, False)
    remaining_controls: int = 0
    fired: bool = False


def research_knee_target(
    state: KneeInterventionState,
    *,
    control: int,
    target: np.ndarray,
    knee_ball_surface_gaps_m: tuple[float, float],
    actual_foot_contact: bool,
    offset_rad: float,
) -> tuple[KneeInterventionState, np.ndarray]:
    """One immutable causal action lease; fixed -0.04/0/+0.04 rad, <=100ms.

    Outputs are merely candidate position targets for a native simulation PD
    owner. No motor, body state, safety gate or real runtime is operated here.
    The caller must keep tracking/force caps and log actual executed torques.
    """
    if (
        type(state) is not KneeInterventionState
        or type(target) is not np.ndarray
        or target.shape != (29,)
        or target.dtype != np.float64
        or not np.isfinite(target).all()
        or type(offset_rad) is not float
        or offset_rad not in (-0.04, 0.0, 0.04)
        or type(state.fired) is not bool
        or type(state.remaining_controls) is not int
        or not 0 <= state.remaining_controls <= 4
        or type(state.active_sides) is not tuple
        or len(state.active_sides) != 2
        or any(type(x) is not bool for x in state.active_sides)
    ):
        raise ValueError("explicit finite simulation target and bounded research state required")
    signal_state, signal = precontact_knee_signal(
        state.signal,
        control=control,
        knee_ball_surface_gaps_m=knee_ball_surface_gaps_m,
        actual_foot_contact=actual_foot_contact,
    )
    first = not state.fired and any(signal)
    sides = signal if first else state.active_sides
    count = 5 if first else state.remaining_controls
    result = target.copy()
    if count:
        for i, enabled in enumerate(sides):
            if enabled:
                result[KNEES[i]] += offset_rad
        count -= 1
    return KneeInterventionState(signal_state, sides, count, state.fired or first), result
