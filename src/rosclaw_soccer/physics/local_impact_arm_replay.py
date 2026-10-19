"""Local-pilot recorded-target physics, with explicit zero-control derivation.

Original-control captures contain no early-action tensors. We never pretend
they were recorded: derive zeros only after checking the full clipped-PD and
measured-actuator attribution, and disclose that derivation in the result.
File/course/source/arm authentication remains the caller's responsibility.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.recorded_early_episode import RecordedEarlyEpisode
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS


def verify_local_recorded_arm(
    replay: RecordedEarlyEpisode, trace: dict[str, np.ndarray], *, arm: str
) -> dict[str, Any]:
    if type(replay) is not RecordedEarlyEpisode or arm not in (
        "original_control",
        "prospective_early_impact",
    ):
        raise ValueError("exact recorded physics engine and declared local pilot arm required")
    if any(k.startswith("impact_feedback_") for k in trace):
        raise ValueError("old last20ms profile cannot stand in for this local experiment")
    derived = arm == "original_control"
    if derived:
        if any(k.startswith("early_impact_feedback_") for k in trace):
            raise ValueError("original-control arm must not contain early-action captures")
        fields = ("pd_requested_torque_nm", "torque_nm", "actual_actuator_force_nm")
        for key in fields:
            a = trace.get(key)
            if (
                not isinstance(a, np.ndarray)
                or a.shape != (300, 1, 10, 29)
                or a.dtype != np.float64
                or not np.isfinite(a).all()
            ):
                raise ValueError("whole typed measured base-PD attribution required")
        limits = np.asarray(G1_HARD_TORQUE_LIMITS)
        base = np.clip(trace[fields[0]], -limits, limits)
        if any(not np.array_equal(trace[key], base) for key in fields[1:]):
            raise ValueError("cannot derive zero control from nonzero physical attribution")
        checked = dict(trace)
        checked["early_impact_feedback_requested_offset_nm"] = np.zeros((300, 1, 29))
        checked["early_impact_feedback_applied_offset_nm"] = np.zeros((300, 1, 10, 29))
    else:
        checked = trace
    review = replay.verify(checked)
    return dict(
        schema="soccer.rsi.local_recorded_arm_physics.v1",
        arm=arm,
        physical_review=review,
        zero_action_arrays_derived_not_recorded=derived,
        zero_action_derivation="CLIPPED_PD_AND_ACTUAL_ACTUATOR_FORCE_VERIFIED" if derived else None,
        original_trace_mutated=False,
        source_and_course_authentication_required_from_caller=True,
        new_physical_steps=review["actual_physical_steps"],
        new_optimizer_updates=0,
        new_neural_inferences=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
