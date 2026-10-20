"""Bind the actual held-gain fields to the predeclared SIM schedule."""

from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_gain_scheduled_episode import G1GainSchedule
from rosclaw_soccer.rsi.switched_velocity_sonic_review import review_switched


def review_gain_switch(
    raw: dict[str, np.ndarray],
    neural: dict[str, np.ndarray],
    params: Any,
    tracker: Any,
    world: Any,
    *,
    handoff: int,
    schedule: G1GainSchedule,
) -> dict[str, Any]:
    if handoff != schedule.handoff_frame:
        raise ValueError("actual handoff must match declared current-state boundary")
    if tuple(tracker.kp) != schedule.final_kp or tuple(tracker.kd) != schedule.final_kd:
        raise ValueError("final gains must be exact qualified SONIC deployment gains")
    if (
        tuple(params.stiffness) != schedule.initial_kp
        or tuple(params.damping) != schedule.initial_kd
    ):
        raise ValueError("initial gains must be exact qualified velocity deployment gains")
    expected = [schedule.gains(f) for f in range(300)]
    for index, key in enumerate(("held_pd_kp", "held_pd_kd")):
        a = raw.get(key)
        if (
            type(a) is not np.ndarray
            or a.shape != (300, 29)
            or a.dtype != np.float64
            or not np.array_equal(a, np.array([row[index] for row in expected]))
        ):
            raise ValueError("every actual held gain must match exact predeclared schedule")
    common = {k: v for k, v in raw.items() if k not in ("held_pd_kp", "held_pd_kd")}
    if schedule.mode == "constant":
        # Preserve old metadata byte values for the constant-schedule qualifier.
        return review_switched(common, neural, params, tracker, world, handoff=handoff)
    result = review_switched(
        common,
        neural,
        params,
        tracker,
        world,
        handoff=handoff,
        held_gains=(raw["held_pd_kp"], raw["held_pd_kd"]),
    )
    return dict(
        result,
        all_declared_pd_and_policy_inputs_reconstructed=True,
        gain_schedule_hash=schedule.contract_hash,
        gain_schedule=schedule.report(),
    )
