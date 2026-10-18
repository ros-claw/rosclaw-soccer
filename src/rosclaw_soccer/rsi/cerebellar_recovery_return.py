"""Separate learning objective; never rewrite the frozen M0 task/safety label.

Bounded recovery costs use measured current-body states, not a teacher rollout.
The result is a complete-episode Monte Carlo target, not per-step GAE or a
qualification. Raw traces and replay provenance remain the caller's obligation.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.cerebellar_torque_policy import CAPS
from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch, outcome_contract
from rosclaw_soccer.sim.contracts import hash_json


def reward_contract() -> dict[str, Any]:
    value = dict(
        schema="soccer.rsi.cerebellar_recovery_training_return.v1",
        outcome_contract_hash=outcome_contract()["contract_hash"],
        pass_bonus=5.0,
        root_angular_speed_squared_coefficient=0.25,
        mean_joint_speed_squared_coefficient=0.01,
        mean_normalized_applied_torque_squared_coefficient=0.01,
        maximum_recovery_penalty=2.0,
        recovery_window="60 whole frames after first foot-contact frame",
        state_samples="canonical qvel before each actual 2ms integration",
        missing_window_cost_diagnostics=None,
        missing_window_applied_penalty=0.0,
        unknown_window_is_not_stable=True,
        formal_task_or_safety_label_changed=False,
        terminal_bootstrap=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(value, contract_hash=hash_json(value))


def training_return(trace: dict[str, Any]) -> dict[str, Any]:
    outcome = label_first_touch(trace)
    velocity = np.asarray(trace.get("cerebellar_canonical_qvel_before_substep"))
    torque = np.asarray(trace.get("applied_cerebellar_torque_offset_nm"))
    clock = np.asarray(trace.get("cerebellar_before_integration_substep_index"))
    if any(
        raw.shape != shape or raw.dtype != np.float64 or not np.isfinite(raw).all()
        for raw, shape in ((velocity, (300, 1, 10, 41)), (torque, (300, 1, 10, 29)))
    ):
        raise ValueError("complete float64 actual-body neural torque capture required")
    if (
        clock.shape != (300, 1, 10, 1)
        or clock.dtype != np.int64
        or not np.array_equal(clock.ravel(), np.arange(3000))
        or np.any(np.abs(torque) > CAPS + 1e-8)
        or np.max(np.abs(velocity)) > 1e6
    ):
        raise ValueError("causal whole-episode clock and bounded actual torque required")
    contract = reward_contract()
    diagnostics = None
    penalty = 0.0
    if outcome.quality_observed:
        first = outcome.first_foot_contact_frame
        assert first is not None
        section = slice((first + 1) * 10, (first + 61) * 10)
        v = velocity.reshape(3000, 41)[section]
        applied = torque.reshape(3000, 29)[section]
        if len(v) != 600:
            raise ValueError("complete post-contact recovery interval required")
        diagnostics = dict(
            mean_root_angular_speed_squared_rad2_s2=float(np.mean(np.sum(v[:, 3:6] ** 2, axis=1))),
            mean_joint_speed_squared_rad2_s2=float(np.mean(v[:, 6:35] ** 2)),
            mean_normalized_applied_torque_squared=float(np.mean((applied / CAPS) ** 2)),
            actual_substep_samples=600,
        )
        penalty = min(
            2.0,
            0.25 * diagnostics["mean_root_angular_speed_squared_rad2_s2"]
            + 0.01 * diagnostics["mean_joint_speed_squared_rad2_s2"]
            + 0.01 * diagnostics["mean_normalized_applied_torque_squared"],
        )
    result = dict(
        schema="soccer.rsi.cerebellar_measured_training_return.v1",
        reward_contract_hash=contract["contract_hash"],
        formal_outcome=outcome.to_dict(),
        formal_training_reward=outcome.training_reward,
        pass_bonus=5.0 * int(outcome.task_pass),
        recovery_cost_diagnostics=diagnostics,
        applied_recovery_penalty=penalty,
        training_return=float(outcome.training_reward + 5.0 * int(outcome.task_pass) - penalty),
        raw_trace_authenticated_here=False,
        new_physical_replay_performed=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return dict(result, report_hash=hash_json(result))
