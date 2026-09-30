"""Independent SIM_ONLY audit of support-knee action and frozen swing path."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.support_knee_nullspace import support_knee_nullspace_delta
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES, audit_taskspace_swing_trace


def audit_support_knee_action_trace(
    replay: Any, report: dict[str, Any], *, frames: int, count: int
) -> dict[str, Any]:
    """Recompute each support target from saved same-frame Jacobians and contact."""
    retreat = report.get("support_knee_retract_m")
    order = report.get("taskspace_joint_order")
    mask = report.get("selected_taskspace_mask")
    if (
        retreat not in (0.04, 0.08)
        or not isinstance(order, list)
        or len(order) != 29
        or not isinstance(mask, list)
        or len(mask) != count
        or any(type(value) is not bool for value in mask)
        or frames < 1
        or count < 1
    ):
        raise ValueError("uncommitted SIM_ONLY support-knee contract")
    shapes = {
        "pre_step_foot_link_position_w": (frames, count, 2, 3),
        "pre_step_foot_linear_jacobian_w": (frames, count, 2, 3, 6),
        "pre_step_knee_x_jacobian_w": (frames, count, 2, 6),
        "taskspace_selected_side": (frames, count),
        "applied_support_knee_joint_delta_rad": (frames, count, 29),
        "applied_taskspace_joint_delta_rad": (frames, count, 29),
        "baseline_taskspace_joint_target_rad": (frames, count, 29),
        "executed_taskspace_joint_target_rad": (frames, count, 29),
        "taskspace_joint_limits_rad": (count, 29, 2),
        "observed_ball_body_contact_force_peak_n": (frames, count, 6),
    }
    if any(
        key not in replay or replay[key].shape != shape or not np.isfinite(replay[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("missing finite support-knee physical trace")
    feet = replay["pre_step_foot_link_position_w"]
    foot_jacobians = replay["pre_step_foot_linear_jacobian_w"]
    knee_jacobians = replay["pre_step_knee_x_jacobian_w"]
    sides = replay["taskspace_selected_side"]
    applied = replay["applied_support_knee_joint_delta_rad"]
    baseline = replay["baseline_taskspace_joint_target_rad"]
    limits = replay["taskspace_joint_limits_rad"]
    forces = replay["observed_ball_body_contact_force_peak_n"]
    joint_ids = tuple(tuple(order.index(name) for name in row) for row in LEG_NAMES)
    first_contact = np.full(count, -1, dtype=np.int64)
    applied_lane_frames = 0
    peak_foot_drift = 0.0
    peak_knee_retract = 0.0
    for frame in range(frames):
        for lane in range(count):
            expected = np.zeros(29, dtype=np.float64)
            side = int(sides[frame, lane])
            if side not in (-1, 0, 1):
                raise ValueError("invalid measured support/swing side")
            if mask[lane] and side >= 0 and first_contact[lane] < 0:
                support = 1 - side
                ids = list(joint_ids[support])
                proposal = support_knee_nullspace_delta(
                    foot_jacobians[frame, lane, support],
                    knee_jacobians[frame, lane, support],
                    baseline[frame, lane, ids],
                    limits[lane, ids],
                    retract_m=retreat,
                    support_foot_grounded=bool(feet[frame, lane, support, 2] < 0.10),
                    swing_foot_airborne=bool(
                        feet[frame, lane, side, 2] - feet[frame, lane, support, 2] >= 0.02
                    ),
                )
                if not proposal.abstained:
                    expected[ids] = (
                        baseline[frame, lane, ids] + np.asarray(proposal.joint_delta_rad)
                    ).astype(np.float32).astype(np.float64) - baseline[frame, lane, ids]
                    peak_foot_drift = max(peak_foot_drift, proposal.predicted_foot_drift_m)
                    peak_knee_retract = max(peak_knee_retract, proposal.predicted_knee_retract_m)
            if not np.allclose(applied[frame, lane], expected, atol=2e-5, rtol=0):
                raise ValueError(f"support-knee joint action drift frame={frame} lane={lane}")
            applied_lane_frames += int(np.any(np.abs(expected) > 1e-6))
            if first_contact[lane] < 0 and np.any(forces[frame, lane] > 1.0):
                first_contact[lane] = frame
    shadow = {key: replay[key] for key in replay.files}
    shadow["applied_taskspace_joint_delta_rad"] = (
        replay["applied_taskspace_joint_delta_rad"] - applied
    )
    shadow["executed_taskspace_joint_target_rad"] = (
        replay["executed_taskspace_joint_target_rad"] - applied
    )
    swing_audit = audit_taskspace_swing_trace(shadow, report, frames=frames, count=count)
    return {
        "support_knee_action_audited": True,
        "support_knee_applied_lane_frames": applied_lane_frames,
        "support_knee_peak_predicted_foot_drift_m": peak_foot_drift,
        "support_knee_peak_predicted_retract_m": peak_knee_retract,
        "frozen_swing_audit": swing_audit,
    }
