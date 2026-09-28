"""Independent causal trace audit of SIM_ONLY task-space swing targets."""

from __future__ import annotations

from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    FORWARD_CAPS_M,
    LATERAL_CAPS_M,
    MAX_JOINT_DELTA_RAD,
    SWING_ACQUISITION_MAX_GAPS_M,
    VERTICAL_OFFSETS_M,
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)

LEG_NAMES = tuple(
    tuple(
        f"{side}_{part}_joint"
        for part in ("hip_pitch", "hip_roll", "hip_yaw", "knee", "ankle_pitch", "ankle_roll")
    )
    for side in ("left", "right")
)


def audit_taskspace_swing_trace(
    replay: Any,
    report: dict[str, Any],
    *,
    frames: int,
    count: int,
) -> dict[str, Any]:
    forward = report.get("taskspace_forward_m")
    lateral = report.get("taskspace_lateral_cap_m", 0.05)
    vertical = report.get("taskspace_vertical_offset_m", 0.0)
    acquisition_gap = report.get("taskspace_acquisition_max_gap_m", 0.95)
    family_hash = report.get("taskspace_family_actor_hash")
    family_actions = report.get("selected_taskspace_actions")
    order = report.get("taskspace_joint_order")
    mask_raw = report.get("selected_taskspace_mask")
    if mask_raw is None and report.get("taskspace_gate_actor_hash") is None and family_hash is None:
        mask_raw = [True] * count
    if family_hash is not None and (
        not isinstance(family_actions, list)
        or len(family_actions) != count
        or any(type(value) is not int or value not in (0, 1, 2) for value in family_actions)
        or mask_raw != [value != 0 for value in family_actions]
    ):
        raise ValueError("uncommitted per-lane task-space family action")
    if family_hash is None and family_actions is not None:
        raise ValueError("unbound task-space family action")
    if (
        forward not in FORWARD_CAPS_M
        or lateral not in LATERAL_CAPS_M
        or vertical not in VERTICAL_OFFSETS_M
        or acquisition_gap not in SWING_ACQUISITION_MAX_GAPS_M
        or report.get("taskspace_leg_joint_names") != [list(row) for row in LEG_NAMES]
        or not isinstance(order, list)
        or len(order) != 29
        or set(order) != set(G1_DDS_JOINT_NAMES)
        or not isinstance(mask_raw, list)
        or len(mask_raw) != count
        or any(type(value) is not bool for value in mask_raw)
    ):
        raise ValueError("uncommitted task-space G1 joint contract")
    mask = np.asarray(mask_raw, dtype=np.bool_)
    shapes = {
        "pre_step_foot_link_position_w": (frames, count, 2, 3),
        "pre_step_foot_linear_jacobian_w": (frames, count, 2, 3, 6),
        "taskspace_selected_side": (frames, count),
        "applied_taskspace_joint_delta_rad": (frames, count, 29),
        "baseline_taskspace_joint_target_rad": (frames, count, 29),
        "executed_taskspace_joint_target_rad": (frames, count, 29),
        "taskspace_joint_limits_rad": (count, 29, 2),
    }
    if any(
        key not in replay or replay[key].shape != shape or not np.isfinite(replay[key]).all()
        for key, shape in shapes.items()
    ):
        raise ValueError("invalid task-space physical trace")
    feet = replay["pre_step_foot_link_position_w"]
    jacobians = replay["pre_step_foot_linear_jacobian_w"]
    selected_sides = replay["taskspace_selected_side"]
    residual = replay["applied_taskspace_joint_delta_rad"]
    baseline = replay["baseline_taskspace_joint_target_rad"]
    executed = replay["executed_taskspace_joint_target_rad"]
    limits = replay["taskspace_joint_limits_rad"]
    ball_local = replay["pre_step_ball_position_local_m"]
    forces = replay["observed_ball_body_contact_force_peak_n"]
    joint_ids = tuple(tuple(order.index(name) for name in row) for row in LEG_NAMES)
    sonic_ids = [order.index(name) for name in G1_DDS_JOINT_NAMES]
    if (
        not np.isin(selected_sides, (-1, 0, 1)).all()
        or np.max(np.abs(residual)) > MAX_JOINT_DELTA_RAD + 1e-5
        or not np.allclose(executed, baseline + residual, atol=2e-5, rtol=0)
        or not np.allclose(
            baseline[:, :, sonic_ids],
            replay["predicted_baseline_joint_target_rad"],
            atol=2e-5,
            rtol=0,
        )
    ):
        raise ValueError("task-space joint action differs from frozen SONIC target")
    first_contact = np.full(count, -1, dtype=np.int64)
    contact_delta = np.zeros((count, 6))
    side = np.full(count, -1, dtype=np.int64)
    applied_frames = 0
    for frame in range(frames):
        for lane in range(count):
            ball_world = ball_local[frame, lane].copy()
            ball_world[1] += lane * 8.0
            if mask[lane] and first_contact[lane] < 0:
                side[lane] = choose_swing_side(
                    feet[frame, lane],
                    ball_world,
                    int(side[lane]),
                    acquisition_max_gap_m=acquisition_gap,
                )
            if selected_sides[frame, lane] != side[lane]:
                raise ValueError("task-space side used future contact or altered support leg")
            expected = np.zeros(29)
            if side[lane] >= 0:
                ids = list(joint_ids[int(side[lane])])
                if first_contact[lane] >= 0:
                    delta = release_joint_delta(
                        contact_delta[lane], frame - int(first_contact[lane])
                    )
                else:
                    delta = swing_joint_delta(
                        feet[frame, lane, side[lane]],
                        ball_world,
                        jacobians[frame, lane, side[lane]],
                        baseline[frame, lane, ids],
                        limits[lane, ids],
                        forward_cap_m=forward,
                        lateral_cap_m=(
                            0.10
                            if family_hash is not None
                            and family_actions is not None
                            and family_actions[lane] == 2
                            else lateral
                        ),
                        vertical_offset_m=(
                            0.04
                            if family_hash is not None
                            and family_actions is not None
                            and family_actions[lane] == 1
                            else vertical
                        ),
                    )
                    delta = (baseline[frame, lane, ids] + delta).astype(np.float32).astype(
                        float
                    ) - baseline[frame, lane, ids]
                expected[ids] = delta
            if not np.allclose(residual[frame, lane], expected, atol=2e-5, rtol=0):
                raise ValueError("task-space residual differs from measured causal state")
            applied_frames += int(np.any(np.abs(expected) > 1e-6))
            if first_contact[lane] < 0 and np.any(forces[frame, lane] > 1.0):
                first_contact[lane] = frame
                if side[lane] >= 0:
                    contact_delta[lane] = residual[frame, lane, list(joint_ids[int(side[lane])])]
    return {
        "taskspace_action_audited": True,
        "taskspace_applied_lane_frames": applied_frames,
        "taskspace_max_joint_delta_rad": float(np.max(np.abs(residual))),
    }
