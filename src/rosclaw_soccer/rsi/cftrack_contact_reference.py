"""Absolute ankle-pitch task reference for the trained CFtrack SIM policy.

Only reference arrays change. This is NOT an ankle-roll contact-pose command,
joint-target playback, balance proof or authorization for physical hardware.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_reference_plan import _c2_offsets
from rosclaw_soccer.rsi.omnicontact_policy_copy import cftrack_state_hash


@dataclass(frozen=True)
class CFtrackContactPlan:
    origin_counter: int
    start_frame: int
    peak_control: int
    active_foot: str
    position_world: tuple[float, ...]
    quaternion_world: tuple[float, ...]
    donor_policy_hash: str


def _multiply(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    result: np.ndarray = np.c_[
        a[:, 0] * b[:, 0] - np.sum(a[:, 1:] * b[:, 1:], axis=1),
        a[:, :1] * b[:, 1:] + b[:, :1] * a[:, 1:] + np.cross(a[:, 1:], b[:, 1:]),
    ]
    return result


def condition_cftrack_contact_reference(policy: Any, plan: CFtrackContactPlan) -> dict[str, Any]:
    if (
        type(plan) is not CFtrackContactPlan
        or any(type(i) is not int or i < 0 for i in (plan.origin_counter, plan.start_frame))
        or type(plan.peak_control) is not int
        or not 1 <= plan.peak_control < 48
        or plan.start_frame < plan.origin_counter
        or plan.origin_counter != policy.counter_step
        or plan.active_foot not in ("left", "right")
        or type(plan.position_world) is not tuple
        or len(plan.position_world) != 3
        or type(plan.quaternion_world) is not tuple
        or len(plan.quaternion_world) != 4
        or any(
            type(v) is not float or not np.isfinite(v) or abs(v) > 100
            for v in (*plan.position_world, *plan.quaternion_world)
        )
        or abs(np.linalg.norm(plan.quaternion_world) - 1) > 1e-6
        or policy.task != "kickball"
        or policy.reference_source != "CFgen"
        or policy.async_stage_thread is not None
        or cftrack_state_hash(policy) != plan.donor_policy_hash
    ):
        raise ValueError("immutable current-policy-bound absolute CFtrack pose plan required")
    key = "ref_" + plan.active_foot + "_ankle_future_"
    pos, quat = getattr(policy, key + "pos"), getattr(policy, key + "quat")
    torso = policy.ref_torso_future_pos
    if (
        any(
            type(a) is not np.ndarray or a.dtype.kind != "f" or not np.isfinite(a).all()
            for a in (pos, quat, torso)
        )
        or pos.ndim != 2
        or pos.shape[1] != 3
        or quat.shape != (len(pos), 4)
        or torso.shape != pos.shape
        or plan.start_frame + 48 >= len(pos)
        or (abs(np.linalg.norm(quat, axis=1) - 1) > 1e-5).any()
    ):
        raise ValueError("complete finite normalized49-frame taskspace reference required")
    start, peak = plan.start_frame, plan.peak_control
    old_p = pos[start : start + 49].astype(np.float64)
    old_q = quat[start : start + 49].astype(np.float64)
    normalized = old_q / np.linalg.norm(old_q, axis=1)[:, None]
    fraction, _ = _c2_offsets(np.ones(29), tuple(range(29)), peak)
    w = fraction[:, :1]
    target_p, target_q = np.array(plan.position_world), np.array(plan.quaternion_world)
    new_p = old_p + w * (target_p - old_p)
    # Short-arc SO(3) interpolation respects quaternion double coverage.
    end = np.tile(target_q, (49, 1))
    end[(normalized @ target_q) < 0] *= -1
    dot = np.clip(np.sum(normalized * end, axis=1), 0, 1)
    theta = np.arccos(dot)[:, None]
    denominator = np.sin(theta)
    small = denominator[:, 0] < 1e-8
    safe = np.where(small[:, None], 1.0, denominator)
    new_q = (np.sin((1 - w) * theta) * normalized + np.sin(w * theta) * end) / safe
    new_q[small] = ((1 - w) * normalized + w * end)[small]
    new_q /= np.linalg.norm(new_q, axis=1)[:, None]
    # Boundary equality is exact even when original float32 quaternions have
    # tiny normalization roundoff. No past/following references are rewritten.
    new_p[[0, 48]], new_q[[0, 48]] = old_p[[0, 48]], old_q[[0, 48]]
    encoded_p, encoded_q = new_p.astype(pos.dtype), new_q.astype(quat.dtype)
    offset = encoded_p.astype(np.float64) - old_p
    sampled_speed = float(np.linalg.norm(np.diff(offset, axis=0) / 0.02, axis=1).max())
    encoded_unit = encoded_q.astype(np.float64)
    encoded_unit /= np.linalg.norm(encoded_unit, axis=1)[:, None]
    conjugate = normalized.copy()
    conjugate[:, 1:] *= -1
    correction = _multiply(encoded_unit, conjugate)
    correction /= np.linalg.norm(correction, axis=1)[:, None]
    steps = np.clip(abs(np.sum(correction[1:] * correction[:-1], axis=1)), 0, 1)
    sampled_angular_speed = float((2 * np.arccos(steps) / 0.02).max())
    workspace = np.linalg.norm(encoded_p - torso[start : start + 49], axis=1)
    if (
        sampled_speed > 3.0 + 1e-8
        or sampled_angular_speed > 8.0 + 1e-8
        or workspace.max() > 1.25
        or (encoded_p[:, 2] < -0.05).any()
        or not np.isfinite(encoded_p).all()
        or not np.isfinite(encoded_q).all()
    ):
        raise ValueError(
            "absolute task reference exceeds declared envelope: "
            f"sampled_linear={sampled_speed:.9g}/3m_s, "
            f"sampled_angular={sampled_angular_speed:.9g}/8rad_s, "
            f"torso_workspace={float(workspace.max()):.9g}/1.25m, "
            f"minimum_z={float(encoded_p[:, 2].min()):.9g}/-0.05m"
        )
    point_error = float(np.linalg.norm(encoded_p[peak] - target_p))
    rotation_error = float(2 * np.arccos(np.clip(abs(encoded_unit[peak] @ target_q), 0, 1)))
    if point_error > 1e-6 or rotation_error > 1e-5:
        raise ValueError("absolute taskspace endpoint commitment failed")
    new_full_p, new_full_q = pos.copy(), quat.copy()
    new_full_p[start : start + 49], new_full_q[start : start + 49] = encoded_p, encoded_q
    setattr(policy, key + "pos", new_full_p)
    setattr(policy, key + "quat", new_full_q)
    return dict(
        schema="soccer.cftrack_absolute_ankle_pitch_reference.v1",
        origin_counter=plan.origin_counter,
        start_frame=start,
        peak_control=peak,
        active_foot=plan.active_foot,
        endpoint_position_world=plan.position_world,
        endpoint_quaternion_world=plan.quaternion_world,
        original_policy_hash=plan.donor_policy_hash,
        endpoint_position_error_m=point_error,
        endpoint_rotation_error_rad=rotation_error,
        maximum_position_reference_correction_m=float(np.linalg.norm(offset, axis=1).max()),
        maximum_sampled_additional_reference_speed_m_s=sampled_speed,
        maximum_sampled_additional_reference_angular_speed_rad_s=sampled_angular_speed,
        sampled_reference_speed_limit_m_s=3.0,
        sampled_reference_angular_speed_limit_rad_s=8.0,
        torso_relative_reference_workspace_limit_m=1.25,
        speed_guard_is_sampled_reference_not_continuous_motor_velocity=True,
        desired_link="ankle_pitch_not_ankle_roll",
        other_body_references_unchanged=True,
        entry_past_and_outside_window_unchanged=True,
        actual_world_or_joint_targets_not_modified=True,
        neural_weights_modified=False,
        hardware_authorized=False,
        physically_qualified=False,
    )
