"""Measured pre-entry state history for the SIM-only native kick adapter.

Prior PD targets are converted into the new policy's action coordinates, not
mislabelled as historical native policy outputs. The newly admitted task goal
is reprojected through prior measured poses; it is not a past goal observation.
"""

from __future__ import annotations

import importlib
import math
from collections import deque
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class KickMeasuredSample:
    frame: int
    time_sec: float
    previous_target_frame: int
    joint_position: tuple[float, ...]
    joint_velocity: tuple[float, ...]
    angular_velocity: tuple[float, ...]
    pelvis_position: tuple[float, ...]
    pelvis_quaternion: tuple[float, ...]
    ball_position: tuple[float, ...]
    previous_pd_target: tuple[float, ...]

    def __post_init__(self) -> None:
        if (
            type(self.frame) is not int
            or self.frame < 1
            or type(self.previous_target_frame) is not int
            or self.previous_target_frame != self.frame - 1
            or type(self.time_sec) not in (float, int)
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0
        ):
            raise ValueError("measured frame and preceding applied-target frame required")
        for values, size in (
            (self.joint_position, 29),
            (self.joint_velocity, 29),
            (self.angular_velocity, 3),
            (self.pelvis_position, 3),
            (self.pelvis_quaternion, 4),
            (self.ball_position, 3),
            (self.previous_pd_target, 29),
        ):
            if (
                type(values) is not tuple
                or len(values) != size
                or any(
                    type(v) not in (float, int) or not math.isfinite(v) or abs(v) > 1e4
                    for v in values
                )
            ):
                raise ValueError("immutable bounded measured vectors required")
        if abs(np.linalg.norm(self.pelvis_quaternion) - 1) > 1e-6:
            raise ValueError("unit measured pelvis quaternion required")

    def numeric_record(self) -> np.ndarray:
        """Lossless float64 trace row; small integer frame fields remain exact."""
        return np.asarray(
            (
                self.frame,
                self.time_sec,
                self.previous_target_frame,
                *self.joint_position,
                *self.joint_velocity,
                *self.angular_velocity,
                *self.pelvis_position,
                *self.pelvis_quaternion,
                *self.ball_position,
                *self.previous_pd_target,
            ),
            dtype=np.float64,
        )


def _rotation(quaternion: tuple[float, ...]) -> np.ndarray:
    w, x, y, z = quaternion
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def install_measured_kick_history(
    policy: Any, samples: tuple[KickMeasuredSample, ...], *, frame: int, time_sec: float
) -> str:
    """Install four previous frames; the next inference appends the current one.

    All checks and projections precede buffer mutation. No inference, physics,
    root pose, ball state, gains, or torque limits are modified here.
    """
    if policy.runtime_mode != "sim" or policy.use_body_frame_ball:
        raise ValueError("measured kick history is simulation-only")
    if (
        type(frame) is not int
        or frame < 5
        or type(time_sec) not in (int, float)
        or not math.isfinite(time_sec)
        or type(samples) is not tuple
        or len(samples) != 5
        or any(not isinstance(s, KickMeasuredSample) for s in samples)
        or tuple(s.frame for s in samples) != tuple(range(frame - 4, frame + 1))
        or abs(samples[-1].time_sec - time_sec) > 1e-9
        or any(
            abs(b.time_sec - a.time_sec - 0.02) > 1e-6
            for a, b in zip(samples, samples[1:], strict=False)
        )
    ):
        raise ValueError("five contiguous actual 50 Hz samples ending at entry required")
    for sample in samples:
        sample.__post_init__()
    module = importlib.import_module(type(policy).__module__)
    mapping = np.asarray(module.ISAAC_TO_MUJOCO)
    default = np.asarray(policy.default_q_mj, dtype=float)
    scale = np.asarray(policy.action_scale_mj, dtype=float)
    low = np.asarray(policy.action_clip_lo_il, dtype=float)
    high = np.asarray(policy.action_clip_hi_il, dtype=float)
    goal = np.asarray(policy.target_pos_w, dtype=float)
    bias = float(policy.state_cmd.target_y_bias)
    if (
        mapping.shape != (29,)
        or mapping.dtype.kind not in "iu"
        or sorted(mapping.tolist()) != list(range(29))
        or any(
            a.shape != (29,) or not np.isfinite(a).all() or np.any(np.abs(a) > 1e4)
            for a in (default, scale, low, high)
        )
        or np.any(np.abs(scale) < 1e-9)
        or np.any(low > high)
        or goal.shape != (3,)
        or not np.isfinite(goal).all()
        or np.any(np.abs(goal) > 1e4)
        or not math.isfinite(bias)
        or abs(bias) > 1e4
        or not np.array_equal(default[mapping], np.asarray(policy.default_q_il))
    ):
        raise ValueError("qualified finite policy mapping, scaling and entry goal required")
    names = (
        "_ang_vel_buf",
        "_jpos_buf",
        "_jvel_buf",
        "_action_buf",
        "_ball_pos_buf",
        "_target_pos_buf",
    )
    if any(
        not isinstance(getattr(policy, n), deque) or getattr(policy, n).maxlen != 5 for n in names
    ):
        raise ValueError("native five-frame history buffers required")
    current = samples[-1]
    for measured, actual in (
        (current.joint_position, policy.state_cmd.q),
        (current.joint_velocity, policy.state_cmd.dq),
        (current.angular_velocity, policy.state_cmd.root_ang_vel_b),
        (current.pelvis_position, policy.state_cmd.pelvis_pos_w),
        (current.pelvis_quaternion, policy.state_cmd.pelvis_quat_w),
        (current.ball_position, policy.state_cmd.ball_pos_w),
    ):
        if not np.array_equal(np.asarray(measured), np.asarray(actual)):
            raise ValueError("history must end at the actual entry observation")
    rows = []
    clipped = 0
    for sample in samples:
        r = _rotation(sample.pelvis_quaternion)
        raw = ((np.asarray(sample.previous_pd_target) - default) / scale)[mapping]
        action = np.clip(raw, low, high).astype(np.float32)
        clipped += int(np.count_nonzero((raw < low) | (raw > high)))
        ball = np.clip(
            r.T @ (np.asarray(sample.ball_position) - sample.pelvis_position), -8, 8
        ).astype(np.float32)
        target = np.clip(
            (r.T @ (goal - sample.pelvis_position)).astype(np.float32)
            + np.array([0, bias, 0], dtype=np.float32),
            -8,
            8,
        ).astype(np.float32)
        rows.append(
            (
                np.asarray(sample.angular_velocity, dtype=np.float32),
                (np.asarray(sample.joint_position)[mapping] - default[mapping]).astype(np.float32),
                np.asarray(sample.joint_velocity, dtype=np.float32)[mapping],
                action,
                ball,
                target,
            )
        )
    if any(not np.isfinite(value).all() for row in rows for value in row):
        raise ValueError("finite projected history required before installation")
    receipt = str(
        hash_json(
            dict(
                samples=[asdict(s) for s in samples],
                goal_at_entry=goal.tolist(),
                target_y_bias=bias,
                projection=dict(
                    mapping=mapping.tolist(),
                    default=default.tolist(),
                    scale=scale.tolist(),
                    low=low.tolist(),
                    high=high.tolist(),
                ),
                projected_rows=[[value.tolist() for value in row] for row in rows],
                clipped_projected_action_components=clipped,
                history_semantics="measured_state_and_previous_pd_target_with_entry_goal_reprojection",
                activation_ceiling="SIM_ONLY",
            )
        )
    )
    for index, name in enumerate(names):
        buffer = getattr(policy, name)
        buffer.clear()
        buffer.extend(row[index].copy() for row in rows[:-1])
    policy.last_action_il = rows[-1][3].copy()
    return receipt
