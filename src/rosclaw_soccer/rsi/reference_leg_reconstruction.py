"""Retarget only documented tracker-consumed reference fields, never runtime state."""

from typing import Any

import numpy as np

from rosclaw_soccer.training.omnicontact_reference import LAB_TO_MUJOCO

RIGHT_LEG = (
    (4, "right_hip_pitch_link"),
    (7, "right_hip_roll_link"),
    (10, "right_hip_yaw_link"),
    (13, "right_knee_link"),
    (22, "right_ankle_pitch_link"),
    (26, "right_ankle_roll_link"),
)


def shifted_reference_frames(frames: np.ndarray, shift: int, length: int) -> np.ndarray:
    """Shift a reference correction only; does not shift the ball or simulator."""
    if (
        type(frames) is not np.ndarray
        or frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or len(frames) < 3
        or np.any(np.diff(frames) != 1)
        or type(shift) is not int
        or not -20 <= shift <= 20
        or type(length) is not int
    ):
        raise ValueError("bounded integer reference phase shift required")
    shifted = frames + shift
    if shifted[0] < 1 or shifted[-1] >= length - 1:
        raise ValueError("shifted window must remain inside the reference")
    return shifted


def reconstruct_right_leg(
    model: Any, layout: Any, motion: dict[str, np.ndarray], frames: np.ndarray, offsets: np.ndarray
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    """Rebuild six leg bodies and their velocity deltas with native MuJoCo FK.

    Source values of unrelated bodies, ee/contact arrays, robot and ball are
    unchanged; other virtual markers have not been validated. This is a partial
    research reference, not a fully retargeted dataset or a physical trajectory.
    """
    import mujoco

    n = len(motion["joint_pos"])
    if (
        layout.model is not model
        or type(frames) is not np.ndarray
        or frames.ndim != 1
        or frames.dtype.kind not in "iu"
        or len(frames) < 3
        or np.any(np.diff(frames) != 1)
        or frames[0] < 1
        or frames[-1] >= n - 1
        or type(offsets) is not np.ndarray
        or offsets.shape != (len(frames), 6)
        or offsets.dtype != np.float64
        or not np.isfinite(offsets).all()
        or np.max(np.abs(offsets)) > 0.35
        or np.any(offsets[[0, -1]] != 0)
        or not np.array_equal(motion["fps"], [50.0])
    ):
        raise ValueError("bounded interior 50Hz correction with zero endpoints required")
    output = {k: v.copy() for k, v in motion.items()}
    full = np.zeros((n, 6))
    full[frames] = offsets
    delta_vel = np.gradient(full, 0.02, axis=0)
    order = np.asarray(LAB_TO_MUJOCO)[6:12]
    output["joint_pos"][:, order] += full
    output["joint_vel"][:, order] += delta_vel
    d = mujoco.MjData(model)
    body_ids = [int(model.body(layout.robot_prefix + name).id) for _, name in RIGHT_LEG]
    pos_err, quat_err = [], []
    for f in range(n):
        d.qpos[layout.rq : layout.rq + 3] = motion["body_pos_w"][f, 0]
        d.qpos[layout.rq + 3 : layout.rq + 7] = motion["body_quat_w"][f, 0]
        d.qpos[layout.qi] = motion["joint_pos"][f, list(LAB_TO_MUJOCO)]
        d.qvel[layout.rv : layout.rv + 3] = motion["body_lin_vel_w"][f, 0]
        mujoco.mj_kinematics(model, d)
        root_rot = d.xmat[layout.pelvis].reshape(3, 3)
        d.qvel[layout.rv + 3 : layout.rv + 6] = root_rot.T @ motion["body_ang_vel_w"][f, 0]
        d.qvel[layout.vi] = motion["joint_vel"][f, list(LAB_TO_MUJOCO)]
        mujoco.mj_forward(model, d)
        old_vel = []
        for (index, _), body in zip(RIGHT_LEG, body_ids, strict=True):
            pos_err.append(float(np.linalg.norm(d.xpos[body] - motion["body_pos_w"][f, index])))
            quat_err.append(
                float(
                    2
                    * np.arccos(
                        np.clip(abs(np.dot(d.xquat[body], motion["body_quat_w"][f, index])), 0, 1)
                    )
                )
            )
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
            old_vel.append(vel)
        if not np.any(full[f]) and not np.any(delta_vel[f]):
            continue
        d.qpos[layout.qi[6:12]] += full[f]
        d.qvel[layout.vi[6:12]] += delta_vel[f]
        mujoco.mj_forward(model, d)
        for (index, _), body, old in zip(RIGHT_LEG, body_ids, old_vel, strict=True):
            vel = np.zeros(6)
            mujoco.mj_objectVelocity(model, d, mujoco.mjtObj.mjOBJ_BODY, body, vel, 0)
            output["body_pos_w"][f, index] = d.xpos[body]
            output["body_quat_w"][f, index] = d.xquat[body]
            output["body_ang_vel_w"][f, index] += vel[:3] - old[:3]
            output["body_lin_vel_w"][f, index] += vel[3:] - old[3:]
    if max(pos_err) > 1e-3 or max(quat_err) > 0.002:
        raise ValueError("source reference and six native body mappings differ")
    return output, dict(
        max_source_body_position_error_m=max(pos_err),
        max_source_body_angle_error_rad=max(quat_err),
        max_offset_velocity_rad_s=float(np.abs(delta_vel).max()),
    )
