"""Convert pre-retargeted G1 kinematics into experimental native FK references.

This is not motion tracking, dynamics, a receiving demonstration, or contact
evidence. MotionDecode lacks ball/contact truth; its source license applies.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout


def _slerp(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    dot = float(np.dot(a, b))
    if dot < 0:
        b, dot = -b, -dot
    dot = float(np.clip(dot, -1, 1))
    if dot > 0.9995:
        result = a + fraction * (b - a)
        return result / np.linalg.norm(result)
    angle = float(np.arccos(dot))
    return (np.sin((1 - fraction) * angle) * a + np.sin(fraction * angle) * b) / np.sin(angle)


def native_motion_reference(
    *, model: Any, values: np.ndarray, source_fps: float, target_fps: float = 50.0
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Resample canonical root+HG29 CSV and derive 14 bodies using native FK.

    Source FPS must be declared, not inferred from the CSV. Root XY is made
    translation-relative; root Z and all rotations and joint angles remain in
    the source frame. Quaternion interpolation is spherical, velocity comes
    from native generalized-coordinate differentiation at the target rate.
    This function does not set state on any active episode.
    """
    if (
        not isinstance(values, np.ndarray)
        or values.ndim != 2
        or values.shape[1] != 36
        or len(values) < 4
        or not np.isfinite(values).all()
        or type(source_fps) is not float
        or type(target_fps) is not float
        or not 1.0 <= target_fps <= source_fps <= 240.0
    ):
        raise ValueError("finite explicit root/HG29 and source/target rates required")
    norm = np.linalg.norm(values[:, 3:7], axis=1)
    if np.max(np.abs(norm - 1)) > 1e-4:
        raise ValueError("unit source quaternion required")
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    time = np.arange(len(values), dtype=np.float64) / source_fps
    target = np.arange(int(np.floor(time[-1] * target_fps)) + 1, dtype=np.float64) / target_fps
    if len(target) < 3:
        raise ValueError("at least three resampled frames required")
    q = np.repeat(model.qpos0[None, :], len(target), axis=0)
    for dim in range(3):
        q[:, layout.rq + dim] = np.interp(target, time, values[:, dim])
    q[:, layout.rq : layout.rq + 2] -= q[0, layout.rq : layout.rq + 2]
    source_quat = values[:, 3:7] / norm[:, None]
    for i, t in enumerate(target):
        left = min(int(np.floor(t * source_fps)), len(values) - 2)
        fraction = (t - time[left]) * source_fps
        q[i, layout.rq + 3 : layout.rq + 7] = _slerp(
            source_quat[left], source_quat[left + 1], float(fraction)
        )
    for j, joint in enumerate(layout.qi):
        q[:, joint] = np.interp(target, time, values[:, 7 + j])
    ranges = model.jnt_range[layout.joints]
    if np.any(q[:, layout.qi] < ranges[:, 0] - 1e-5) or np.any(
        q[:, layout.qi] > ranges[:, 1] + 1e-5
    ):
        raise ValueError("source motion exceeds native joint limits")
    v = np.zeros((len(target), model.nv))
    for i in range(len(target) - 1):
        mujoco.mj_differentiatePos(model, v[i], 1.0 / target_fps, q[i], q[i + 1])
    v[-1] = v[-2]
    ids = [int(model.body("robot/" + name).id) for name in TRACKING_BODIES]
    data = mujoco.MjData(model)
    position, quaternion, linear, angular = [], [], [], []
    jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
    for qi, vi in zip(q, v, strict=True):
        data.qpos[:] = qi
        data.qvel[:] = vi
        mujoco.mj_forward(model, data)
        position.append(data.xpos[ids].copy())
        quaternion.append(data.xquat[ids].copy())
        frame_linear, frame_angular = [], []
        for body in ids:
            mujoco.mj_jacBody(model, data, jp, jr, body)
            frame_linear.append(jp @ vi)
            frame_angular.append(jr @ vi)
        linear.append(frame_linear)
        angular.append(frame_angular)
    ref = {
        name: np.asarray(value, dtype=np.float32)
        for name, value in dict(
            position=position, quaternion=quaternion, linear=linear, angular=angular
        ).items()
    }
    plan = {
        "joint": q[:, layout.qi].astype(np.float32),
        "joint_velocity": v[:, layout.vi].astype(np.float32),
        "native_qpos": q.copy(),
        "native_qvel": v.copy(),
    }
    for name, body in (("pelvis", "pelvis"), ("torso", "torso_link")):
        index = TRACKING_BODIES.index(body)
        for field in ref:
            plan[name + "_" + field] = ref[field][:, index].copy()
    return plan, ref
