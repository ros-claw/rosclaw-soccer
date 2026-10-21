"""Independent source Torch reward/guard formulas on actual native body states.

No physical steps or policy calls. Original source modules must be pinned by
the calling runner BEFORE importing this optional verification helper.
"""

from types import SimpleNamespace


def source_tracking_oracle(course, frames):
    import mujoco
    import numpy as np
    import torch
    from mjlab.tasks.tracking.mdp import rewards
    from mjlab.utils.lab_api.math import quat_apply, quat_inv, quat_mul, yaw_quat
    from src.tasks.soccer.mdp import shared_terminations as term

    from rosclaw_soccer.rsi.native_course_tracking import TRACKING_BODIES

    def tensor(value):
        return torch.as_tensor(np.asarray(value), dtype=torch.float64)

    pos, quat, lin, ang, actual_pos, actual_quat, actual_lin, actual_ang, gravity = (
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
        [],
    )
    for i, data in enumerate(course.batch.data):
        index = course.index[i]
        ref = course.tracking.references[int(course.bank["source_id"][index])]
        f = int(frames[i])
        pos.append(ref["position"][f] + course.bank["origin"][index])
        quat.append(ref["quaternion"][f])
        lin.append(ref["linear"][f])
        ang.append(ref["angular"][f])
        actual_pos.append(data.xpos[course.tracking.ids].copy())
        actual_quat.append(data.xquat[course.tracking.ids].copy())
        vl, va = [], []
        for b in course.tracking.ids:
            jp, jr = np.zeros((3, course.model.nv)), np.zeros((3, course.model.nv))
            mujoco.mj_jacBody(course.model, data, jp, jr, b)
            vl.append(jp @ data.qvel)
            va.append(jr @ data.qvel)
        actual_lin.append(vl)
        actual_ang.append(va)
        gravity.append(-data.xmat[course.layout.pelvis].reshape(3, 3)[2])
    pos, quat, lin, ang, actual_pos, actual_quat, actual_lin, actual_ang = map(
        tensor, (pos, quat, lin, ang, actual_pos, actual_quat, actual_lin, actual_ang)
    )
    anchor = course.tracking.anchor
    delta = actual_pos[:, anchor].clone()
    delta[:, 2] = pos[:, anchor, 2]
    orientation = yaw_quat(quat_mul(actual_quat[:, anchor], quat_inv(quat[:, anchor])))
    orientation = orientation[:, None].expand(-1, 14, -1)
    relative_position = delta[:, None] + quat_apply(orientation, pos - pos[:, anchor : anchor + 1])
    relative_orientation = quat_mul(orientation, quat)
    command = SimpleNamespace(
        cfg=SimpleNamespace(body_names=TRACKING_BODIES),
        anchor_pos_w=pos[:, anchor],
        anchor_quat_w=quat[:, anchor],
        robot_anchor_pos_w=actual_pos[:, anchor],
        robot_anchor_quat_w=actual_quat[:, anchor],
        body_pos_relative_w=relative_position,
        body_quat_relative_w=relative_orientation,
        robot_body_pos_w=actual_pos,
        robot_body_quat_w=actual_quat,
        body_lin_vel_w=lin,
        body_ang_vel_w=ang,
        robot_body_lin_vel_w=actual_lin,
        robot_body_ang_vel_w=actual_ang,
    )
    env = SimpleNamespace(
        num_envs=course.worlds,
        device="cpu",
        command_manager=SimpleNamespace(get_term=lambda name: command),
        scene={"robot": SimpleNamespace(data=SimpleNamespace(projected_gravity_b=tensor(gravity)))},
    )
    values = torch.stack(
        (
            rewards.motion_global_anchor_orientation_error_exp(env, "motion", 0.4),
            rewards.motion_relative_body_position_error_exp(env, "motion", 0.3),
            rewards.motion_relative_body_orientation_error_exp(env, "motion", 0.4),
            rewards.motion_global_body_linear_velocity_error_exp(env, "motion", 1.0),
            rewards.motion_global_body_angular_velocity_error_exp(env, "motion", 3.14),
        ),
        dim=1,
    ).numpy()
    guards = torch.stack(
        (
            term.bad_orientation(env, np.deg2rad(70)),
            term.bad_anchor_pos_z(env, threshold=0.25),
            term.bad_anchor_ori(env, threshold=0.8),
            term.bad_ee_body_pos_z(env, threshold=0.25),
        ),
        dim=1,
    ).numpy()
    return values, guards
