"""Scratch-only SIM kinematics for causal foot/ball motion observations."""

import numpy as np


def first_contact_control_entry(
    qpos: np.ndarray,
    qvel: np.ndarray,
    force: np.ndarray,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, int, tuple[int, ...]] | None:
    """Extract the *control-entry* state before the first observed receiver contact.

    Event-based retrospective selection is not itself a deployable policy and
    no post-control state is permitted in the returned observation.
    """
    n = len(qpos)
    if (
        not 1 <= n <= 600
        or qpos.shape != (n, 10, 79)
        or qvel.shape != (n, 10, 76)
        or force.shape != (n, 10, 2, 3)
        or initial_qpos.shape != (79,)
        or initial_qvel.shape != (76,)
        or any(x.dtype != np.float64 for x in (qpos, qvel, force, initial_qpos, initial_qvel))
        or any(not np.isfinite(x).all() for x in (qpos, qvel, force, initial_qpos, initial_qvel))
        or (force < 0).any()
    ):
        raise ValueError("exact finite shared physical trajectory and initial state required")
    events = np.argwhere((force[:, :, 1, :] > 1).any(axis=2))
    if not len(events):
        return None
    control, substep = map(int, events[0])
    categories = tuple(int(i) for i in np.flatnonzero(force[control, substep, 1] > 1))
    q = initial_qpos if control == 0 else qpos[control - 1, -1]
    v = initial_qvel if control == 0 else qvel[control - 1, -1]
    return q.copy(), v.copy(), control, substep, categories


def measure_foot_ball_kinematics(
    model,
    qpos: np.ndarray,
    qvel: np.ndarray,
    *,
    foot_body_names: tuple[str, str],
    ball_body_name: str,
    ball_dofadr: int,
) -> np.ndarray:
    """Two ankle-fixed reference points; columns are pos/velocity relative to ball.

    Points are not inferred contact locations and values do not include the
    ball's surface angular velocity. This uses fresh scratch MjData only and
    cannot step a world or write a controller's state.
    """
    import mujoco

    if (
        qpos.shape != (model.nq,)
        or qvel.shape != (model.nv,)
        or qpos.dtype != np.float64
        or qvel.dtype != np.float64
        or not np.isfinite(qpos).all()
        or not np.isfinite(qvel).all()
        or len(foot_body_names) != 2
        or len(set(foot_body_names)) != 2
        or any(not isinstance(x, str) or not x for x in (*foot_body_names, ball_body_name))
        or type(ball_dofadr) is not int
        or not 0 <= ball_dofadr <= model.nv - 6
    ):
        raise ValueError("finite exact native kinematic state and explicit bodies required")
    foot_ids = [int(model.body(name).id) for name in foot_body_names]
    ball_id = int(model.body(ball_body_name).id)
    joint = int(model.body_jntadr[ball_id])
    if (
        joint < 0
        or model.jnt_type[joint] != int(mujoco.mjtJoint.mjJNT_FREE)
        or int(model.jnt_dofadr[joint]) != ball_dofadr
        or ball_id in foot_ids
    ):
        raise ValueError("ball linear velocity must be the actual free-body translation")
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = qpos, qvel
    mujoco.mj_forward(model, data)
    ball_position = data.xpos[ball_id].copy()
    ball_velocity = qvel[ball_dofadr : ball_dofadr + 3].copy()
    observations = []
    for side, body in enumerate(foot_ids):
        local = np.array([0.13, -0.02 if side == 0 else 0.02, -0.025])
        point = data.xpos[body] + data.xmat[body].reshape(3, 3) @ local
        jacobian = np.zeros((3, model.nv))
        mujoco.mj_jac(model, data, jacobian, None, point, body)
        observations.append(
            np.concatenate((point - ball_position, jacobian @ qvel - ball_velocity))
        )
    result = np.stack(observations).astype(np.float64)
    if result.shape != (2, 6) or not np.isfinite(result).all():
        raise ValueError("finite left/right relative point observations required")
    return result
