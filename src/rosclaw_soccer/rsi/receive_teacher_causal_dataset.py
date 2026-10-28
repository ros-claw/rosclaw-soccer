"""Causal native-step observations paired with actually applied teacher torque."""

import numpy as np

FEATURE_DIM = 140


def causal_teacher_samples(
    *,
    initial_qpos: np.ndarray,
    initial_qvel: np.ndarray,
    initial_actuator_torque: np.ndarray,
    qpos: np.ndarray,
    qvel: np.ndarray,
    actual_torque: np.ndarray,
    parent_torque: np.ndarray,
    ball_forces: np.ndarray,
    receiver_layout,
    teammate_layout,
) -> tuple[np.ndarray, np.ndarray]:
    """Use the actual state BEFORE each 2ms action, never future contact labels.

    The initial snapshot must have no prior receiver-ball contact. Parent
    torque is the same boundary's actor/PD proposal; previous torque and
    contact derive strictly from earlier physical steps. This is teacher
    imitation data, not successful reception or on-policy RL data.
    """
    if qpos.ndim != 3 or qpos.shape[1] != 10 or len(qpos) < 1:
        raise ValueError("ten native substeps for every control required")
    n = qpos.shape[0] * 10
    nq, nv, nu = qpos.shape[-1], qvel.shape[-1], actual_torque.shape[-1]
    if (
        initial_qpos.shape != (nq,)
        or initial_qvel.shape != (nv,)
        or initial_actuator_torque.shape != (nu,)
        or qvel.shape != (n // 10, 10, nv)
        or actual_torque.shape != (n // 10, 10, nu)
        or parent_torque.shape != actual_torque.shape
        or ball_forces.shape != (n // 10, 10, 2, 3)
        or not all(
            np.isfinite(x).all()
            for x in (
                initial_qpos,
                initial_qvel,
                initial_actuator_torque,
                qpos,
                qvel,
                actual_torque,
                parent_torque,
                ball_forces,
            )
        )
        or (ball_forces < 0).any()
    ):
        raise ValueError("finite complete aligned native states and torques required")
    q_after, v_after = qpos.reshape(n, nq), qvel.reshape(n, nv)
    applied = actual_torque.reshape(n, nu)
    parent = parent_torque.reshape(n, nu)
    before_q = np.concatenate((initial_qpos[None], q_after[:-1]), axis=0)
    before_v = np.concatenate((initial_qvel[None], v_after[:-1]), axis=0)
    previous_torque = np.concatenate((initial_actuator_torque[None], applied[:-1]), axis=0)
    forces = ball_forces.reshape(n, 2, 3)[:, 1, :]
    previous_contact = np.concatenate((np.zeros((1, 3), dtype=bool), forces[:-1] > 1.0))
    seen = np.zeros(n, dtype=np.float64)
    age = np.ones(n, dtype=np.float64)
    last = None
    for step in range(n):
        if step and (forces[step - 1, :2] > 1.0).any():
            last = step - 1
        if last is not None:
            seen[step] = 1.0
            age[step] = min((step - last) / 150.0, 1.0)
    receiver, teammate = receiver_layout, teammate_layout
    parts = (
        before_q[:, receiver.qi],
        before_v[:, receiver.vi],
        before_q[:, receiver.rq + 3 : receiver.rq + 7],
        before_q[:, receiver.rq + 2 : receiver.rq + 3],
        before_v[:, receiver.rv : receiver.rv + 6],
        before_q[:, receiver.bq : receiver.bq + 3] - before_q[:, receiver.rq : receiver.rq + 3],
        before_v[:, receiver.bv : receiver.bv + 3],
        before_q[:, teammate.rq : teammate.rq + 2] - before_q[:, receiver.rq : receiver.rq + 2],
        previous_torque[:, receiver.ai],
        parent[:, receiver.ai],
        previous_contact.astype(np.float64),
        seen[:, None],
        age[:, None],
    )
    features = np.concatenate(parts, axis=1)
    targets = applied[:, receiver.ai] - parent[:, receiver.ai]
    if features.shape != (n, FEATURE_DIM) or targets.shape != (n, 29):
        raise ValueError("explicit 140-dimensional causal and HG29 contracts required")
    return features.astype(np.float32), targets.astype(np.float32)
