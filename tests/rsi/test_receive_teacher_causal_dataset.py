from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.rsi.receive_teacher_causal_dataset import FEATURE_DIM, causal_teacher_samples


def test_features_use_pre_step_state_and_only_preceding_contact():
    receiver = SimpleNamespace(
        qi=np.arange(7, 36), vi=np.arange(6, 35), ai=np.arange(29), rq=0, rv=0, bq=36, bv=35
    )
    teammate = SimpleNamespace(rq=43)
    initial_q = np.zeros(50)
    initial_v = np.zeros(41)
    initial_tau = np.zeros(58)
    q = np.zeros((1, 10, 50))
    v = np.zeros((1, 10, 41))
    torque = np.zeros((1, 10, 58))
    parent = np.zeros_like(torque)
    forces = np.zeros((1, 10, 2, 3))
    q[0, 0, 7] = 2
    torque[0, 0, 0] = 3
    forces[0, 0, 1, 0] = 2
    x, y = causal_teacher_samples(
        initial_qpos=initial_q,
        initial_qvel=initial_v,
        initial_actuator_torque=initial_tau,
        qpos=q,
        qvel=v,
        actual_torque=torque,
        parent_torque=parent,
        ball_forces=forces,
        receiver_layout=receiver,
        teammate_layout=teammate,
    )
    assert x.shape == (10, FEATURE_DIM) and y.shape == (10, 29)
    assert x[0, 0] == 0 and x[1, 0] == 2
    assert y[0, 0] == 3 and y[1, 0] == 0
    assert x[0, -5:].tolist() == [0, 0, 0, 0, 1]
    assert x[1, -5:-1].tolist() == [1, 0, 0, 1]
    assert x[1, -1] == np.float32(1 / 150)
