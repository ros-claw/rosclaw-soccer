from types import SimpleNamespace

import mujoco
import numpy as np
import pytest
from test_shared_native_torque_layer import world

from rosclaw_soccer.rsi.receive_teacher_torque_replay import replay_saved_teacher_torques
from rosclaw_soccer.rsi.shared_native_receive_teacher import SharedNativeReceiveTeacher


def test_two_body_teacher_saved_torque_reintegration_and_tamper(monkeypatch):
    import rosclaw_soccer.rsi.shared_native_receive_teacher as module

    monkeypatch.setattr(
        module,
        "locomotion_contact_teacher_effect",
        lambda **kwargs: SimpleNamespace(torque_nm=np.full(29, 2.0), active=True),
    )
    w = world()
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    initial = np.empty(mujoco.mj_stateSize(w.model, spec))
    mujoco.mj_getState(w.model, w.data, initial, spec)
    targets = {p: np.zeros(29) for p in w.layouts}
    first = w.advance(targets)
    teacher = SharedNativeReceiveTeacher(w, "b/", horizon_controls=1)
    mujoco.mj_forward(w.model, w.data)
    second = w.advance(targets, torque_layers={"b/": teacher})
    history = {
        k: np.stack((first[k], second[k])) for k in ("qpos", "qvel", "torque", "ball_forces")
    }
    assert replay_saved_teacher_torques(w.model, initial, history, ball_body_name="ball") == 20
    history["qvel"][1, 3, 0] += 1
    with pytest.raises(ValueError, match="diverged"):
        replay_saved_teacher_torques(w.model, initial, history, ball_body_name="ball")
