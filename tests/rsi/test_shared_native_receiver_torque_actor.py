from pathlib import Path

import numpy as np
import pytest
import torch
from test_shared_native_torque_layer import world

from rosclaw_soccer.rsi.shared_native_receiver_torque_actor import SharedNativeReceiverTorqueActor
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_train_receive_teacher_torque_bc import ResidualActor


def checkpoint(path: Path) -> str:
    actor = ResidualActor()
    for p in actor.parameters():
        torch.nn.init.zeros_(p)
    torch.save(
        dict(
            actor=actor.state_dict(),
            observation_mean=torch.zeros(140),
            observation_std=torch.ones(140),
            dataset_hash="test-data",
            architecture="tanh-256-256-tanh20",
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        path,
    )
    return digest(path)


def test_zero_actor_is_actual_parent_and_consecutive_feedback(tmp_path):
    w, reference = world(), world()
    path = tmp_path / "actor.pt"
    h = checkpoint(path)
    actor = SharedNativeReceiverTorqueActor(
        w, "b/", checkpoint=path, checkpoint_hash=h, dataset_hash="test-data", horizon_controls=1
    )
    targets = {p: np.zeros(29) for p in w.layouts}
    base = reference.advance(targets)
    result = w.advance(targets, torque_layers={"b/": actor})
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(result[key], base[key])
    assert actor.commands == actor.applied == 10
    assert actor.active_commands == 0 and actor.last_observation.shape == (140,)


def test_reject_wrong_checkpoint_and_unobserved_feedback(tmp_path):
    w = world()
    path = tmp_path / "actor.pt"
    h = checkpoint(path)
    with pytest.raises(ValueError, match="lineage"):
        SharedNativeReceiverTorqueActor(
            w, "b/", checkpoint=path, checkpoint_hash=h, dataset_hash="wrong", horizon_controls=1
        )
    actor = SharedNativeReceiverTorqueActor(
        w, "b/", checkpoint=path, checkpoint_hash=h, dataset_hash="test-data", horizon_controls=1
    )
    with pytest.raises(ValueError, match="actual applied"):
        actor.note_applied(np.zeros(29))
    assert actor.failed


def test_nonzero_actor_actually_controls_only_its_own_torques(tmp_path):
    w = world()
    path = tmp_path / "actor.pt"
    checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["actor"]["net.4.bias"][:] = 0.2
    torch.save(state, path)
    actor = SharedNativeReceiverTorqueActor(
        w,
        "b/",
        checkpoint=path,
        checkpoint_hash=digest(path),
        dataset_hash="test-data",
        horizon_controls=1,
    )
    result = w.advance({p: np.zeros(29) for p in w.layouts}, torque_layers={"b/": actor})
    assert actor.commands == actor.applied == actor.active_commands == 10
    assert np.all(np.abs(result["torque"][:, w.layouts["b/"].ai]) <= w.parameters["b/"].effort)
    assert np.any(result["applied_torque_change"][:, w.layouts["b/"].ai])
    assert not np.any(result["applied_torque_change"][:, w.layouts["a/"].ai])


def test_near_ball_gate_far_field_preserves_actual_parent_physics(tmp_path):
    w, reference = world(), world()
    path = tmp_path / "actor.pt"
    checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["actor"]["net.4.bias"][:] = 0.2
    torch.save(state, path)
    actor = SharedNativeReceiverTorqueActor(
        w,
        "b/",
        checkpoint=path,
        checkpoint_hash=digest(path),
        dataset_hash="test-data",
        horizon_controls=1,
        near_ball_gate=True,
    )
    targets = {p: np.zeros(29) for p in w.layouts}
    base = reference.advance(targets)
    result = w.advance(targets, torque_layers={"b/": actor})
    for key in ("qpos", "qvel", "torque", "ball_forces"):
        np.testing.assert_array_equal(result[key], base[key])
    assert actor.gate_closed_commands == actor.commands == actor.applied == 10
    assert actor.active_commands == 0


def test_near_ball_gate_allows_own_model_near_foot(tmp_path):
    import mujoco

    from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld

    old = world()
    data = mujoco.MjData(old.model)
    data.qpos[:] = old.data.qpos
    mujoco.mj_forward(old.model, data)
    foot = old.model.body("b/left_ankle_roll_link").id
    data.qpos[old.layouts["b/"].bq : old.layouts["b/"].bq + 3] = data.xpos[foot] + [0.1, 0, 0]
    mujoco.mj_forward(old.model, data)
    w = SharedNativeMotorWorld(old.model, data, players=old.parameters, ball_body_name="ball")
    path = tmp_path / "actor.pt"
    checkpoint(path)
    state = torch.load(path, map_location="cpu", weights_only=True)
    state["actor"]["net.4.bias"][:] = 0.2
    torch.save(state, path)
    actor = SharedNativeReceiverTorqueActor(
        w,
        "b/",
        checkpoint=path,
        checkpoint_hash=digest(path),
        dataset_hash="test-data",
        horizon_controls=1,
        near_ball_gate=True,
    )
    result = w.advance({p: np.zeros(29) for p in w.layouts}, torque_layers={"b/": actor})
    assert actor.active_commands > 0
    assert np.any(result["applied_torque_change"][:, w.layouts["b/"].ai])
    assert not np.any(result["applied_torque_change"][:, w.layouts["a/"].ai])
