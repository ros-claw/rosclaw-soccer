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
