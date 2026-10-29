from dataclasses import replace

import mujoco
import numpy as np
import pytest
import torch
from test_continuous_velocity_stream import fake_policy
from test_shared_native_motor_world import world
from test_velocity_prior_actor import tensors

from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
from rosclaw_soccer.rsi.adaptive_passer_velocity import AdaptivePasserVelocityRecovery
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def test_zero_adaptive_player_preserves_actual_clock_and_blend(monkeypatch):
    policy = fake_policy()
    policy.params = replace(policy.params, stiffness=(20.0,) * 29, damping=(1.0,) * 29)
    monkeypatch.setattr(
        "rosclaw_soccer.providers.g1.continuous_velocity_stream.G1VelocityPolicy",
        lambda *a, **k: policy,
    )
    stream = ContinuousVelocityStream(
        None, None, expected_model_hash="mock", expected_config_hash="mock", horizon_controls=400
    )
    actor = (
        PrecontactReceivingMotor(
            VelocityPriorActor(tensors(), reference_features=14),
            torch.as_tensor(stream.params.action_scale, dtype=torch.float32),
        )
        .eval()
        .requires_grad_(False)
    )
    w = world()
    player = AdaptivePasserVelocityRecovery(w, "a/", "b/", stream, actor)
    player.activate(motion_complete=True, passer_foot_seen=True)
    command = np.zeros(3, np.float32)
    for frame in range(21):
        mujoco.mj_forward(w.model, w.data)
        target = player.observe(command)
        w.advance({"a/": target, "b/": np.zeros(29)})
        row = player.commit(transition_target=target if frame < 20 else None)
        assert row["active"] and stream.next_frame == frame + 1
    np.testing.assert_array_equal(
        player.last_adaptive_trace["target"], player.last_adaptive_trace["original_target"]
    )
    np.testing.assert_array_equal(stream.last_action, player.last_adaptive_trace["action"])
    assert player.last_adaptive_trace["physical_execution_proven"] is False


def test_cannot_use_same_teammate_or_trainable_actor():
    w = world()
    with pytest.raises(ValueError):
        AdaptivePasserVelocityRecovery(w, "a/", "a/", object(), object())
