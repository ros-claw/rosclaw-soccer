from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.velocity_prior_actor import LearnedPriorNativeSession, VelocityPriorActor


@pytest.fixture
def setup_checkpoint(tmp_path, monkeypatch):
    from rosclaw_soccer.rsi import velocity_prior_actor as mod

    model = VelocityPriorActor(tensors(), reference_features=79)
    state = {"mlp." + k: v for k, v in model.state_dict().items()}
    path = tmp_path / "foot-only.pt"
    torch.save(
        dict(
            foot_only_intent="foot-only-v1",
            actor_observation_dim=177,
            critic_observation_dim=304,
            iteration=16,
            force_decoder="addressed",
            fresh_optimizer=True,
            initial_actor_state_dict=state,
            algorithm={"actor_state_dict": state},
        ),
        path,
    )
    monkeypatch.setattr(mod, "file_hash", lambda p: "sha256:checkpoint")
    monkeypatch.setattr(
        mod.VelocityPolicyParameters,
        "load",
        lambda *a, **k: SimpleNamespace(contract_hash="sha256:params"),
    )
    monkeypatch.setattr(
        VelocityPriorActor,
        "from_onnx",
        lambda *a, **k: VelocityPriorActor(tensors(), reference_features=k["reference_features"]),
    )
    return dict(
        model=Path("mock.onnx"),
        config=Path("mock.yaml"),
        expected_model_hash="sha256:model",
        expected_config_hash="sha256:config",
        checkpoint=path,
        expected_checkpoint_hash="sha256:checkpoint",
        arm="initial",
    )


def test_explicit_foot_only_constructor_keeps_distinct_protocol(setup_checkpoint):
    obj = LearnedPriorNativeSession(
        **setup_checkpoint, reference_mode="declared79", checkpoint_intent="foot-only-v1"
    )
    assert obj.reference_features == 79
    assert obj.next_frame == 0 and obj.neural_calls == 0


def test_default_old_certificate_boundary_does_not_accept_new_intent(setup_checkpoint):
    with pytest.raises(ValueError, match="stable-prior"):
        LearnedPriorNativeSession(**setup_checkpoint, reference_mode="declared79")


@pytest.mark.parametrize("mode", ["zero79", "zero70", "declared70"])
def test_new_intent_requires_measured_ball_and_reference_mode(setup_checkpoint, mode):
    with pytest.raises(ValueError, match="foot-only"):
        LearnedPriorNativeSession(
            **setup_checkpoint, reference_mode=mode, checkpoint_intent="foot-only-v1"
        )
