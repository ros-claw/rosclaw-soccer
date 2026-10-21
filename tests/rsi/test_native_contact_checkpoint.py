"""New native course weights cannot inherit old checkpoint admission."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from rosclaw_soccer.rsi import native_contact_checkpoint as module


@pytest.mark.parametrize(
    "state",
    [
        {"schema": "legacy", "stable_prior_rehearsal": True},
        {
            "schema": "soccer.rsi.native_contact24_ppo.v1",
            "protocol": "native-contact24-v1",
            "episode_count": 8,
        },
    ],
)
def test_checkpoint_reader_rejects_old_intent_or_posthoc_checkpoint(monkeypatch, state):
    monkeypatch.setattr(module, "file_hash", lambda p: "sha256:test")

    def load(*args, **kwargs):
        assert kwargs["weights_only"] is True
        return state

    monkeypatch.setattr(module.torch, "load", load)
    with pytest.raises(ValueError, match="fixed final20"):
        module.NativeContactSession(
            Path("actor.onnx"),
            Path("params.yaml"),
            expected_model_hash="sha256:onnx",
            expected_config_hash="sha256:yaml",
            checkpoint=Path("native.pt"),
            expected_checkpoint_hash="sha256:test",
            parent_checkpoint=Path("parent.pt"),
            expected_parent_hash="sha256:test",
            expected_native_model_hash="sha256:body",
            arm="final",
        )


def test_residual_reader_checks_foundation_intent_and_preserves_initial_function(
    monkeypatch, tmp_path
):
    import torch
    from torch import nn

    from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor

    class FakeBase(nn.Module):
        reference_features = 79

        def __init__(self):
            super().__init__()
            self.head = nn.Linear(177, 29)
            self.register_buffer("prior_mean", torch.zeros(1, 98))
            self.register_buffer("prior_denominator", torch.ones(1, 98))

        def forward(self, obs):
            return self.head(obs)

    monkeypatch.setattr(module.VelocityPriorActor, "from_onnx", lambda *a, **k: FakeBase())
    monkeypatch.setattr(
        module.VelocityPolicyParameters,
        "load",
        lambda *a, **k: SimpleNamespace(contract_hash="params"),
    )
    base = FakeBase()
    actor = NativeResidualContactActor(base)
    baseline = {"baseline." + k: v.clone() for k, v in actor.state_dict().items()}
    with torch.no_grad():
        actor.adapter[-1].bias.fill_(0.1)
    parent_path, checkpoint = tmp_path / "parent.pt", tmp_path / "residual.pt"
    torch.save(
        {"algorithm": {"actor_state_dict": {"mlp." + k: v for k, v in base.state_dict().items()}}},
        parent_path,
    )
    parent_hash = module.file_hash(parent_path)
    state = dict(
        schema=module.NativeResidualContactSession.checkpoint_schema,
        protocol="native-contact24-v1",
        actor_kind="bounded-residual",
        adapter_config=dict(
            near_radius_m=0.35,
            far_radius_m=0.7,
            maximum_normalized_residual=0.2,
            missing_ball9_disables_adapter=True,
            hidden_dims=[128, 64],
            input_dim=177,
            output_dim=29,
        ),
        adapter_proof_hash="sha256:proof",
        episode_count=20,
        value_warmup_episodes=4,
        ppo_episodes=16,
        actor_observation_dim=177,
        critic_observation_dim=177,
        native_pd_position_targets=29,
        parent_checkpoint_hash=parent_hash,
        original_onnx_hash="onnx",
        parameter_hash="yaml",
        native_model_hash="body",
        activation_ceiling="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        fresh_optimizer=True,
        counts=dict(
            ppo_steps=16, value_only_steps=64, optimizer_completed=80, optimizer_attempted=80
        ),
        actor_state_dict=actor.state_dict(),
        policy_state_dict=baseline,
    )

    def reader(kind=module.NativeResidualContactSession, arm="final"):
        torch.save(state, checkpoint)
        return kind(
            Path("actor.onnx"),
            Path("params.yaml"),
            expected_model_hash="onnx",
            expected_config_hash="yaml",
            checkpoint=checkpoint,
            expected_checkpoint_hash=module.file_hash(checkpoint),
            parent_checkpoint=parent_path,
            expected_parent_hash=parent_hash,
            expected_native_model_hash="body",
            arm=arm,
        )

    initial, final = reader(arm="initial"), reader()
    obs = torch.randn(5, 177)
    obs[:, 168:170] = torch.tensor([0.1, 0.1])
    assert torch.equal(initial.model(obs), base(obs))
    assert not torch.equal(final.model(obs), base(obs))
    obs[:, 168:170] = torch.tensor([0.9, 0.0])
    assert torch.equal(final.model(obs), base(obs))
    with pytest.raises(ValueError, match="fixed final20"):
        reader(module.NativeContactSession)
    state["actor_state_dict"]["base.head.bias"] = state["actor_state_dict"]["base.head.bias"] + 0.01
    with pytest.raises(ValueError, match="immutable foundation"):
        reader()
