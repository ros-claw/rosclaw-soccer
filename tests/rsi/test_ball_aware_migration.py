import pytest

torch = pytest.importorskip("torch")
from test_velocity_prior_actor import tensors  # noqa: E402

from rosclaw_soccer.rsi.ball_aware_migration import expand_ball_aware_state  # noqa: E402
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor  # noqa: E402


def test_migration_preserves_original_actor_and_enables_new_gradients():
    parent = VelocityPriorActor(tensors())
    # Nonzero reference weights represent an actually learned parent.
    with torch.no_grad():
        parent.mlp[0].weight[:, 98:] = 0.001
    state = {"mlp." + k: v for k, v in parent.state_dict().items()}
    expanded = expand_ball_aware_state(state, part="actor")
    actor = VelocityPriorActor(tensors(), reference_features=79)
    actor.load_state_dict({k: expanded["mlp." + k] for k in actor.state_dict()})
    obs = torch.randn(16, 168)
    wide = torch.cat((obs, torch.randn(16, 9)), dim=1)
    torch.testing.assert_close(actor(wide), parent(obs), atol=1e-6, rtol=0)
    assert expanded["mlp.mlp.0.weight"][:, 168:].count_nonzero() == 0
    assert parent.mlp[0].weight.shape == (512, 168)
    actor(wide).square().sum().backward()
    assert actor.mlp[0].weight.grad[:, 168:].abs().max() > 0


def test_critic_extension_preserves_old_normalization_and_zero_new_influence():
    state = {"mlp.0.weight": torch.randn(512, 295)}
    for name, value in (("_mean", 2.0), ("_var", 3.0), ("_std", 4.0)):
        state["obs_normalizer." + name] = torch.full((1, 295), value)
    state["obs_normalizer.count"] = torch.tensor(100.0)
    expanded = expand_ball_aware_state(state, part="critic")
    assert expanded["mlp.0.weight"].shape == (512, 304)
    torch.testing.assert_close(expanded["mlp.0.weight"][:, :295], state["mlp.0.weight"])
    assert expanded["mlp.0.weight"][:, 295:].count_nonzero() == 0
    assert expanded["obs_normalizer._mean"][0, -1] == 0
    assert expanded["obs_normalizer._var"][0, -1] == 1
    assert expanded["obs_normalizer.count"] == 100
    with pytest.raises(ValueError):
        expand_ball_aware_state(state, part="actor")
