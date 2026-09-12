"""The training bit changes sample selection, never actor/critic inference."""

import pytest

from rosclaw_soccer.training.binary_learning_window import attach_binary_learning_window
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig

torch = pytest.importorskip("torch")


class Learner(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = torch.nn.Linear(3, 2)
        self.logstd = torch.nn.Parameter(torch.zeros(2))

    def forward(self, observation):
        action = self.layer(observation)
        return action, action.sum(1)


def test_outputs_parameters_rng_and_gradients_unchanged():
    learner = Learner()
    before = torch.get_rng_state().clone()
    wrapper = attach_binary_learning_window(learner, observation_size=3)
    assert torch.equal(before, torch.get_rng_state())
    assert wrapper.logstd is learner.logstd
    assert {id(p) for p in wrapper.parameters()} == {id(p) for p in learner.parameters()}
    obs = torch.randn(4, 3, requires_grad=True)
    expected = learner(obs)
    for bit in (0.0, 1.0):
        result = wrapper(torch.cat((obs, torch.full((4, 1), bit)), 1))
        for value, reference in zip(result, expected, strict=True):
            assert torch.equal(value, reference)
    wrapper(torch.cat((obs, torch.ones(4, 1)), 1))[0].sum().backward()
    assert torch.isfinite(obs.grad).all()
    assert all(name.startswith("learner.") for name in wrapper.state_dict())


@pytest.mark.parametrize("bit", [0.5, -1.0, 2.0, float("nan"), float("inf")])
def test_nonbinary_bit_rejected(bit):
    wrapper = attach_binary_learning_window(Learner(), observation_size=3)
    with pytest.raises(ValueError):
        wrapper(torch.tensor([[0.0, 0.0, 0.0, bit]]))


@pytest.mark.parametrize("size", [True, 0, -1, 4096, 3.0])
def test_invalid_contract_rejected(size):
    with pytest.raises(ValueError):
        attach_binary_learning_window(Learner(), observation_size=size)


def test_invalid_actor_and_inputs_rejected():
    with pytest.raises(ValueError):
        attach_binary_learning_window(torch.nn.Linear(3, 2), observation_size=3)
    wrapper = attach_binary_learning_window(Learner(), observation_size=3)
    for obs in (
        torch.zeros(1, 3),
        torch.zeros(0, 4),
        torch.zeros(1, 4, dtype=torch.int64),
        [[0, 0, 0, 1]],
    ):
        with pytest.raises(ValueError):
            wrapper(obs)


def test_explicit_extended_ppo_contract():
    config = FullBodyPPOUpdateConfig(
        observation_size=140, action_size=32, learning_observation_index=139
    )
    assert config.learning_observation_index == 139
    assert (
        FullBodyPPOUpdateConfig(observation_size=139, action_size=32).learning_observation_index
        is None
    )


def test_all_selected_update_matches_original_and_masked_likelihood_is_checked():
    from rosclaw_soccer.training.coupled_ball_residual import (
        build_coupled_ball_residual_actor_critic,
    )
    from rosclaw_soccer.training.full_body_ppo import update_full_body_ppo

    original = build_coupled_ball_residual_actor_critic()
    duplicate = build_coupled_ball_residual_actor_critic()
    duplicate.load_state_dict(original.state_dict())
    wrapper = attach_binary_learning_window(duplicate, observation_size=139)
    obs = torch.zeros(4, 2, 139)
    obs[:, :, 136] = 1
    obs[:, :, 138] = 0.6
    with torch.no_grad():
        mean, value = original(obs.reshape(-1, 139))
        normal = torch.distributions.Normal(mean, original.logstd.clamp(-2.5, -0.3).exp())
        raw = normal.sample()
        data = dict(
            obs=obs,
            raw=raw.reshape(4, 2, 32),
            logp=normal.log_prob(raw).sum(1).reshape(4, 2),
            value=value.reshape(4, 2),
            reward=torch.randn(4, 2),
            alive=torch.ones(4, 2),
            next_alive=torch.ones(4, 2),
        )
    data["next_alive"][-1] = 0
    extended = {key: value.clone() for key, value in data.items()}
    extended["obs"] = torch.cat((obs, torch.ones(4, 2, 1)), 2)
    one = torch.optim.Adam(original.parameters(), lr=5e-5)
    two = torch.optim.Adam(wrapper.parameters(), lr=5e-5)
    before = torch.get_rng_state().clone()
    result = update_full_body_ppo(
        original, one, data, FullBodyPPOUpdateConfig(observation_size=139, action_size=32, epochs=1)
    )
    torch.set_rng_state(before)
    masked_result = update_full_body_ppo(
        wrapper,
        two,
        extended,
        FullBodyPPOUpdateConfig(
            observation_size=140, action_size=32, epochs=1, learning_observation_index=139
        ),
    )
    assert result["optimizer_steps"] == masked_result["optimizer_steps"]
    for key, value in original.state_dict().items():
        assert torch.equal(value, duplicate.state_dict()[key])
    # A masked sample's starting likelihood remains authenticated, not ignored.
    fresh = build_coupled_ball_residual_actor_critic()
    fresh_wrapper = attach_binary_learning_window(fresh, observation_size=139)
    with torch.no_grad():
        mean, value = fresh(obs.reshape(-1, 139))
        normal = torch.distributions.Normal(mean, fresh.logstd.clamp(-2.5, -0.3).exp())
        extended["logp"] = normal.log_prob(raw).sum(1).reshape(4, 2)
        extended["value"] = value.reshape(4, 2)
    extended["obs"][0, :, -1] = 0
    extended["logp"][0, 0] += 0.1
    with pytest.raises(ValueError, match="likelihood"):
        update_full_body_ppo(
            fresh_wrapper,
            torch.optim.Adam(fresh_wrapper.parameters()),
            extended,
            FullBodyPPOUpdateConfig(
                observation_size=140, action_size=32, learning_observation_index=139
            ),
        )
