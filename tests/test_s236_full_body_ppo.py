import copy

import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.full_body_ppo import FullBodyPPOUpdateConfig, update_full_body_ppo


def case():
    torch = pytest.importorskip("torch")
    torch.manual_seed(236)
    actor = build_ball_residual_actor_critic()
    optimizer = torch.optim.Adam(actor.parameters(), lr=1e-4)
    obs = torch.randn((4, 2, 133)) * 0.1
    with torch.no_grad():
        mean, value = actor(obs.reshape(-1, 133))
        normal = torch.distributions.Normal(mean, actor.logstd.clamp(-2.5, -0.3).exp())
        raw = normal.sample()
        data = dict(
            obs=obs,
            raw=raw.reshape(4, 2, 29),
            value=value.reshape(4, 2),
            logp=normal.log_prob(raw).sum(1).reshape(4, 2),
            reward=torch.randn((4, 2)),
            alive=torch.ones((4, 2)),
            next_alive=torch.ones((4, 2)),
        )
    return torch, actor, optimizer, data


def test_optimizer_replays_exactly_from_same_weights_data_and_rng():
    torch, actor, optimizer, data = case()
    initial, original = copy.deepcopy(actor.state_dict()), copy.deepcopy(data)
    rng = torch.get_rng_state()
    config = FullBodyPPOUpdateConfig(epochs=2, minibatch_size=4)
    result = update_full_body_ppo(actor, optimizer, data, config)
    updated = copy.deepcopy(actor.state_dict())
    replay = build_ball_residual_actor_critic()
    replay.load_state_dict(initial)
    repeated_optimizer = torch.optim.Adam(replay.parameters(), lr=1e-4)
    torch.set_rng_state(rng)
    assert update_full_body_ppo(replay, repeated_optimizer, data, config) == result
    assert result["optimizer_steps"] == 4 and result["on_policy_inputs_verified"]
    assert not result["promotion_eligible"]
    assert any(not torch.equal(initial[k], updated[k]) for k in initial)
    assert all(torch.equal(updated[k], replay.state_dict()[k]) for k in updated)
    assert all(torch.equal(data[k], original[k]) for k in data)


@pytest.mark.parametrize("field", ["logp", "value", "reward", "next_alive"])
def test_invalid_evidence_is_rejected_before_weights_change(field):
    torch, actor, optimizer, data = case()
    before = copy.deepcopy(actor.state_dict())
    if field in {"logp", "value"}:
        data[field] += 0.01
    elif field == "reward":
        data[field][0, 0] = float("nan")
    else:
        data[field][0, 0] = 0
    with pytest.raises(ValueError):
        update_full_body_ppo(actor, optimizer, data)
    assert all(torch.equal(before[k], actor.state_dict()[k]) for k in before)


def test_optimizer_cannot_include_another_learner():
    torch, actor, _, data = case()
    other = torch.nn.Parameter(torch.zeros(1))
    optimizer = torch.optim.Adam([*actor.parameters(), other])
    with pytest.raises(ValueError, match="exclusively"):
        update_full_body_ppo(actor, optimizer, data)


@pytest.mark.parametrize("field", ["obs", "raw", "value", "logp", "reward"])
@pytest.mark.parametrize("kind", ["bool", "integer", "list"])
def test_nonfloating_learning_evidence_rejected_before_inference_or_optimizer(field, kind):
    torch, actor, optimizer, data = case()
    before = copy.deepcopy(actor.state_dict())
    rng = torch.get_rng_state().clone()
    data[field] = (
        data[field].tolist()
        if kind == "list"
        else data[field].to(torch.bool if kind == "bool" else torch.int64)
    )
    with pytest.raises(ValueError, match="rollout tensors"):
        update_full_body_ppo(actor, optimizer, data)
    assert all(torch.equal(before[k], actor.state_dict()[k]) for k in before)
    assert not optimizer.state
    assert torch.equal(rng, torch.get_rng_state())


@pytest.mark.parametrize(
    "kwargs", [{"epochs": 0}, {"epochs": True}, {"gamma": 1}, {"target_kl": float("nan")}]
)
def test_update_budget_is_bounded(kwargs):
    with pytest.raises(ValueError):
        FullBodyPPOUpdateConfig(**kwargs)
