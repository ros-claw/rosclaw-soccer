import copy

import pytest

from rosclaw_soccer.training.ball_residual import build_ball_residual_actor_critic
from rosclaw_soccer.training.success_distillation import (
    SuccessDistillationConfig,
    distill_successful_motor_actions,
)

torch = pytest.importorskip("torch")


def data():
    generator = torch.Generator().manual_seed(480)
    return dict(
        observations=torch.randn(8, 133, generator=generator) * 0.1,
        raw_actions=torch.randn(8, 29, generator=generator) * 0.1,
        sample_weights=torch.full((8,), 0.125),
    )


def test_supervised_math_exact_and_protected_weights_unchanged():
    agent = build_ball_residual_actor_critic()
    reference = copy.deepcopy(agent)
    before = {k: t.clone() for k, t in agent.state_dict().items()}
    flags = [p.requires_grad for p in agent.parameters()]
    tensors = data()
    config = SuccessDistillationConfig(steps=4)
    result = distill_successful_motor_actions(agent, **tensors, config=config)
    optimizer = torch.optim.Adam(reference.actor.parameters(), lr=0.001)
    for _ in range(4):
        mean, _ = reference(tensors["observations"])
        loss = ((mean - tensors["raw_actions"]).square().mean(1) * tensors["sample_weights"]).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.actor.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    assert all(torch.equal(v, reference.state_dict()[k]) for k, v in agent.state_dict().items())
    assert all(
        torch.equal(v, before[k])
        for k, v in agent.state_dict().items()
        if not k.startswith("actor.")
    )
    assert [p.requires_grad for p in agent.parameters()] == flags
    assert result["optimizer_steps"] == 4
    assert result["final_weighted_mse"] < result["initial_weighted_mse"]
    assert result["promotion_eligible"] is False


@pytest.mark.parametrize("bad", ["nan", "shape", "grad", "dtype", "weight", "bound", "list"])
def test_invalid_data_rejected_before_any_update(bad):
    agent = build_ball_residual_actor_critic()
    before = {k: v.clone() for k, v in agent.state_dict().items()}
    tensors = data()
    if bad == "nan":
        tensors["raw_actions"][0, 0] = float("nan")
    elif bad == "shape":
        tensors["observations"] = torch.zeros(8, 136)
    elif bad == "grad":
        tensors["raw_actions"].requires_grad_(True)
    elif bad == "dtype":
        tensors["observations"] = tensors["observations"].double()
    elif bad == "weight":
        tensors["sample_weights"][0] = -1
    elif bad == "bound":
        tensors["raw_actions"][0, 0] = 21
    else:
        tensors["observations"] = []
    with pytest.raises(ValueError):
        distill_successful_motor_actions(agent, **tensors)
    assert all(torch.equal(v, before[k]) for k, v in agent.state_dict().items())


def test_protected_parameter_alias_is_rejected():
    agent = build_ball_residual_actor_critic()
    agent.critic[0].weight = agent.actor[0].weight
    with pytest.raises(ValueError, match="share parameters"):
        distill_successful_motor_actions(agent, **data())


@pytest.mark.parametrize("steps", [True, 0, 10001, 1.5])
def test_invalid_step_budget(steps):
    with pytest.raises(ValueError):
        SuccessDistillationConfig(steps=steps)


@pytest.mark.parametrize("shape", [(139, 29), (133, 32), (136, 32), (True, 29), (139.0, 32)])
def test_explicit_contract_pairs_only(shape):
    with pytest.raises(ValueError):
        SuccessDistillationConfig(observation_size=shape[0], action_size=shape[1])


def test_coupled_contract_opt_in_and_exact_actor_only_update():
    from rosclaw_soccer.training.coupled_ball_residual import (
        build_coupled_ball_residual_actor_critic,
    )

    agent = build_coupled_ball_residual_actor_critic()
    reference = copy.deepcopy(agent)
    before = {key: tensor.clone() for key, tensor in agent.state_dict().items()}
    observations = torch.zeros(8, 139)
    observations[:, 136] = 1
    observations[:, 138] = 0.6
    actions = torch.full((8, 32), 0.1)
    weights = torch.full((8,), 0.125)
    tensors = dict(observations=observations, raw_actions=actions, sample_weights=weights)
    with pytest.raises(ValueError):
        distill_successful_motor_actions(agent, **tensors)
    assert all(torch.equal(tensor, before[key]) for key, tensor in agent.state_dict().items())
    config = SuccessDistillationConfig(
        steps=4, learning_rate=0.0001, observation_size=139, action_size=32
    )
    result = distill_successful_motor_actions(agent, **tensors, config=config)
    optimizer = torch.optim.Adam(reference.actor.parameters(), lr=config.learning_rate)
    for _ in range(config.steps):
        loss = ((reference.actor(observations) - actions).square().mean(1) * weights).sum()
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(reference.actor.parameters(), 1, error_if_nonfinite=True)
        optimizer.step()
    assert result["optimizer_steps"] == 4
    assert result["final_weighted_mse"] < result["initial_weighted_mse"]
    assert all(
        torch.equal(tensor, reference.state_dict()[key])
        for key, tensor in agent.state_dict().items()
    )
    assert all(
        torch.equal(tensor, before[key])
        for key, tensor in agent.state_dict().items()
        if not key.startswith("actor.")
    )
