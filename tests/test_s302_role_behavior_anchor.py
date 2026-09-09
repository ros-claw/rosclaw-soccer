from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training.role_behavior_anchor import RoleBehaviorAnchor


def anchor():
    policy = NearBallResidualPolicy.initialize(
        tuple(f"agent.{i}" for i in range(8)), "sha256:" + "a" * 64
    )
    observations = np.random.default_rng(302).normal(size=(40, 8, 56))
    active = np.ones((40, 8), dtype=bool)
    return RoleBehaviorAnchor(policy, observations, active, "sha256:" + "b" * 64)


def test_anchor_is_immutable_and_all_inputs_are_content_bound():
    value = anchor()
    digest = value.anchor_hash
    with pytest.raises(ValueError):
        value.observations.setflags(write=True)
    with pytest.raises(ValueError):
        value.active.setflags(write=True)
    assert replace(value, coefficient=200).anchor_hash != digest
    assert replace(value, source_hash="sha256:" + "c" * 64).anchor_hash != digest
    changed = value.observations.copy()
    changed[0, 0, 0] += 0.01
    assert replace(value, observations=changed).anchor_hash != digest


@pytest.mark.parametrize("coefficient", [0, True, -1, float("nan"), 1001])
def test_anchor_rejects_invalid_coefficient(coefficient):
    with pytest.raises(ValueError):
        replace(anchor(), coefficient=coefficient)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 1001, np.iinfo(np.int64).min])
def test_anchor_rejects_nonfinite_or_out_of_range_states_including_integer_overflow(bad):
    value = anchor()
    with pytest.raises(ValueError):
        replace(value, observations=np.full(value.observations.shape, bad))


def test_anchor_requires_explicit_nonempty_role_samples():
    value = anchor()
    for active in (np.zeros((40, 8), dtype=bool), np.ones((40, 8)), np.ones((40, 7), dtype=bool)):
        with pytest.raises(ValueError):
            replace(value, active=active)


def test_anchor_kl_matches_gaussian_distribution_and_has_no_reference_gradient():
    torch = pytest.importorskip("torch")
    value = anchor()
    parameters = {
        k: torch.tensor(v[0].copy(), requires_grad=True) for k, v in value.policy.weights.items()
    }
    assert float(value.kl_loss(parameters, 0).detach()) == pytest.approx(0, abs=1e-12)
    parameters["b2"] = torch.full((12,), 0.02, dtype=torch.float64, requires_grad=True)
    actual = value.kl_loss(parameters, 0)
    wanted = torch.distributions.kl_divergence(
        torch.distributions.Normal(
            torch.zeros(12, dtype=torch.float64), parameters["log_std"].exp()
        ),
        torch.distributions.Normal(parameters["b2"], parameters["log_std"].exp()),
    ).sum()
    assert float(actual.detach()) == pytest.approx(float(wanted.detach()), abs=1e-12)
    actual.backward()
    assert torch.all(parameters["b2"].grad > 0)
    no_samples = value.active.copy()
    no_samples[:, 0] = False
    assert float(replace(value, active=no_samples).kl_loss(parameters, 0).detach()) == 0
