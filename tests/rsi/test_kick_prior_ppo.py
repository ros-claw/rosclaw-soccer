import math

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from test_kick_prior_actor import tensors  # noqa: E402

from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp  # noqa: E402
from rosclaw_soccer.rsi.kick_prior_actor import KickPriorActor  # noqa: E402
from rosclaw_soccer.rsi.kick_prior_ppo import (  # noqa: E402
    KickValueHead,
    freeze_kick_features,
    own_ppo_objective,
    value_features,
)


def fixture():
    torch.set_num_threads(1)
    actor = KickPriorActor(tensors())
    freeze_kick_features(actor)
    x = torch.from_numpy(np.random.default_rng(1).normal(size=(5, 547)).astype(np.float32))
    old = actor(x).detach()
    actions = old + torch.linspace(-0.01, 0.01, 145).reshape(5, 29)
    std = torch.full((29,), math.log(0.015))
    e = dict(
        observations=x,
        old_means=old,
        raw_actions=actions,
        old_logp=conditional_logp(old, actions, std, rho=0.9),
        advantages=torch.linspace(-1.0, 1.0, 5),
        actual_pd_commitment_verified=True,
        external_teacher=False,
    )
    return actor, x, old, std, e


def test_only_motor_readout_gets_policy_gradients():
    actor, x, old, std, episode = fixture()
    loss, stats = own_ppo_objective(actor, [episode], x, old, std)
    loss.backward()
    assert float(stats["mean_kl"].detach()) == pytest.approx(0.0, abs=1e-10)
    assert all(p.grad is None for n, p in actor.named_parameters() if not n.startswith("actor.6."))
    assert all(
        p.grad is not None and bool(torch.isfinite(p.grad).all())
        for n, p in actor.named_parameters()
        if n.startswith("actor.6.")
    )


def test_critic_has_no_actor_gradient_and_actual_clock():
    actor, x, _, _, _ = fixture()
    critic = KickValueHead()
    features = value_features(actor, x, torch.arange(295, 300))
    assert features.shape == (5, 130) and not features.requires_grad
    assert features[-1, -1] == 0
    critic(features).sum().backward()
    assert all(p.grad is None for p in actor.parameters())
    assert critic.net[-1].bias.grad is not None
    with pytest.raises(ValueError):
        critic(features.requires_grad_())


@pytest.mark.parametrize("defect", ["teacher", "not_executed", "wrong_logp", "nonfinite"])
def test_invalid_own_experience_refused(defect):
    actor, x, old, std, episode = fixture()
    if defect == "teacher":
        episode["external_teacher"] = True
    elif defect == "not_executed":
        episode["actual_pd_commitment_verified"] = False
    elif defect == "wrong_logp":
        episode["old_logp"] = episode["old_logp"] + 1
    else:
        episode["advantages"][0] = float("nan")
    with pytest.raises(ValueError):
        own_ppo_objective(actor, [episode], x, old, std)
