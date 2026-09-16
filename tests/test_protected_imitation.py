import numpy as np
import pytest

from rosclaw_soccer.training.event_protected_actor import build_event_protected_actor_critic
from rosclaw_soccer.training.protected_imitation import fit_protected_demonstration

torch = pytest.importorskip("torch")


class Seed(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.actor = torch.nn.Sequential(
            torch.nn.Linear(3, 8), torch.nn.Tanh(), torch.nn.Linear(8, 2)
        )
        self.critic = torch.nn.Linear(3, 1)
        self.logstd = torch.nn.Parameter(torch.zeros(2))

    def forward(self, x):
        return self.actor(x), self.critic(x).squeeze(-1)


def fixture():
    torch.manual_seed(41)
    seed = build_event_protected_actor_critic(Seed(), observation_size=3, action_size=2)
    obs = np.zeros((16, 4), np.float32)
    obs[:, 0] = np.linspace(-1, 1, len(obs))
    obs[:, -1] = 1
    with torch.no_grad():
        target = seed(torch.from_numpy(obs))[0].numpy().copy() + np.float32(0.1)
    return seed, obs, target


def test_fit_preserves_inputs_prefix_critic_scale_and_replays_exactly():
    seed, obs, target = fixture()
    before = {k: v.clone() for k, v in seed.state_dict().items()}
    flags = {k: p.requires_grad for k, p in seed.named_parameters()}
    arrays = obs.copy(), target.copy()
    first, stats = fit_protected_demonstration(seed, obs, target, steps=32)
    second, again = fit_protected_demonstration(seed, obs, target, steps=32)
    assert stats == again
    assert stats["final_loss"] < stats["initial_loss"]
    assert stats["on_policy"] is False and stats["promotion_eligible"] is False
    assert all(torch.equal(v, second.state_dict()[k]) for k, v in first.state_dict().items())
    assert all(torch.equal(v, seed.state_dict()[k]) for k, v in before.items())
    assert flags == {k: p.requires_grad for k, p in seed.named_parameters()}
    assert flags == {k: p.requires_grad for k, p in first.named_parameters()}
    assert np.array_equal(obs, arrays[0]) and np.array_equal(target, arrays[1])
    assert all(
        torch.equal(v, first.state_dict()[k])
        for k, v in before.items()
        if not k.startswith("plastic.actor.")
    )
    prior = torch.from_numpy(obs.copy())
    prior[:, -1] = 0
    with torch.no_grad():
        assert torch.equal(first(prior)[0], seed(prior)[0])


@pytest.mark.parametrize("fault", ["nan", "inf", "float64", "empty", "ignored", "shape", "huge"])
def test_reject_invalid_or_unexecuted_demonstrations(fault):
    seed, obs, target = fixture()
    if fault == "nan":
        obs[0, 0] = np.nan
    elif fault == "inf":
        target[0, 0] = np.inf
    elif fault == "float64":
        obs = obs.astype(np.float64)
    elif fault == "empty":
        obs, target = obs[:0], target[:0]
    elif fault == "ignored":
        obs[0, -1] = 0
    elif fault == "shape":
        target = target[:, :1]
    else:
        target[0, 0] = 21
    with pytest.raises(ValueError):
        fit_protected_demonstration(seed, obs, target, steps=1)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": True},
        {"steps": 0},
        {"steps": 1025},
        {"learning_rate": float("nan")},
        {"learning_rate": 0.1},
        {"retention_coefficient": -1},
        {"retention_coefficient": True},
    ],
)
def test_fit_budget_validation(kwargs):
    with pytest.raises(ValueError):
        fit_protected_demonstration(*fixture(), **kwargs)


@pytest.mark.parametrize("shared_storage", [False, True])
def test_reject_actor_anchor_parameter_aliases(shared_storage):
    seed, obs, target = fixture()
    if shared_storage:
        seed.plastic.actor[0].weight = torch.nn.Parameter(seed.anchor.actor[0].weight.detach())
    else:
        seed.plastic.actor = seed.anchor.actor
    with pytest.raises(ValueError, match="share"):
        fit_protected_demonstration(seed, obs, target, steps=1)


def test_bad_model_weights_fail_before_fit():
    seed, obs, target = fixture()
    with torch.no_grad():
        seed.plastic.actor[0].weight[0, 0] = float("nan")
    with pytest.raises(ValueError):
        fit_protected_demonstration(seed, obs, target, steps=1)
