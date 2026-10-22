import copy

import torch

from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import same_training_state
from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
from rosclaw_soccer.rsi.motor_critic_regression import regress_motor_critic


def test_critic_can_fit_returns_without_motor_weight_moment_or_counter_changes():
    torch.manual_seed(91)
    actor = [torch.nn.Parameter(torch.zeros(2)) for _ in range(16)]
    critic = MotorCritic()
    optimizer = torch.optim.Adam(
        [dict(params=actor, lr=0.00001), dict(params=critic.parameters(), lr=0.001)]
    )
    for p in actor + list(critic.parameters()):
        p.grad = torch.ones_like(p)
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    original = [p.detach().clone() for p in actor]
    slots = [copy.deepcopy(optimizer.state[p]) for p in actor]
    x = torch.randn(128, 288)
    targets = 2 * x[:, 0]
    counters = []
    h = regress_motor_critic(
        critic, optimizer, actor, x, targets, steps=32, on_update=counters.append
    )
    assert counters == list(range(1, 33))
    assert h[-1]["post_step_MSE"] < h[0]["pre_step_MSE"]
    assert all(torch.equal(p.detach(), v) for p, v in zip(actor, original, strict=True))
    assert all(
        same_training_state(optimizer.state[p], s) for p, s in zip(actor, slots, strict=True)
    )
    assert all(float(optimizer.state[p]["step"]) == 33 for p in critic.parameters())
