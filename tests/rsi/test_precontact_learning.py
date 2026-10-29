import numpy as np
import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.precontact_feedback import precontact_feedback
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor, precontact_gate
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def test_gate_is_causal_smooth_and_zero_when_ball_far_or_receding():
    x = torch.zeros(3, 112)
    x[:, 98] = torch.tensor([1.0, 1.6, 1.0])
    x[:, 101] = torch.tensor([-2.0, -2.0, 2.0])
    assert precontact_gate(x).tolist() == [1.0, 0.0, 0.0]
    x[0, 98] = 1.4
    assert precontact_gate(x)[0].item() == pytest.approx(0.5, abs=1e-6)
    x[0, 101] = float("nan")
    with pytest.raises(ValueError):
        precontact_gate(x)


def test_feedback_observes_order_and_penalizes_same_step_nonfoot():
    force = np.zeros((4, 10, 3))
    force[1, 2, 0] = 11
    force[2, 0, 2] = 2
    reward, event = precontact_feedback(force)
    assert reward.tolist() == [0, 0.5, -1, 0]
    assert event["first_foot_substep"] == 12
    force[1, 2, 2] = 2
    reward, _ = precontact_feedback(force)
    assert reward.tolist() == [0, -1, 0, 0]


def test_trainable_motor_preserves_reference_outside_gate():
    actor = PrecontactReceivingMotor(
        VelocityPriorActor(tensors(), reference_features=14), torch.full((29,), 0.25)
    )
    x = torch.zeros(2, 112)
    x[:, 98] = torch.tensor([1.0, 1.6])
    x[:, 101] = -2.0
    reference = torch.zeros(2, 29)
    with torch.no_grad():
        actor.plastic.mlp[6].bias.add_(0.1)
    out, evidence = actor.forward_with_evidence(x, reference)
    assert not torch.equal(out[0], reference[0])
    assert torch.equal(out[1], reference[1])
    assert evidence["realized_delta_radians"][0].abs().max() <= 0.2
    out[0].sum().backward()
    assert actor.plastic.mlp[6].bias.grad.abs().sum() > 0
    assert all(p.grad is None for p in actor.baseline.parameters())
