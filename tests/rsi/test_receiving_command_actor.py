import numpy as np
import pytest
import torch
from test_continuous_velocity_stream import fake_policy

from rosclaw_soccer.providers.g1.velocity_policy import observation98, phase_delta
from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor, command_inputs


def actor():
    return ReceivingCommandActor(torch.zeros(112), torch.ones(112))


def test_original_velocity_bounds_and_all_six_tensors_have_gradients():
    torch.manual_seed(123)
    a = actor()
    inputs = torch.randn(23, 112)
    commands = a(inputs)
    assert commands.shape == (23, 3)
    assert (
        (commands >= torch.tensor([-0.5, -0.5, -1])) & (commands <= torch.tensor([1.0, 0.5, 1.0]))
    ).all()
    (commands - torch.randn_like(commands)).square().mean().backward()
    assert len(tuple(a.parameters())) == 6
    assert all(p.requires_grad and p.grad is not None and p.grad.norm() > 0 for p in a.parameters())
    assert not a.observation_mean.requires_grad and not a.observation_scale.requires_grad


@pytest.mark.parametrize("fault", ["shape", "nan", "dtype", "scale", "grad"])
def test_training_normalization_cannot_be_nonfinite_or_mutable_autograd(fault):
    mean, scale = torch.zeros(112), torch.ones(112)
    if fault == "shape":
        mean = mean[:98]
    elif fault == "nan":
        mean[0] = float("nan")
    elif fault == "dtype":
        mean = mean.double()
    elif fault == "scale":
        scale[0] = 0
    else:
        mean.requires_grad_(True)
    with pytest.raises(ValueError):
        ReceivingCommandActor(mean, scale)


def test_nonfinite_network_output_is_rejected_not_clipped_to_velocity_limit():
    a = actor()
    with torch.no_grad():
        next(a.parameters()).fill_(float("nan"))
    with pytest.raises(ValueError):
        a(torch.zeros(1, 112))


def test_inputs_recompute_command_and_gait_without_teacher_command_or_advancing_state():
    q, v = np.zeros(43), np.zeros(41)
    q[3] = 1
    q[36:39] = [1, 0, 0.11]
    previous = np.linspace(-0.01, 0.01, 29, dtype=np.float32)
    phase = phase_delta()
    params = fake_policy().params
    inputs, command, following, launch = command_inputs(
        q,
        v,
        np.zeros(3),
        params=params,
        last_action=previous,
        phase=phase,
        home_xy=np.zeros(2),
        pass_direction=np.array([1.0, 0.0]),
        anchor_world=np.array([0.0, 0.0, 0.11]),
        launch_seen=False,
        foot_contact_seen=False,
        other_body_contact_seen=False,
    )
    expected, ep = observation98(q, v, np.zeros(3), command, previous, phase, params)
    np.testing.assert_array_equal(inputs[:98], expected)
    np.testing.assert_array_equal(inputs[69:98], previous)
    assert inputs.shape == (112,) and inputs.dtype == np.float32
    assert ep == following and phase == phase_delta() and not launch
    assert np.all(inputs[9:11] == 0)  # stopped causal proposal gait mask
