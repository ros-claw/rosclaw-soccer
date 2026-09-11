import pytest

from rosclaw_soccer.training.redirect_reward import rolling_redirect_contact_reward


def case():
    torch = pytest.importorskip("torch")
    return torch, dict(
        velocity_before=torch.tensor([[0.8, 0.0, 0.0]]),
        velocity_after=torch.tensor([[-0.8, 0.0, 0.0]]),
        target_delta_xy=torch.tensor([[-2.0, 0.0]]),
        foot_contact=torch.tensor([True]),
    )


def test_reward_distinguishes_redirect_from_equal_speed_motion():
    _, args = case()
    assert rolling_redirect_contact_reward(**args).item() == pytest.approx(0.4)
    args["target_delta_xy"].neg_()
    assert rolling_redirect_contact_reward(**args).item() == pytest.approx(-0.4)


def test_no_contact_no_credit_and_upward_change_penalized():
    _, args = case()
    args["foot_contact"].fill_(False)
    assert rolling_redirect_contact_reward(**args).item() == 0
    args["foot_contact"].fill_(True)
    args["velocity_after"][0, 2] = 0.5
    assert rolling_redirect_contact_reward(**args).item() == 0


def test_hint_is_rotation_invariant_and_stationary_velocity_has_no_credit():
    torch, args = case()
    expected = rolling_redirect_contact_reward(**args)
    for key in ("velocity_before", "velocity_after", "target_delta_xy"):
        xy = args[key][:, :2].clone()
        args[key][:, 0], args[key][:, 1] = -xy[:, 1], xy[:, 0]
    assert torch.equal(rolling_redirect_contact_reward(**args), expected)
    args["velocity_after"] = args["velocity_before"].clone()
    assert rolling_redirect_contact_reward(**args).item() == 0


@pytest.mark.parametrize("fault", ["nan", "zero_target", "gradient", "shape", "dtype", "mask"])
def test_bad_inputs_fail_closed(fault):
    torch, args = case()
    if fault == "nan":
        args["velocity_before"][0, 0] = float("nan")
    elif fault == "zero_target":
        args["target_delta_xy"].zero_()
    elif fault == "gradient":
        args["velocity_after"].requires_grad_()
    elif fault == "shape":
        args["target_delta_xy"] = torch.zeros(2, 2)
    elif fault == "dtype":
        args["velocity_after"] = args["velocity_after"].double()
    else:
        args["foot_contact"] = torch.ones(1)
    with pytest.raises(ValueError):
        rolling_redirect_contact_reward(**args)
