import pytest

from rosclaw_soccer.training.directed_velocity_reward import contact_velocity_progress

torch = pytest.importorskip("torch")


def test_sideways_kick_cannot_earn_forward_only_progress():
    before = torch.tensor([[0.0, 0.0], [0.0, 0.0], [0.0, 0.0]])
    after = torch.tensor([[0.8, 1.2], [0.8, 0.0], [0.8, 0.0]])
    target = torch.tensor([[0.8, 0.0]]).repeat(3, 1)
    result = contact_velocity_progress(before, after, target, torch.tensor([True, True, False]))
    assert torch.allclose(result, torch.tensor([-0.3, 0.3, 0.0]))


def test_overspeed_and_reverse_progress_are_penalized():
    before = torch.tensor([[0.8, 0.0], [0.0, 0.0]])
    after = torch.tensor([[1.8, 0.0], [-0.2, 0.0]])
    target = torch.tensor([[0.8, 0.0]]).repeat(2, 1)
    assert (
        contact_velocity_progress(before, after, target, torch.ones(2, dtype=torch.bool)) < 0
    ).all()


@pytest.mark.parametrize("invalid", ["nan", "gradient", "dtype", "shape", "mask"])
def test_invalid_inputs_rejected(invalid):
    before, after, target = (torch.zeros(2, 2) for _ in range(3))
    mask = torch.ones(2, dtype=torch.bool)
    if invalid == "nan":
        after[0, 0] = float("nan")
    elif invalid == "gradient":
        before.requires_grad_(True)
    elif invalid == "dtype":
        target = target.double()
    elif invalid == "shape":
        target = target[:1]
    else:
        mask = mask.float()
    with pytest.raises(ValueError):
        contact_velocity_progress(before, after, target, mask)
