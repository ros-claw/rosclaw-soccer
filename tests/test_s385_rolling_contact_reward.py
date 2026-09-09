import pytest

from rosclaw_soccer.training.reception_reward import rolling_reception_contact_reward

torch = pytest.importorskip("torch")


def reward(before, after, contact=True):
    return rolling_reception_contact_reward(
        velocity_before=torch.tensor([before], dtype=torch.float32),
        velocity_after=torch.tensor([after], dtype=torch.float32),
        foot_contact=torch.tensor([contact]),
    ).item()


def test_reward_prefers_clean_slowdown_over_vertical_pop():
    assert reward([0.6, 0, 0], [0.3, 0, 0]) == pytest.approx(0.4)
    assert reward([0.6, 0, 0], [0.6, 0, 0.3]) < 0
    assert reward([0.6, 0, 0], [0.3, 0, 0.3]) < reward([0.6, 0, 0], [0.3, 0, 0])


def test_no_foot_event_no_contact_credit():
    assert reward([0.6, 0, 0], [0.1, 0, 0], False) == 0


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 101.0])
def test_invalid_velocity_rejected(value):
    with pytest.raises(ValueError):
        reward([value, 0, 0], [0, 0, 0])


def test_nonboolean_contact_and_gradients_rejected():
    with pytest.raises(ValueError):
        reward([0, 0, 0], [0, 0, 0], 1)
    with pytest.raises(ValueError):
        rolling_reception_contact_reward(
            velocity_before=torch.zeros(1, 3, requires_grad=True),
            velocity_after=torch.zeros(1, 3),
            foot_contact=torch.tensor([True]),
        )


def test_empty_or_misaligned_velocity_rejected():
    for before, after in [
        (torch.zeros(0, 3), torch.zeros(0, 3)),
        (torch.zeros(2, 3), torch.zeros(1, 3)),
    ]:
        with pytest.raises(ValueError):
            rolling_reception_contact_reward(
                velocity_before=before, velocity_after=after, foot_contact=torch.tensor([True])
            )
