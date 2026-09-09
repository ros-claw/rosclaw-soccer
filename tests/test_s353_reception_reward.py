import pytest

from rosclaw_soccer.training.reception_reward import (
    controlled_reception_mask,
    reception_terminal_reward,
)


def outcomes():
    torch = pytest.importorskip("torch")
    return dict(
        seen_foot_contact=torch.tensor([False, True, True, True]),
        forbidden_contact=torch.zeros(4, dtype=torch.bool),
        body_safe=torch.ones(4, dtype=torch.bool),
        stable_tail=torch.tensor([False, False, True, True]),
        first_contact_sec=torch.tensor([-1.0, 0.3, 0.3, 1.1]),
        elapsed_sec=1.2,
    )


def test_failed_touch_is_not_worse_than_avoiding_ball_and_late_touch_is_not_control():
    torch = pytest.importorskip("torch")
    data = outcomes()
    assert torch.equal(controlled_reception_mask(**data), torch.tensor([False, False, True, False]))
    assert torch.equal(reception_terminal_reward(**data), torch.tensor([-2.0, -1.0, 10.0, -1.0]))


@pytest.mark.parametrize("field", ["forbidden_contact", "body_safe"])
def test_body_or_contact_failure_cannot_earn_control_reward(field):
    data = outcomes()
    data[field][2] = field == "forbidden_contact"
    assert reception_terminal_reward(**data)[2].item() == -1


@pytest.mark.parametrize("bad", ["future", "missing-event", "nonfinite", "boolean-time"])
def test_inconsistent_physical_clock_is_rejected(bad):
    data = outcomes()
    if bad == "future":
        data["first_contact_sec"][2] = 2
    if bad == "missing-event":
        data["first_contact_sec"][0] = 0.2
    if bad == "nonfinite":
        data["first_contact_sec"][0] = float("nan")
    if bad == "boolean-time":
        data["elapsed_sec"] = True
    with pytest.raises(ValueError):
        controlled_reception_mask(**data)
