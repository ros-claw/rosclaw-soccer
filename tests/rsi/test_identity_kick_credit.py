import pytest

torch = pytest.importorskip("torch")
from rosclaw_soccer.rsi.identity_kick_credit import IdentityKickCredit  # noqa: E402


def values():
    forces = torch.zeros((4, 10, 2))
    forces[0, 2, 0] = 20  # left then right: left is really first
    forces[0, 7, 1] = 30
    forces[1, 1, 1] = 20  # wrong foot; no reward even if left follows
    forces[1, 3, 0] = 30
    forces[2, 4] = 20  # simultaneous ambiguous
    return dict(
        forces=forces,
        expected=torch.zeros(4, dtype=torch.long),
        ball_velocity=torch.tensor([[2.0, 0.0, 0.0]]).repeat(4, 1),
        goal_delta=torch.tensor([[3.0, 0.0, 0.0]]).repeat(4, 1),
    )


def test_actual_first_identity_and_ambiguity_not_nearest_foot():
    tracker = IdentityKickCredit(4, "cpu")
    data = values()
    reward = tracker.step(step=1, **data)
    assert reward["contact"].tolist() == [1, 0, 0, 0]
    assert reward["wrong_contact"].tolist() == [0, 1, 1, 0]
    assert reward["alignment"].tolist() == [1, 0, 0, 0]
    assert tracker.step(step=1, **data) is reward
    second = tracker.step(step=2, **data)
    assert second["contact"].sum() == 0
    for step in range(3, 7):
        reward = tracker.step(step=step, **data)
    assert reward["speed"].sum() == 0
    tracker.reset(torch.tensor([1], dtype=torch.long))
    data["forces"][1] = 0
    data["forces"][1, 8, 0] = 20
    assert tracker.step(step=7, **data)["contact"].tolist() == [0, 1, 0, 0]
    with pytest.raises(ValueError, match="skipped"):
        tracker.step(step=9, **data)


def test_no_contact_does_not_earn_credit_for_already_fast_ball():
    tracker = IdentityKickCredit(4, "cpu")
    data = values()
    data["forces"].zero_()
    assert all(v.sum() == 0 for v in tracker.step(step=0, **data).values())


@pytest.mark.parametrize("kind", ["nan", "shape", "side", "force", "reset"])
def test_invalid_credit_fails_closed(kind):
    tracker = IdentityKickCredit(4, "cpu")
    data = values()
    if kind == "nan":
        data["ball_velocity"][0, 0] = float("nan")
    elif kind == "shape":
        data["forces"] = torch.zeros((4, 2))
    elif kind == "side":
        data["expected"][0] = -1
    elif kind == "force":
        data["forces"][0, 0, 0] = -1
    else:
        with pytest.raises(ValueError):
            tracker.reset(torch.tensor([-1]))
        return
    with pytest.raises(ValueError):
        tracker.step(step=0, **data)
