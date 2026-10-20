import pytest
import torch

from rosclaw_soccer.rsi.identity_kick_credit import IdentityKickCredit


def inputs(*, step=0, foot=True, bad=False):
    force = torch.zeros(2, 10, 2)
    if foot:
        force[:, 4, 0] = 20
    forbidden = torch.zeros(2, 10)
    if bad:
        forbidden[0, 8] = 2
    return dict(
        step=step,
        forces=force,
        expected=torch.zeros(2, dtype=torch.long),
        ball_velocity=torch.tensor([[2.0, 0, 0], [2.0, 0, 0]]),
        goal_delta=torch.tensor([[5.0, 0, 0], [5.0, 0, 0]]),
        forbidden_forces=forbidden,
    )


def test_any_forbidden_contact_vetoes_this_interval_and_only_that_world():
    credit = IdentityKickCredit(2, "cpu")
    r = credit.step(**inputs(bad=True))
    assert r["contact"].tolist() == [0, 1]
    assert r["wrong_contact"].tolist() == [0, 0]
    assert r["forbidden_contact"].tolist() == [1, 0]
    assert r["speed"][0] == 0 and r["alignment"][0] == 0
    r = credit.step(**inputs(step=1))
    assert r["contact"].tolist() == [0, 0]
    credit.reset(torch.tensor([0]))
    r = credit.step(**inputs(step=2))
    assert r["contact"].tolist() == [1, 0]


def test_late_knee_contact_blocks_velocity_credit_without_fabricating_future():
    credit = IdentityKickCredit(2, "cpu")
    first = credit.step(**inputs())
    assert first["contact"].tolist() == [1, 1]
    r = credit.step(**inputs(step=1, foot=False, bad=True))
    assert r["forbidden_contact"].tolist() == [1, 0]
    assert r["speed"][0] == 0 and r["speed"][1] > 0
    r = credit.step(**inputs(step=2, foot=False, bad=True))
    assert r["forbidden_contact"].tolist() == [0, 0]
    assert r["speed"][0] == 0


def test_forbidden_threshold_is_strictly_above_one_newton():
    values = inputs()
    values["forbidden_forces"][:, 0] = 1
    assert IdentityKickCredit(2, "cpu").step(**values)["contact"].tolist() == [1, 1]
    values["forbidden_forces"][0, 0] = 1.001
    assert IdentityKickCredit(2, "cpu").step(**values)["contact"].tolist() == [0, 1]


def test_intent_cannot_silently_switch_inside_same_step_or_lifetime():
    credit = IdentityKickCredit(2, "cpu")
    values = inputs()
    r = credit.step(**values)
    assert credit.step(**values) is r
    values["forbidden_forces"] = None
    with pytest.raises(ValueError, match="intent"):
        credit.step(**values)
    values["step"] = 1
    with pytest.raises(ValueError, match="intent"):
        credit.step(**values)


@pytest.mark.parametrize("kind", ["shape", "nan", "negative"])
def test_invalid_forbidden_force_fails_closed(kind):
    values = inputs()
    if kind == "shape":
        values["forbidden_forces"] = torch.zeros(2, 9)
    elif kind == "nan":
        values["forbidden_forces"][0, 0] = float("nan")
    else:
        values["forbidden_forces"][0, 0] = -1
    with pytest.raises(ValueError, match="forbidden"):
        IdentityKickCredit(2, "cpu").step(**values)
