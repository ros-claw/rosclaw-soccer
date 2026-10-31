"""Physics microstep contact labeling for football skill reward."""

import numpy as np
import pytest

from rosclaw_soccer.sim.ball_contact_evidence import (
    classify_ball_body_contacts,
    classify_kick_foot_contacts,
)


def test_foot_first_is_not_clean_if_knee_follows() -> None:
    force = np.zeros((2, 10, 6))
    force[1, 4, 1] = 185.0
    force[1, 5, 5] = 108.0
    result = classify_ball_body_contacts(force)
    assert result.first_foot_microstep == 14
    assert result.first_nonfoot_microstep == 15
    assert result.foot_first
    assert not result.clean_foot_only


def test_knee_first_and_clean_foot_distinguished() -> None:
    knee = np.zeros((1, 10, 6))
    knee[0, 2, 5] = 353.0
    assert not classify_ball_body_contacts(knee).foot_first
    foot = np.zeros((1, 10, 6))
    foot[0, 2, 0] = 205.0
    result = classify_ball_body_contacts(foot)
    assert result.foot_first and result.clean_foot_only


@pytest.mark.parametrize(
    "force",
    [
        np.ones((1, 10, 2)),
        np.full((1, 10, 6), float("nan")),
        -np.ones((1, 10, 6)),
    ],
)
def test_invalid_contacts_fail_closed(force: np.ndarray) -> None:
    with pytest.raises(ValueError):
        classify_ball_body_contacts(force)


def test_kick_foot_identity_does_not_credit_other_foot() -> None:
    force = np.zeros((2, 10, 3))
    force[1, 4, 1] = 157.0
    result = classify_kick_foot_contacts(force, designated_column=0)
    assert result.first_contact == "other"
    assert result.first_designated_microstep is None
    assert result.first_other_microstep == 14
    assert result.designated_contact_microsteps == 0
    assert result.other_contact_microsteps == 1
    reversed_result = classify_kick_foot_contacts(force, designated_column=1)
    assert reversed_result.first_contact == "designated"


def test_kick_foot_first_simultaneous_threshold_and_absence() -> None:
    force = np.zeros((1, 5, 3))
    assert classify_kick_foot_contacts(force, designated_column=0).first_contact == "none"
    force[0, 1, :2] = 1.0  # contact threshold is strict, as in the legacy contract
    force[0, 2, :2] = 2.0
    result = classify_kick_foot_contacts(force, designated_column=0)
    assert result.first_contact == "simultaneous"
    assert result.first_designated_microstep == result.first_other_microstep == 2
    force[0, 2, 0] = 0
    force[0, 3, 0] = 3
    assert classify_kick_foot_contacts(force, designated_column=0).first_contact == "other"


@pytest.mark.parametrize("column", [-1, 2, True, 0.0, "0"])
def test_kick_foot_requires_explicit_valid_column(column: object) -> None:
    with pytest.raises(ValueError):
        classify_kick_foot_contacts(np.zeros((1, 5, 3)), designated_column=column)  # type: ignore[arg-type]


def test_kick_foot_rejects_invalid_trace_and_threshold() -> None:
    with pytest.raises(ValueError):
        classify_kick_foot_contacts(np.full((1, 5, 3), np.nan), designated_column=0)
    with pytest.raises(ValueError):
        classify_kick_foot_contacts(np.zeros((1, 5, 3)), designated_column=0, threshold_n=0)
