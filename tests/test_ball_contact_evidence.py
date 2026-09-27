"""Physics microstep contact labeling for football skill reward."""

import numpy as np
import pytest

from rosclaw_soccer.sim.ball_contact_evidence import classify_ball_body_contacts


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
