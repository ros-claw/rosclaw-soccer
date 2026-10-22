import numpy as np
import pytest

from rosclaw_soccer.rsi.motor_option_credit import (
    contact_skill_return,
    observed_option_tail,
    terminal_option_returns,
)


def test_neural_standing_delay_is_not_silently_dropped():
    r = terminal_option_returns(289, 100, 10.0)
    assert r.shape == (289,)
    assert r[0] == pytest.approx(10 * 0.995**388)
    assert r[-1] == pytest.approx(10 * 0.995**100)
    assert np.all(r[1:] > r[:-1])


def test_physical_failures_keep_credit_without_phantom_recovery():
    assert (
        observed_option_tail(np.array([False, True]), None, body_failed=True, standing_healthy=None)
        == 0
    )
    assert (
        observed_option_tail(
            np.zeros(289, dtype=bool),
            np.array([False, True]),
            body_failed=True,
            standing_healthy=False,
        )
        == 2
    )
    assert (
        observed_option_tail(
            np.zeros(289, dtype=bool),
            np.zeros(100, dtype=bool),
            body_failed=False,
            standing_healthy=True,
        )
        == 100
    )
    with pytest.raises(ValueError):
        observed_option_tail(
            np.zeros(289, dtype=bool), None, body_failed=False, standing_healthy=None
        )
    with pytest.raises(ValueError):
        observed_option_tail(np.array([True, False]), None, body_failed=True, standing_healthy=None)


def test_body_failure_is_not_worth_a_precise_shot():
    result = dict(
        body_failed=False,
        forbidden=False,
        nonfoot_impulse_Ns=0.0,
        postcontact=dict(
            correct_foot_first=True,
            post_contact_quality_observed=True,
            goal_progress_1p2s_m=4.0,
            goal_lateral_1p2s_m=0.1,
            maximum_pelvis_retreat_after_contact_m=0.0,
        ),
    )
    clean = contact_skill_return(result)
    result["forbidden"] = True
    assert contact_skill_return(result) < clean - 8
    result["body_failed"] = True
    assert contact_skill_return(result) == -30


@pytest.mark.parametrize(
    "rows,stand,reward", [(0, 100, 1.0), (301, 0, 1.0), (1, 101, 1.0), (1, 0, float("nan"))]
)
def test_invalid_or_unobserved_credit_rejected(rows, stand, reward):
    with pytest.raises(ValueError):
        terminal_option_returns(rows, stand, reward)
