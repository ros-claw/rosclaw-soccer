import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_safe_continuation_credit import receiver_safe_continuation_credit


def score(length, termination, *, failed=False, excluded=False, handoff=2, controls=4):
    f, v, body = (
        np.zeros((length, 10, 2, 3)),
        np.zeros((length, 10, 4)),
        np.zeros((length, 2), bool),
    )
    f[0, 1, 1, 0] = 11
    f[0, 2, 1, 2] = 2
    f[3, 0, 1, 0] = 11
    v[3, 0, 0] = 1.1
    body[-1, 0] = failed
    return receiver_safe_continuation_credit(
        f,
        v,
        body,
        termination=termination,
        controls=controls,
        handoff_control=handoff,
        ball_velocity_index=0,
        return_direction=np.array([1.0, 0.0]),
        excluded_contact_observed=excluded,
    )


def test_body_failure_suppresses_all_early_bonuses():
    reward, info = score(5, "BODY_FAILURE", failed=True)
    assert np.all(reward <= 0)
    assert not info["full_horizon_safety_eligible"]
    assert not info["events"]["directed_return_proxy"]["accepted"]


def test_complete_safe_horizon_allows_observed_proxies():
    reward, info = score(600, "WORLD_HORIZON_COMPLETE")
    assert info["full_horizon_safety_eligible"]
    assert info["events"]["directed_return_proxy"]["accepted"]
    assert reward.sum() > 0


def test_censor_and_unknown_contact_do_not_allow_bonuses():
    reward, info = score(300, "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300", handoff=None)
    assert np.all(reward <= 0)
    assert info["safety_observation_censored"]
    reward, info = score(600, "WORLD_HORIZON_COMPLETE", excluded=True)
    assert np.all(reward <= 0)
    assert not info["full_horizon_safety_eligible"]


@pytest.mark.parametrize(
    "length,termination,failed,handoff",
    [
        (299, "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300", False, None),
        (300, "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300", False, 2),
        (5, "BODY_FAILURE", False, 2),
        (5, "WORLD_HORIZON_COMPLETE", False, 2),
        (5, "unknown", False, 2),
    ],
)
def test_reject_inconsistent_or_unknown_termination(length, termination, failed, handoff):
    with pytest.raises(ValueError):
        score(length, termination, failed=failed, handoff=handoff)
