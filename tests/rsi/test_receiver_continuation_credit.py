import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_continuation_credit import receiver_continuation_credit


def evaluate(force, velocity, failures, **kwargs):
    return receiver_continuation_credit(
        force,
        velocity,
        failures,
        controls=kwargs.pop("controls", len(force)),
        handoff_control=kwargs.pop("handoff_control", None),
        ball_velocity_index=0,
        return_direction=np.array([1.0, 0.0]),
        excluded_contact_observed=kwargs.pop("excluded_contact_observed", False),
        **kwargs,
    )


def test_no_contact_is_not_success():
    f, v, failed = np.zeros((5, 10, 2, 3)), np.zeros((5, 10, 4)), np.zeros((5, 2), bool)
    reward, info = evaluate(f, v, failed)
    assert not reward.any()
    assert not info["events"]
    assert info["handoff_is_not_possession"]


def test_actual_qualified_handoff_and_later_strike_are_discounted_from_true_time():
    f, v, failed = np.zeros((8, 10, 2, 3)), np.zeros((8, 10, 4)), np.zeros((8, 2), bool)
    f[1, 1, 1, 0] = 11
    f[1, 3, 1, 2] = 2
    f[6, 2, 1, 0] = 11
    v[6, 2, 0] = 1.1
    reward, info = evaluate(f, v, failed, controls=4, handoff_control=4)
    assert info["events"]["foot_qualified_handoff_proxy"]["delay"] == 1
    assert info["events"]["directed_return_proxy"]["delay"] == 3
    assert reward[1] == pytest.approx(-0.2)
    assert reward[3] == pytest.approx(0.99 + 0.99**3)


def test_handoff_requires_prior_foot_and_excluded_contact_penalty():
    f, v, failed = np.zeros((5, 10, 2, 3)), np.zeros((5, 10, 4)), np.zeros((5, 2), bool)
    reward, info = evaluate(f, v, failed, handoff_control=2, excluded_contact_observed=True)
    assert "foot_qualified_handoff_proxy" not in info["events"]
    assert reward[-1] == -2


def test_body_failure_must_be_final_and_penalized_without_fabricated_tail():
    f, v, failed = np.zeros((5, 10, 2, 3)), np.zeros((5, 10, 4)), np.zeros((5, 2), bool)
    failed[-1, 0] = True
    reward, info = evaluate(f, v, failed, controls=3)
    assert reward[-1] == pytest.approx(-2 * 0.99**2)
    assert info["first_failure_control"] == 4
    failed[1, 0] = True
    with pytest.raises(ValueError):
        evaluate(f, v, failed)


def test_reject_bad_events_and_nonfinite_data():
    f, v, failed = np.zeros((5, 10, 2, 3)), np.zeros((5, 10, 4)), np.zeros((5, 2), bool)
    with pytest.raises(ValueError):
        evaluate(f, v, failed, handoff_control=5)
    f[0, 0, 1, 0] = np.nan
    with pytest.raises(ValueError):
        evaluate(f, v, failed)
