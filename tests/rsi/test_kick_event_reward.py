import numpy as np
import pytest

from rosclaw_soccer.rsi.kick_event_reward import event_reward_contract, kick_event_rewards


def trace(n=360):
    ball = np.zeros((n, 3), np.float64)
    body = np.zeros((n, 10, 7), np.float64)
    body[:, :, 2] = 0.8
    body[:, :, 3] = 1
    forces = np.zeros((n, 10, 3), np.float64)
    return dict(ball_after=ball, pelvis_native=body, force_native=forces)


def shot(n=360, foot=100):
    x = trace(n)
    x["force_native"][foot, 3, 0] = 2
    x["ball_after"][foot:, 0] = np.arange(n - foot) / 20
    return x


def test_quality_is_paid_when_observed_not_at_contact_or_end():
    result = kick_event_rewards(**shot())
    assert np.flatnonzero(result["rewards"]).tolist() == [100, 160]
    assert result["rewards"][100] == 1 and result["rewards"][160] == 3
    assert result["rewards"].sum() == 4
    assert result["quality_observed_control"] == 160
    assert result["discount"] == 1
    assert event_reward_contract()["worst_unsafe_total_return_upper_bound"] == -6


@pytest.mark.parametrize("cut", [1, 99, 100, 101, 160, 161, 299, 300, 340])
def test_prefix_invariance_with_changed_future_and_shortened_trace(cut):
    source = shot()
    original = kick_event_rewards(**source)["rewards"][:cut]
    shortened = {k: v[:cut].copy() for k, v in source.items()}
    np.testing.assert_array_equal(kick_event_rewards(**shortened)["rewards"], original)
    changed = {k: v.copy() for k, v in source.items()}
    changed["ball_after"][cut:, 1] = 7
    changed["force_native"][cut:, :, 2] = 50
    changed["pelvis_native"][cut:, :, 2] = 0.4
    np.testing.assert_array_equal(kick_event_rewards(**changed)["rewards"][:cut], original)


@pytest.mark.parametrize("when", [0, 99, 100, 101, 160, 299])
def test_nonfoot_cost_cannot_be_bought_by_arbitrarily_long_ball_travel(when):
    x = shot()
    x["ball_after"][100:, 0] *= 10000
    x["force_native"][when, 0, 2] = 1.000001
    r = kick_event_rewards(**x)
    assert r["components"][when, 2] == -10
    assert r["first_nonfoot_control"] == when
    assert r["rewards"].sum() <= -6


def test_posttask_nonfoot_does_not_rewrite_exam_but_late_body_failure_costs_now():
    x = shot()
    x["force_native"][310, 0, 2] = 2
    x["pelvis_native"][320:, :, 2] = 0.6
    r = kick_event_rewards(**x)
    assert r["first_nonfoot_control"] is None
    assert r["first_body_failure_control"] == 320
    assert r["rewards"][320] == -10
    assert r["rewards"][160] == 3
    assert r["rewards"].sum() == -6


def test_deadline_requires_actual_observation_and_last_valid_quality_boundary():
    assert kick_event_rewards(**trace(299))["rewards"].sum() == 0
    assert kick_event_rewards(**trace(300))["rewards"][-1] == -3
    valid = kick_event_rewards(**shot(300, 239))
    assert valid["quality_observed_control"] == 299
    censored = kick_event_rewards(**shot(360, 240))
    assert censored["quality_observed_control"] is None
    assert censored["rewards"][240] == 1 and censored["rewards"][299] == -2
    assert censored["rewards"][300:].sum() == 0


def test_first_events_only_and_upright_native_substep_not_control_average():
    x = shot()
    x["force_native"][101:130, :, 0] = 20
    x["force_native"][200:230, :, 2] = 20
    x["pelvis_native"][250, 9, 3:7] = [0, 1, 0, 0]
    r = kick_event_rewards(**x)
    assert r["components"][:, 0].sum() == 1
    assert r["components"][:, 2].sum() == -10
    assert r["first_body_failure_control"] == 250
    assert r["components"][250, 3] == -10


@pytest.mark.parametrize(
    "defect", ["nan", "negative_force", "quaternion", "dtype", "shape", "empty", "oversize"]
)
def test_reject_invalid_physical_measurements(defect):
    x = trace()
    if defect == "nan":
        x["ball_after"][0, 0] = float("nan")
    elif defect == "negative_force":
        x["force_native"][0, 0, 0] = -1
    elif defect == "quaternion":
        x["pelvis_native"][0, 0, 3] = 0
    elif defect == "dtype":
        x["force_native"] = x["force_native"].astype(np.float32)
    elif defect == "shape":
        x["force_native"] = x["force_native"][:, :9]
    elif defect == "empty":
        x = trace(0)
    else:
        x = trace(361)
    with pytest.raises(ValueError):
        kick_event_rewards(**x)
