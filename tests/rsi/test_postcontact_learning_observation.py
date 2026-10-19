"""Current-state feature math, not learning or physical skill evidence."""

import inspect

import numpy as np
import pytest

from rosclaw_soccer.rsi.postcontact_learning_observation import (
    current_postcontact_features,
    observation_contract,
)


def inputs():
    q = np.zeros(43)
    q[2] = 0.793
    q[3] = q[39] = 1
    q[36:39] = [0.8, 0.2, 0.13]
    return q, np.zeros(41), np.zeros(29), np.array([2.0, 0, 0, 0, 0, 0])


def feature(q, v, t, p, frame=191):
    return current_postcontact_features(q, v, t, frame=frame, previous_contact_forces=p)


def test_whole_owned_current_observation_and_declared_boundary():
    q, v, t, p = inputs()
    originals = [a.copy() for a in (q, v, t, p)]
    x = feature(q, v, t, p)
    assert x.shape == (121,) and x.dtype == np.float32 and not x.flags.writeable
    np.testing.assert_array_equal(x[-3:-1], [1, 0])
    assert x[-1] == pytest.approx(109 / 300)
    for a, b in zip((q, v, t, p), originals, strict=True):
        np.testing.assert_array_equal(a, b)
    assert not observation_contract()["future_outcome_or_event_time_used"]
    assert not observation_contract()["learning_or_execution_authorized"]
    assert "SIM ground truth" in observation_contract()["ball_state"]


def test_horizontal_translation_and_quaternion_double_cover():
    q, v, t, p = inputs()
    original = feature(q, v, t, p)
    q[:2] += [100.0, -50.0]
    q[36:38] += [100.0, -50.0]
    q[3:7] *= -1
    q[39:43] *= -1
    np.testing.assert_allclose(feature(q, v, t, p), original, rtol=0, atol=1e-6)


def test_ball_state_current_target_contact_side_and_clock_are_observed():
    q, v, t, p = inputs()
    base = feature(q, v, t, p)
    for a, index in ((q, 36), (v, 35), (t, 3)):
        a[index] += 0.1
        assert not np.array_equal(feature(q, v, t, p), base)
        a[index] -= 0.1
    p[:2] = [0, 3]
    np.testing.assert_array_equal(feature(q, v, t, p)[-3:-1], [0, 1])
    assert feature(q, v, t, p, 299)[-1] == pytest.approx(1 / 300)


@pytest.mark.parametrize(
    "bad",
    ["no_contact", "weak", "negative", "nan", "dtype", "qshort", "quat", "clock", "bool_clock"],
)
def test_wrong_or_predicted_only_observation_rejected(bad):
    q, v, t, p = inputs()
    f = 191
    if bad == "no_contact":
        p[:] = 0
    elif bad == "weak":
        p[0] = 1
    elif bad == "negative":
        p[2] = -1
    elif bad == "nan":
        v[0] = np.nan
    elif bad == "dtype":
        q = q.astype(np.float32)
    elif bad == "qshort":
        q = q[:-1]
    elif bad == "quat":
        q[3] = 2
    elif bad == "clock":
        f = 300
    else:
        f = True
    with pytest.raises(ValueError):
        feature(q, v, t, p, f)


def test_no_future_input_or_source_case_shortcut_in_function_signature():
    names = set(inspect.signature(current_postcontact_features).parameters)
    assert names == {
        "qpos",
        "qvel",
        "current_nominal_target_rad",
        "frame",
        "previous_contact_forces",
    }
