"""Known linear synthetic system; not a G1 locomotion success fixture."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.locomotion_response_model import fit_training, predict, validate_prediction
from rosclaw_soccer.rsi.locomotion_response_protocol import response_courses


def fixture():
    w = np.zeros((5, 2))
    w[:2] = 0.9 * np.eye(2)
    w[2:4] = 0.1 * np.eye(2)
    rows = []
    for c in response_courses():
        u = np.array([c.command(f)[:2] for f in range(300)])
        v = predict(w, 7, u, np.zeros(2))
        p = np.zeros((300, 2))
        p[1:] = np.cumsum(0.01 * (v[:-1] + v[1:]), axis=0)
        rows.append((u, v, p))
    return rows


def test_train_only_delay_identification_and_free_rollout_validation():
    rows = fixture()
    fit = fit_training([(u, v) for u, v, _ in rows[:9]])
    assert fit["selected"]["delay_frames"] == 7
    assert not fit["validation_used_for_fit"] and fit["neural_optimizer_updates"] == 0
    weights = np.array(fit["selected"]["weights"])
    for u, v, p in rows[9:]:
        r = validate_prediction(weights, 7, u, v, p, onset_frame=75)
        assert r["predictor_research_gate"]
        assert not r["actual_motor_control_validated"]
        assert r["maximum_position_error_m"] < 0.001


def test_validation_rows_cannot_be_appended_to_training():
    rows = fixture()
    with pytest.raises(ValueError):
        fit_training([(u, v) for u, v, _ in rows])


@pytest.mark.parametrize("change", ["shape", "nan", "delay", "explosion"])
def test_model_prediction_fails_closed(change):
    w = np.zeros((5, 2))
    u = np.zeros((300, 2))
    delay = 0
    if change == "shape":
        u = u[:299]
    elif change == "nan":
        w[0, 0] = np.nan
    elif change == "delay":
        delay = 21
    else:
        w[4] = 10.0
    with pytest.raises(ValueError):
        predict(w, delay, u, np.zeros(2))


def test_position_drift_veto_not_hidden_by_good_one_step_prediction():
    u, v, p = fixture()[9]
    p = p.copy()
    p[200:, 0] += 1.0
    w = np.zeros((5, 2))
    w[:2] = 0.9 * np.eye(2)
    w[2:4] = 0.1 * np.eye(2)
    r = validate_prediction(w, 7, u, v, p, onset_frame=75)
    assert r["velocity_rmse_mps"] < 1e-12 and not r["predictor_research_gate"]
