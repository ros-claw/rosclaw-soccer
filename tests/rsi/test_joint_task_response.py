import numpy as np
import pytest

from rosclaw_soccer.rsi.joint_task_response import propose_joint_task_response, response_features


def probes(baseline):
    rows = []
    for i in range(29):
        for sign in (-1, 1):
            y = baseline.copy()
            if i == 0:
                y[0] += 50 * sign * 0.03
                y[2] = max(0, y[2] - 100 * sign * 0.03)
            rows.append((i, sign, y))
    return rows


def test_model_coordinates_ball_quality_and_collision_severity_without_claiming_physics():
    baseline = np.array([0.7, 0.0, 2.0], dtype=np.float64)
    coeff, report = propose_joint_task_response(baseline, probes(baseline))
    assert coeff is not None and coeff.shape == (2, 29)
    assert coeff[0, 0] > 0 and abs(coeff).sum() <= 0.06000001
    assert not coeff[1].any()
    assert report["predicted_features"][0] >= 1.05
    assert report["predicted_features"][2] <= 0.500001
    assert not report["physical_response_verified"]


def test_no_censored_or_body_failure_values_fabricated():
    baseline = np.array([0.7, 0.0, 2.0], dtype=np.float64)
    assert propose_joint_task_response(None, probes(baseline))[0] is None
    assert (
        propose_joint_task_response(baseline, [(i, s, None) for i, s, _ in probes(baseline)])[0]
        is None
    )


def test_infeasible_model_abstains_and_missing_probes_reject():
    baseline = np.array([0.1, 0.0, 2.0], dtype=np.float64)
    rows = [(i, s, baseline.copy()) for i, s, _ in probes(baseline)]
    assert propose_joint_task_response(baseline, rows)[0] is None
    with pytest.raises(ValueError, match="all29"):
        propose_joint_task_response(baseline, rows[:-1])


def test_formal_task_force_excludes_diagnostic_tail():
    force = np.zeros((1280, 1, 3))
    force[700:, 0, 2] = 100
    prediction = dict(controls=128, donor_control_frame=230, ball_forces=force)
    window = dict(
        predicted_body_healthy=True, quality_observed=True, forward_60_m=2.0, lateral_60_m=0.1
    )
    assert response_features(prediction, window)[2] == 0
