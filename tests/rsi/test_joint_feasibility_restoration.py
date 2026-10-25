import numpy as np
import pytest

from rosclaw_soccer.rsi.joint_feasibility_restoration import (
    restore_joint_feasibility,
    task_violation,
)


def test_small_local_progress_is_possible_without_claiming_complete_task_feasibility():
    baseline = np.array([0.3, 0.8, 50.0])
    jac = np.zeros((3, 29))
    jac[:, 0] = [3.0, -2.0, -50.0]
    candidate, report = restore_joint_feasibility(
        baseline, jac, np.zeros(29), [0], force_scale=50.0
    )
    assert candidate is not None and 0 < candidate[0] <= 0.03
    assert report["predicted_violation"] < report["current_violation"]
    assert report["predicted_violation"] > 0
    assert report["requires_new_complete_healthy_NN_physics"]


def test_accumulated_reference_does_not_expand_per_joint_authority():
    baseline = np.array([0.3, 0.8, 50.0])
    jac = np.zeros((3, 29))
    jac[:, 0] = [3.0, -2.0, -50.0]
    center = np.zeros(29)
    center[0] = 0.055
    candidate, _ = restore_joint_feasibility(baseline, jac, center, [0], force_scale=50.0)
    assert candidate is not None
    assert abs(candidate).max() <= 0.06 and candidate[0] - center[0] < 0.006


def test_no_model_progress_abstains():
    candidate, _ = restore_joint_feasibility(
        np.array([0.3, 0.8, 50.0]),
        np.zeros((3, 29)),
        np.zeros(29),
        list(range(29)),
        force_scale=50.0,
    )
    assert candidate is None


def test_force_normalization_is_fixed_not_reduced_to_fake_progress():
    x = np.array([0.8, 0.8, 20.0])
    assert task_violation(x, force_scale=50.0) != task_violation(x, force_scale=20.0)
    with pytest.raises(ValueError, match="fixed initial"):
        task_violation(x, force_scale=0.5)
