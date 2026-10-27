import numpy as np
import pytest

from rosclaw_soccer.rsi.actual_pass_contact_risk import contact_risk_data, training_recall_threshold


def test_risk_uses_before_state_and_groups_shared_contexts():
    n = 32
    data = dict(
        qpos_before=np.zeros((n, 79)),
        qvel_before=np.zeros((n, 76)),
        receiver_observation=np.zeros((n, 98)),
        receiver_action_applied=np.zeros((n, 29)),
        ball_force_max_after=np.zeros((n, 2, 3)),
        course_index=np.arange(n),
    )
    data["ball_force_max_after"][7, 1, 2] = 2.0
    courses = [dict(index=i, course=dict(context=(1, 4, 20, 21)[i % 4])) for i in range(32)]
    x, y, groups = contact_risk_data(data, courses)
    assert x.shape == (32, 282)
    assert y.sum() == 1 and y[7] == 1
    assert groups[7] == groups[3] == 21
    del data["qvel_before"]
    with pytest.raises(ValueError, match="complete causal"):
        contact_risk_data(data, courses)


def test_aggregate_courses_preserve_shared_context_groups():
    n = 72
    data = dict(
        qpos_before=np.zeros((n, 79)),
        qvel_before=np.zeros((n, 76)),
        receiver_observation=np.zeros((n, 98)),
        receiver_action_applied=np.zeros((n, 29)),
        ball_force_max_after=np.zeros((n, 2, 3)),
        course_index=np.arange(n),
    )
    courses = [dict(index=i, course=dict(context=(1, 4, 20, 21)[i % 4])) for i in range(n)]
    x, _, groups = contact_risk_data(data, courses)
    assert x.shape == (72, 282)
    assert (groups == 1).sum() == 18
    data["course_index"][0] = 999
    with pytest.raises(ValueError, match="declared course"):
        contact_risk_data(data, courses)


def test_training_only_threshold_preserves_95_percent_positive_recall():
    p = np.arange(1, 101, dtype=float) / 200
    labels = np.ones(100, dtype=bool)
    threshold = training_recall_threshold(p, labels)
    assert (p >= threshold).mean() >= 0.95
    with pytest.raises(ValueError, match="positives"):
        training_recall_threshold(p, ~labels)


def test_new_context_banks_are_grouped_without_fixed_four_context_assumption():
    n = 168
    data = dict(
        qpos_before=np.zeros((n, 79)),
        qvel_before=np.zeros((n, 76)),
        receiver_observation=np.zeros((n, 98)),
        receiver_action_applied=np.zeros((n, 29)),
        ball_force_max_after=np.zeros((n, 2, 3)),
        course_index=np.arange(n),
    )
    courses = [dict(index=i, course=dict(context=i % 16)) for i in range(n)]
    _, _, groups = contact_risk_data(data, courses)
    assert len(set(groups)) == 16
    assert groups[0] == groups[16]
