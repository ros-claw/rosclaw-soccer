"""Leave-one-course-out diagnostics must not train on held-out labels."""

import numpy as np
from rsi_r1_temporal_success_diagnostics_v197 import _auc, _cross_validate


def test_auc_counts_pairwise_ties() -> None:
    assert _auc(np.asarray([1.0, 2.0, 2.0]), np.asarray([False, True, False])) == 0.75


def test_transferable_direction_is_learned_from_other_courses() -> None:
    groups = np.repeat(np.arange(3), 4)
    labels = np.tile(np.asarray([False, False, True, True]), 3)
    vectors = np.tile(np.asarray([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0]]), (3, 1))
    auc, per_course = _cross_validate(vectors, labels, groups, (0, 1, 2))
    assert auc == 1.0
    assert per_course == {0: 1.0, 1: 1.0, 2: 1.0}


def test_inconsistent_course_directions_do_not_transfer() -> None:
    groups = np.repeat(np.arange(3), 4)
    labels = np.tile(np.asarray([False, False, True, True]), 3)
    vectors = np.tile(np.asarray([[-2.0, 0], [-1.0, 0], [1.0, 0], [2.0, 0]]), (3, 1))
    vectors[groups == 2] *= -1
    auc, per_course = _cross_validate(vectors, labels, groups, (0, 1, 2))
    assert auc < 1.0
    assert per_course[2] == 0.0
