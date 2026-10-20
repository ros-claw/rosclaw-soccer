from dataclasses import asdict

import numpy as np
import pytest

from rosclaw_soccer.rsi.motor_student_bank import (
    training_observation_statistics,
    validation_course_indices,
)
from rosclaw_soccer.training.expanded_entry_courses import expanded_entry_courses


def test_validation_is_whole_course_stratified_and_label_independent():
    courses = [asdict(c) for c in expanded_entry_courses()]
    selected = validation_course_indices(courses)
    assert len(selected) == 8
    assert [
        sum(c["course_index"] in selected and c["course"][2] == s for c in courses)
        for s in (0, -0.15, 0.15)
    ] == [4, 2, 2]
    for c in courses:
        c["score"] = "arbitrary label ignored"
    assert validation_course_indices(courses) == selected
    assert validation_course_indices(list(reversed(courses))) == selected


def test_observation_statistics_never_consume_validation_values():
    x = np.zeros((40, 300, 108), dtype=np.float32)
    train = np.zeros(40, dtype=np.bool_)
    train[:32] = True
    mean, std = training_observation_statistics(x, train)
    x[~train] = 1000
    new_mean, new_std = training_observation_statistics(x, train)
    np.testing.assert_array_equal(mean, new_mean)
    np.testing.assert_array_equal(std, new_std)
    np.testing.assert_allclose(std, 0.05)


def test_incomplete_bank_cannot_silently_rebalance():
    with pytest.raises(ValueError):
        validation_course_indices([asdict(c) for c in expanded_entry_courses()[:-1]])
