import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.sampling_credit_diagnostics import sampling_credit_inventory


def record(group=0):
    return dict(
        group=group,
        baseline_course_index=group,
        outcome=dict(
            reward=3.0,
            high_quality=True,
            clean_foot_only=True,
            minimum_pelvis_z_m=0.75,
            maximum_lateral_excursion_m=2.0,
            first_contact_frame=65,
            contact_body_indices=[0],
            forward_60_m=1.5,
            lateral_60_m=0.2,
        ),
    )


def test_failed_positive_credit_is_preserved_not_sanitized():
    rows = [record(0), record(1)]
    rows[1]["outcome"].update(
        high_quality=False, forward_60_m=0.5, lateral_60_m=0.8, minimum_pelvis_z_m=0.5
    )
    values = np.stack([-np.ones(270), np.ones(270)])
    original = copy.deepcopy(rows)
    before = values.copy()
    result = sampling_credit_inventory(rows, values, expected_episodes=2)
    strata = result["overlapping_outcome_strata"]
    assert strata["not_high_quality"]["positive_rows"] == 270
    assert strata["high_quality"]["negative_rows"] == 270
    assert strata["unsafe_pelvis"]["episode_groups"] == [1]
    assert strata["direction_error"]["episode_groups"] == [1]
    assert result["aggregate"]["signed_credit_sum"] == 0
    assert result["aggregate"]["absolute_credit_sum"] == 540
    assert result["contact_time_strata"]["before_first_contact"]["frame_samples"] == 70
    assert result["contact_time_strata"]["at_or_after_first_contact"]["frame_samples"] == 470
    assert rows == original
    np.testing.assert_array_equal(values, before)
    for key in (
        "causal_credit_assignment_proven",
        "source_physics_validated_here",
        "runtime_selection_authorized",
        "training_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert result[key] is False


@pytest.mark.parametrize("first,pre,post", [(0, 0, 270), (30, 0, 270), (299, 269, 1)])
def test_contact_partition_uses_actual_training_frame_origin(first, pre, post):
    row = record()
    row["outcome"]["first_contact_frame"] = first
    result = sampling_credit_inventory([row], np.zeros((1, 270)), expected_episodes=1)
    stages = result["contact_time_strata"]
    assert stages["before_first_contact"]["frame_samples"] == pre
    assert stages["at_or_after_first_contact"]["frame_samples"] == post
    assert result["aggregate"]["zero_rows"] == 270


def test_no_contact_has_no_fictitious_contact_time_and_empty_success_stratum():
    row = record()
    row["outcome"].update(
        first_contact_frame=None, contact_body_indices=[], high_quality=False, clean_foot_only=False
    )
    result = sampling_credit_inventory([row], np.zeros((1, 270)), expected_episodes=1)
    assert result["contact_time_strata"]["episode_without_contact"]["frame_samples"] == 270
    assert result["overlapping_outcome_strata"]["high_quality"]["mean_credit"] is None


@pytest.mark.parametrize(
    "values",
    [np.ones((1, 269)), np.ones((1, 270), dtype=np.float32), np.full((1, 270), np.nan)],
)
def test_incomplete_or_nonfinite_credit_rejected(values):
    with pytest.raises(ValueError, match="float64"):
        sampling_credit_inventory([record()], values, expected_episodes=1)


@pytest.mark.parametrize("groups,expected", [([0], 2), ([0, 0], 2), ([1, 0], 2)])
def test_no_failure_dropping_duplication_or_reordering(groups, expected):
    with pytest.raises(ValueError, match="complete ordered"):
        sampling_credit_inventory(
            [record(g) for g in groups], np.ones((len(groups), 270)), expected_episodes=expected
        )


def test_contradictory_physical_quality_rejected():
    row = record()
    row["outcome"]["high_quality"] = False
    with pytest.raises(ValueError, match="contradict"):
        sampling_credit_inventory([row], np.zeros((1, 270)), expected_episodes=1)


def test_overflow_rejected_not_emitted_as_json_infinity():
    with pytest.raises(ValueError, match="finite credit totals"):
        sampling_credit_inventory([record()], np.full((1, 270), 1e308), expected_episodes=1)
