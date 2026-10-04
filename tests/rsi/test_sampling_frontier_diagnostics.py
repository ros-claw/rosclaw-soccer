import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.sampling_frontier_diagnostics import contact_timeline, sampling_frontier


def record(group=0, context=0):
    return dict(
        group=group,
        baseline_course_index=context,
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


def test_partial_union_is_only_offline_donor_index_not_policy_success():
    result = sampling_frontier([record(1, 2), record(3, 2)], expected_episodes=8)
    assert not result["complete"]
    assert result["observed_contexts"] == result["observed_successful_contexts"] == 1
    assert result["contexts"][0]["successful_donor_groups"] == [1, 3]
    for field in (
        "partial_results_extrapolated",
        "source_physics_validated_here",
        "runtime_selection_authorized",
        "training_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert result[field] is False


def test_overlapping_failures_are_retained_without_changing_rewards():
    row = record()
    row["outcome"].update(
        high_quality=False,
        minimum_pelvis_z_m=0.5,
        maximum_lateral_excursion_m=5.0,
        forward_60_m=0.5,
        lateral_60_m=0.8,
    )
    original = copy.deepcopy(row)
    result = sampling_frontier([row], expected_episodes=1)
    assert result["complete"]
    assert result["overlapping_failure_counts"] == dict(
        unsafe_pelvis=1, out_of_play=1, insufficient_forward=1, direction_error=1
    )
    assert result["contexts"][0]["maximum_observed_terminal_return"] == -117
    assert row == original


@pytest.mark.parametrize("field,value", [("high_quality", False), ("clean_foot_only", False)])
def test_contradictory_success_labels_rejected(field, value):
    row = record()
    row["outcome"][field] = value
    with pytest.raises(ValueError, match="contradict"):
        sampling_frontier([row], expected_episodes=1)


def test_duplicate_episode_does_not_inflate_frontier():
    with pytest.raises(ValueError, match="unique"):
        sampling_frontier([record(), record()], expected_episodes=2)


def test_safety_is_separate_from_original_football_quality_definition():
    row = record()
    row["outcome"]["minimum_pelvis_z_m"] = 0.5
    result = sampling_frontier([row], expected_episodes=1)
    assert result["observed_high_quality_episodes"] == 1
    assert result["overlapping_failure_counts"] == {"unsafe_pelvis": 1}


def test_missing_contact_is_separate_from_wrong_body_contact():
    a, b = record(0), record(1)
    a["outcome"].update(high_quality=False, clean_foot_only=False, first_contact_frame=None)
    b["outcome"].update(high_quality=False, clean_foot_only=False, contact_body_indices=[2])
    result = sampling_frontier([a, b], expected_episodes=2)
    assert result["overlapping_failure_counts"] == {"no_contact": 1, "nonfoot_contact": 1}


@pytest.mark.parametrize("delay", [0, 1, 20, 35])
def test_nonfoot_first_and_postkick_recollision_are_distinct(delay):
    forces = np.zeros((300, 6))
    forces[65, 0] = 2
    forces[65 + delay, 4] = 2
    outcome = record()["outcome"]
    outcome.update(clean_foot_only=False, contact_body_indices=[0, 4])
    before = forces.copy()
    result = contact_timeline(forces, outcome)
    assert result["kind"] == ("first_event_nonfoot" if delay == 0 else "foot_first_then_nonfoot")
    assert result["secondary_nonfoot_lag_frames"] == (delay or None)
    assert result["runtime_selection_authorized"] is False
    assert np.array_equal(before, forces)


def test_contact_force_threshold_matches_original_strict_greater_than_one():
    forces = np.ones((300, 6))
    outcome = record()["outcome"]
    outcome.update(first_contact_frame=None, clean_foot_only=False, contact_body_indices=[])
    assert contact_timeline(forces, outcome)["kind"] == "no_contact"
    outcome["first_contact_frame"] = 0
    with pytest.raises(ValueError, match="contradict"):
        contact_timeline(forces, outcome)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1])
def test_invalid_forces_rejected(value):
    forces = np.zeros((300, 6))
    forces[65, 0] = value
    with pytest.raises(ValueError, match="finite"):
        contact_timeline(forces, record()["outcome"])
