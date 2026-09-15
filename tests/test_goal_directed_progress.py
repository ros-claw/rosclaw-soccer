import numpy as np
import pytest

from rosclaw_soccer.training.goal_directed_progress import goal_directed_progress


def test_approach_and_retreat_are_signed_not_positive_only_activity():
    positions = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.0], [0.0, 0.0]])
    targets = np.tile([2.0, 0.0], (3, 1))
    before = positions.copy()
    result = goal_directed_progress(
        positions=positions, transition_targets=targets, valid_transitions=np.ones(3, dtype=bool)
    )
    np.testing.assert_allclose(result, [1.0, -0.5, -0.5])
    assert abs(result.sum()) < 1e-12
    assert np.linalg.norm(np.diff(positions, axis=0), axis=1).sum() == 2.0
    assert not result.flags.writeable
    np.testing.assert_array_equal(positions, before)


def test_moving_target_cannot_reward_a_stationary_body():
    result = goal_directed_progress(
        positions=np.zeros((4, 3)),
        transition_targets=np.array([[5.0, 0.0, 1.0], [2.0, 4.0, 0.0], [0.0, 0.0, 0.0]]),
        valid_transitions=np.ones(3, dtype=bool),
    )
    np.testing.assert_array_equal(result, np.zeros(3))


def test_excluded_transition_is_missing_not_false_success():
    result = goal_directed_progress(
        positions=np.array([[0.0], [1.0], [2.0]]),
        transition_targets=np.full((2, 1), 3.0),
        valid_transitions=np.array([False, True]),
    )
    assert np.isnan(result[0]) and result[1] == 1.0


def test_changing_targets_do_not_establish_useful_tactics_or_closed_path_cancellation():
    positions = np.array([[0.0], [1.0], [0.0]])
    result = goal_directed_progress(
        positions=positions,
        transition_targets=positions[1:].copy(),
        valid_transitions=np.ones(2, dtype=bool),
    )
    # Target provenance matters: chasing targets can still credit a closed path.
    np.testing.assert_array_equal(result, [1.0, 1.0])


@pytest.mark.parametrize(
    "fault", ["nan", "int", "rank", "target_count", "mask_int", "mask_count", "dimensions", "empty"]
)
def test_invalid_or_ambiguous_alignment_rejected(fault):
    positions = np.zeros((3, 2))
    targets = np.ones((2, 2))
    mask = np.ones(2, dtype=bool)
    if fault == "nan":
        positions[0, 0] = np.nan
    elif fault == "int":
        positions = positions.astype(int)
    elif fault == "rank":
        positions = positions[0]
    elif fault == "target_count":
        targets = targets[:1]
    elif fault == "mask_int":
        mask = mask.astype(int)
    elif fault == "mask_count":
        mask = mask[:1]
    elif fault == "dimensions":
        positions, targets = np.zeros((3, 4)), np.ones((2, 4))
    else:
        positions, targets, mask = positions[:1], targets[:0], mask[:0]
    with pytest.raises(ValueError):
        goal_directed_progress(
            positions=positions, transition_targets=targets, valid_transitions=mask
        )
