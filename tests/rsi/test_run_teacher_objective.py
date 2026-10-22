import copy

import pytest

from rosclaw_soccer.rsi.run_teacher_objective import run_teacher_rank


def outcome(progress=4.0, lateral=1.5, forbidden=True, stable=True, quality=False):
    return dict(
        result=dict(
            body_failed=False,
            forbidden=forbidden,
            postcontact=dict(
                post_contact_quality_observed=True,
                correct_foot_first=True,
                other_foot_contact_in_window=False,
                goal_progress_1p2s_m=progress,
                goal_lateral_1p2s_m=lateral,
            ),
        ),
        stand=dict(healthy=True, stability_observed=stable),
        healthy_quality=quality,
    )


def test_zero_direction_error_does_not_reward_no_kick():
    assert run_teacher_rank(outcome()) > run_teacher_rank(
        outcome(
            progress=-0.046,
            lateral=0.128,
            forbidden=False,
        )
    )


def test_unstable_lucky_contact_not_useful():
    assert run_teacher_rank(outcome()) > run_teacher_rank(
        outcome(
            progress=5,
            lateral=0.001,
            stable=False,
            forbidden=False,
        )
    )


def test_true_success_dominates_high_speed_failed_shot():
    assert run_teacher_rank(outcome(quality=True, lateral=0.2, forbidden=False)) > run_teacher_rank(
        outcome(progress=8)
    )


def test_precision_comparison_only_after_usefulness():
    assert run_teacher_rank(outcome(lateral=0.2)) > run_teacher_rank(outcome(lateral=1.5))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), None])
def test_observed_malformed_displacement_rejected(value):
    with pytest.raises(ValueError):
        run_teacher_rank(outcome(progress=value))


def test_censored_contact_is_not_a_zero_error_success():
    row = outcome()
    row["result"]["postcontact"].update(
        post_contact_quality_observed=False, goal_progress_1p2s_m=None, goal_lateral_1p2s_m=None
    )
    before = copy.deepcopy(row)
    assert run_teacher_rank(row)[1] == 0
    assert row == before
