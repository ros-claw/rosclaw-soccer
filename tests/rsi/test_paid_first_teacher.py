import pytest

from rosclaw_soccer.rsi.paid_first_teacher import first_teacher_config, first_teacher_rank


def row(good, reward=0.0):
    return dict(
        stand=dict(healthy=True, stability_observed=True),
        kick=dict(
            body_failed=False,
            forbidden=not good,
            terminal_reward=reward,
            postcontact=dict(
                correct_foot_first=True,
                post_contact_quality_observed=True,
                goal_progress_1p2s_m=2.0,
                goal_lateral_1p2s_m=0.0,
                other_foot_contact_in_window=False,
                maximum_pelvis_retreat_after_contact_m=0.0,
            ),
        ),
    )


def test_known_success_must_not_be_traded_for_new_success_or_higher_reward():
    protected = [False, True]
    assert first_teacher_rank([row(False), row(True)], protected) > first_teacher_rank(
        [row(True, 100), row(False, 100)], protected
    )
    assert first_teacher_rank([row(True), row(True)], protected) > first_teacher_rank(
        [row(False, 100), row(True, 100)], protected
    )


def test_stable_recovery_takes_precedence_over_a_clean_kick():
    bad = [row(True), row(True)]
    bad[0]["stand"]["healthy"] = False
    assert first_teacher_rank([row(False), row(True)], [False, True]) > first_teacher_rank(
        bad, [False, True]
    )


def test_pilot_has_fixed_seeds_pairs_and_finite_compute_budget():
    configs = [first_teacher_config(i) for i in range(3)]
    assert len({c["random_seed"] for c in configs}) == 3
    assert all(c["pairs"] == [[4, 5], [14, 15]] and c["max_episodes"] == 108 for c in configs)
    assert all(c["Fresh_rows_read"] == 0 and c["neural_optimizer_updates"] == 0 for c in configs)


@pytest.mark.parametrize("seed", [True, 3, -1, None])
def test_unregistered_teacher_seed_rejected(seed):
    with pytest.raises(ValueError):
        first_teacher_config(seed)
