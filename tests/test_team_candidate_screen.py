import pytest

from rosclaw_soccer.training.team_candidate_screen import TeamScreenArm, screen_team_candidate
from rosclaw_soccer.training.team_episode_cut import TeamEpisodeCut


def arm(safety=None, exit_frame=None, latch=None, windows=0, observed=5000):
    terminal = min((x for x in (safety, exit_frame) if x is not None), default=None)
    return TeamScreenArm(
        TeamEpisodeCut(safety, exit_frame, terminal), latch, observed, 5000, windows
    )


def test_case21_sampled_safety_is_not_hidden_by_clean_substep_latch():
    screen = screen_team_candidate(arm(exit_frame=482), arm(safety=159, exit_frame=482))
    assert screen.sampled_safety_regression
    assert not screen.substep_latch_regression
    assert not screen.eligible_for_fresh_validation
    assert not screen.promotion_authorized


def test_no_capture_improvement_is_not_a_success():
    assert not screen_team_candidate(arm(), arm()).eligible_for_fresh_validation


def test_substep_latch_regression_blocks_even_with_gain():
    screen = screen_team_candidate(arm(), arm(latch=100, windows=1))
    assert screen.substep_latch_regression and not screen.eligible_for_fresh_validation


def test_post_exit_sampled_failure_is_censored_but_gain_never_authorizes_promotion():
    screen = screen_team_candidate(arm(exit_frame=100), arm(safety=101, exit_frame=100, windows=1))
    assert not screen.sampled_safety_regression
    assert screen.eligible_for_fresh_validation and not screen.promotion_authorized


def test_earlier_sampled_failure_blocks():
    assert screen_team_candidate(
        arm(safety=200), arm(safety=199, windows=1)
    ).sampled_safety_regression


def test_missing_or_malformed_evidence_fails_closed():
    with pytest.raises(ValueError):
        arm(observed=4999)
    with pytest.raises(ValueError):
        arm(latch=True)
    with pytest.raises(ValueError):
        TeamScreenArm(TeamEpisodeCut(5, None, None), None, 5000, 5000, 0)
    with pytest.raises(ValueError):
        screen_team_candidate(
            arm(), TeamScreenArm(TeamEpisodeCut(None, None, None), None, 10, 10, 1)
        )
