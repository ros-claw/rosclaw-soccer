from rosclaw_soccer.training.motion_portfolio_holdout import motion_portfolio_holdout_courses
from scripts.rsi_diagnose_motion_expert_coverage import diagnostic_indices


def test_diagnostic_is_fixed_stratified_not_success_filtered():
    indices = diagnostic_indices()
    courses = motion_portfolio_holdout_courses()
    assert len(indices) == len(set(indices)) == 24
    assert all(i in range(192) for i in indices)
    assert len({courses[i].stratum for i in indices}) == 6
    for base in range(0, 192, 32):
        assert [i - base for i in indices if base <= i < base + 32] == [0, 8, 16, 24]
