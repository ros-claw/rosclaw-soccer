import pytest
from test_single_mode_pass_curriculum import records

from rosclaw_soccer.rsi.single_mode_pass_curriculum import single_mode_pass_curriculum
from rosclaw_soccer.rsi.team_pass_single_mode import selected_config, selected_schedule


def test_same_bounded_budget_but_no_conflicting_positive_mode_imitation():
    rows = records()
    plan = single_mode_pass_curriculum(rows)
    for seed in (95023, 95024, 95025):
        c = selected_config(seed, "cpu")
        assert (
            c["steps"] == 64 and c["initial_actor_steps"] == 128 and c["final_actor_steps"] == 192
        )
        assert c["PPO_steps_this_stage"] == 0 and c["raw_positive_not_automatically_fit"]
        schedule = selected_schedule(rows, seed)
        assert schedule == selected_schedule(rows, seed)
        fit = {j for r in schedule for j in r["teacher_indices"] + r["anchor_indices"]} | {
            r["new_anchor_index"] for r in schedule
        }
        assert fit == set(plan["fit_indices"])
        assert not fit & set(plan["unselected_positive_indices"])
    assert selected_schedule(rows, 95023) != selected_schedule(rows, 95024)
    with pytest.raises(ValueError):
        selected_config(85023, "cpu")
