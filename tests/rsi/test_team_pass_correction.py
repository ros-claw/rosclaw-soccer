import copy

import pytest
from test_team_pass_consolidation import rows

from rosclaw_soccer.rsi.team_pass_consolidation import RETAINED
from rosclaw_soccer.rsi.team_pass_correction import (
    correction_config,
    correction_pools,
    correction_schedule,
)


def bank():
    result = rows()
    for c in RETAINED:
        result.append(
            dict(
                index=len(result),
                role="current_acquired17",
                source_context=c,
                task_success=True,
                bc_weight=1,
            )
        )
    for state in range(40):
        for task in (0, 1):
            positive = (state + task) % 2 == 0
            result.append(
                dict(
                    index=len(result),
                    role="own_state_expert",
                    task_success=positive,
                    bc_weight=int(positive),
                    source_course=dict(state_id=state, task=task, assisted=True),
                )
            )
    return result


def test_all_history_coverage_and_no_negative_imitation_or_counter_reset():
    data = bank()
    pools = correction_pools(data)
    assert len(pools[0]) == 34 and len(data) == 274
    for seed in (85023, 85024, 85025):
        config = correction_config(seed, "cpu")
        assert config["final_actor_steps"] - config["initial_actor_steps"] == 64
        assert config["initial_actor_steps"] == 128 and config["critic_steps"] == 48
        assert config["PPO_steps_this_stage"] == 0
        schedule = correction_schedule(data, seed)
        assert schedule == correction_schedule(data, seed)
        assert len(schedule) == 64
        selected = [
            set(j for row in schedule for j in row[k])
            for k in ("anchor_indices", "teacher_indices")
        ]
        assert selected == [set(pools[0]), set(pools[1])]
        assert {r["new_anchor_index"] for r in schedule} == set(pools[2])
        assert all(data[i]["task_success"] for group in pools for i in group)
    assert correction_schedule(data, 85023) != correction_schedule(data, 85024)


@pytest.mark.parametrize(
    "fault", ["missing", "duplicate", "lost", "context", "label", "role", "task"]
)
def test_partial_or_mislabeled_correction_bank_rejects(fault):
    data = copy.deepcopy(bank())
    if fault == "missing":
        data.pop()
    elif fault == "duplicate":
        data[-1] = copy.deepcopy(data[-2])
    elif fault == "lost":
        data[177].update(task_success=False, bc_weight=0)
    elif fault == "context":
        data[177]["source_context"] = 0
    elif fault == "label":
        data[-1]["task_success"] = 1
    elif fault == "role":
        data[-1]["role"] = "advice_only"
    else:
        data[-1]["source_course"]["task"] = 9
    with pytest.raises(ValueError):
        correction_pools(data)
