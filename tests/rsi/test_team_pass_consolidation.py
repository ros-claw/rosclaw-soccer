import copy

import pytest

from rosclaw_soccer.rsi.team_pass_consolidation import RETAINED, pass_config, pass_pools


def rows():
    records = [
        dict(index=i, role="retained_neural17", source_context=c, bc_weight=1, task_success=True)
        for i, c in enumerate(RETAINED)
    ]
    for i in range(160):
        positive = i % 5 == 1 or i % 7 == 0
        records.append(
            dict(
                index=len(records),
                role="new_teacher" if i % 2 else "new_unassisted",
                bc_weight=int(positive),
                task_success=positive,
            )
        )
    return records


def test_all17_anchors_and_every_new_negative_stay_in_bank_but_not_imitation():
    records = rows()
    old, teachers, new = pass_pools(records)
    assert old == list(range(17)) and teachers and new
    assert len(records) == 177 and len(old) + len(teachers) + len(new) < 177
    assert all(records[i]["task_success"] for i in (*old, *teachers, *new))


@pytest.mark.parametrize(
    "fault", ["partial", "duplicate", "old_loss", "context", "role", "flag", "negative_bc"]
)
def test_invalid_or_lossy_bank_not_silently_reduced(fault):
    records = rows()
    if fault == "partial":
        records.pop()
    elif fault == "duplicate":
        records[1] = copy.deepcopy(records[0])
    elif fault == "old_loss":
        records[0].update(task_success=False, bc_weight=0)
    elif fault == "context":
        records[0]["source_context"] = 999
    elif fault == "role":
        records[-1]["role"] = "invented_positive"
    elif fault == "flag":
        records[0]["task_success"] = 1
    else:
        r = next(r for r in records if not r["task_success"])
        r["bc_weight"] = 1
    with pytest.raises(ValueError):
        pass_pools(records)


def test_fixed_three_seed_bc_phase_carries_monotone_actor_and_frozen_critic():
    for seed in (75023, 75024, 75025):
        c = pass_config(seed, "cpu")
        assert c["steps"] == 16 and c["final_actor_steps"] - c["initial_actor_steps"] == 16
        assert c["initial_actor_steps"] == 112 and c["critic_steps"] == 48
        assert c["PPO_steps_this_stage"] == 0 and c["final_only"]
    for seed, device in ((65023, "cpu"), (True, "cpu"), (75023, "cuda:3")):
        with pytest.raises(ValueError):
            pass_config(seed, device)
