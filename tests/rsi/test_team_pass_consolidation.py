import copy
import hashlib
import json

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


def test_new_bank_keeps_original_retention_exam_via_bound_inherited_lineage(tmp_path):
    from rosclaw_soccer.rsi.contact_consolidation_data import (
        CORRECTION_BANK_STATUS,
        context_witness_hash,
    )
    from rosclaw_soccer.rsi.team_pass_consolidation import BANK_STATUS

    def save(name, value):
        folder = tmp_path / name
        folder.mkdir()
        path = folder / "complete.json"
        path.write_text(json.dumps(value))
        return path, "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()

    old, old_hash = save(
        "old-bank", dict(status=CORRECTION_BANK_STATUS, context_witness_hash="sha256:original")
    )
    warm, warm_hash = save("warm", dict(data_hash=old_hash, parent_lineage_hash="sha256:parent"))
    data = dict(
        status=BANK_STATUS,
        warm_start_hash=warm_hash,
        parent_lineage_hash="sha256:parent",
        input_file_hashes={str(old): old_hash, str(warm): warm_hash},
    )
    assert context_witness_hash(data, "team-pass-consolidated-v1") == "sha256:original"
    from rosclaw_soccer.rsi.team_pass_correction import (
        CORRECTION_BANK_STATUS as PASS_CORRECTION_BANK,
    )

    original_pass, original_pass_hash = save("original-pass", data)
    pass_warm, pass_warm_hash = save(
        "pass-warm", dict(data_hash=original_pass_hash, parent_lineage_hash="sha256:parent")
    )
    new_data = dict(
        status=PASS_CORRECTION_BANK,
        base_data_hash=original_pass_hash,
        warm_start_hash=pass_warm_hash,
        parent_lineage_hash="sha256:parent",
        input_file_hashes={str(original_pass): original_pass_hash, str(pass_warm): pass_warm_hash},
    )
    assert context_witness_hash(new_data, "team-pass-corrected-v2") == "sha256:original"
    warm.write_text("{}")
    with pytest.raises(ValueError, match="lineage"):
        context_witness_hash(data, "team-pass-consolidated-v1")
