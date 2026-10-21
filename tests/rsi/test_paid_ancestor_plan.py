import hashlib
import json

import pytest
from test_paid_ancestor_study import sources

from rosclaw_soccer.rsi.paid_ancestor_study import (
    SCHEDULES,
    ancestor_train_cases,
    case_condition,
    load_ancestor_plan,
)
from rosclaw_soccer.rsi.paid_fresh_protocol import fresh_lifecycle_cases
from rosclaw_soccer.sim.contracts import hash_json


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def fixture():
    original = sources()
    d = dict(
        schema="paid.ancestor_counterfactual.TRAIN20.v1",
        source_head="head",
        cases=ancestor_train_cases(original),
        repeats=2,
        shards=4,
        cases_per_shard=5,
        schedules={k: list(v) for k, v in SCHEDULES.items()},
        bank="train",
        Fresh_fit_rows=0,
        hardware_authorized=False,
        promotion_authorized=False,
        excluded_context_hashes=[
            hash_json(case_condition(c)) for c in fresh_lifecycle_cases(original)
        ],
    )
    r = dict(
        status="PAID_TRAIN20_ANCESTOR_STUDY_PREREGISTERED_NO_ROLLOUT",
        source_head="head",
        cases_preflighted=20,
        own_native_steps=0,
        rollouts=0,
        bank="train",
        Fresh_fit_rows=0,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return d, r


def publish(root, d, r):
    dp = root / "declaration.json"
    dp.write_text(json.dumps(d))
    r["declaration_hash"] = digest(dp)
    rp = root / "complete.json"
    rp.write_text(json.dumps(r))
    return digest(rp)


def test_exact_new_train_contract_can_be_read_without_allocating_physics(tmp_path):
    d, r = fixture()
    h = publish(tmp_path, d, r)
    assert load_ancestor_plan(tmp_path, h) == (r, d)


@pytest.mark.parametrize(
    "fault",
    ["schedule", "missing", "test_overlap", "bank", "mass", "velocity", "preflight", "hardware"],
)
def test_changed_or_unqualified_plan_cannot_unlock_shard_execution(tmp_path, fault):
    d, r = fixture()
    if fault == "schedule":
        d["schedules"]["ancestor_third"][-1] = "student"
    elif fault == "missing":
        d["cases"].pop()
    elif fault == "test_overlap":
        d["excluded_context_hashes"][0] = hash_json(case_condition(d["cases"][0]))
    elif fault == "bank":
        d["bank"] = "fresh"
    elif fault == "mass":
        d["cases"][0]["ball_mass_kg"] = 1
    elif fault == "velocity":
        d["cases"][0]["velocity_world_initial"] = [9, 9]
    elif fault == "preflight":
        r["cases_preflighted"] = 19
    else:
        d["hardware_authorized"] = True
    h = publish(tmp_path, d, r)
    with pytest.raises(ValueError):
        load_ancestor_plan(tmp_path, h)
