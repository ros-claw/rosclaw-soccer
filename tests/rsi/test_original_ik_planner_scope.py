"""No selected failures or changed denominator may masquerade as progress."""

import copy
import json

import pytest

from rosclaw_soccer.rsi.original_ik_planner_scope import (
    PROFILE,
    assess_ik_scope,
    ik_scope,
    require_ik_scope,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_reference_student_exam import load_inputs
from tests.rsi.test_first_touch_stage_diagnostics import outcome
from tests.rsi.test_planner_navigation_pulse_scope import config as navigation_config


def config():
    return dict(
        navigation_config(),
        exam_profile=PROFILE,
        signed_predecessors=dict(negative={}, positive={}),
        leg_reference_kind="ORIGINAL_BOUNDED_IK",
        original_teacher_all300_off_parity_required=True,
    )


def receipts():
    scope = ik_scope("sha256:" + "a" * 64)
    rows = [dict(j, outcome=outcome(None if j["group"] == 28 else 151)) for j in scope["jobs"]]
    for r in rows:
        r["receipt_hash"] = hash_json(r)
    return scope, rows


def test_scope_uses_three_training_courses_original_ik_and_no_optimizer():
    require_ik_scope(config())
    scope, rows = receipts()
    assert scope["source_seeds"] == [20261078, 20261093, 20261077]
    assert scope["combined_body_substeps"] == 36000
    assert [r["lateral_delta_m_s"] for r in rows] == [0.0, 0.08] * 3
    result = assess_ik_scope(scope, rows)
    assert result["stages"]["original_m0_pairs"]["net_success_gain"] == 0
    assert result["all_failures_retained"]
    assert result["optimizer_updates"] == 0
    assert not result["teacher_qualified"]
    assert assess_ik_scope(json.loads(json.dumps(scope)), rows) == result


@pytest.mark.parametrize(
    "mutation", ["drop", "duplicate", "reorder", "change_group", "unsealed", "change_scope"]
)
def test_all_six_cases_and_failures_are_mandatory(mutation):
    scope, rows = receipts()
    if mutation == "drop":
        rows.pop(0)
    elif mutation == "duplicate":
        rows[0] = copy.deepcopy(rows[1])
    elif mutation == "reorder":
        rows.reverse()
    elif mutation == "change_group":
        rows[0]["group"] = 92
    elif mutation == "change_scope":
        scope["new_optimizer_updates"] = 1
    else:
        rows[0]["receipt_hash"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError):
        assess_ik_scope(scope, rows)


@pytest.mark.parametrize(
    "key,value",
    [
        ("signed_predecessors", {"positive": {}}),
        ("leg_reference_kind", "ZERO_GRU"),
        ("original_teacher_all300_off_parity_required", 1),
        ("negative_predecessor", {}),
        ("reference_student_model", None),
        ("new_physical_budget", 7),
        ("new_optimizer_updates", True),
        ("use_mesh_surface_swing", True),
    ],
)
def test_explicit_intent_cannot_widen_scope_or_mix_reference_models(key, value):
    cfg = config()
    cfg[key] = value
    with pytest.raises(ValueError):
        require_ik_scope(cfg)


@pytest.mark.parametrize("mode", [False, True, 1])
def test_dedicated_loader_cannot_bypass_immutable_source_validation(tmp_path, mode):
    cfg = dict(
        config(),
        source_root=str(tmp_path),
        core_root=str(tmp_path),
        output_root=str(tmp_path / "unused"),
        source_head="not-this-source",
    )
    p = tmp_path / "config.json"
    p.write_text(json.dumps(cfg))
    with pytest.raises(ValueError):
        load_inputs(p, original_ik_research=mode)
    assert not (tmp_path / "unused").exists()


def test_loader_cannot_select_both_reference_intents(tmp_path):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config()))
    with pytest.raises(ValueError, match="explicit"):
        load_inputs(p, pulse_research=True, original_ik_research=True)
