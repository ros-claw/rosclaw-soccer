"""Fixed curriculum/paired counts; synthetic labels never imply physical gain."""

import json

import pytest

from rosclaw_soccer.rsi.foot_heading_scope import (
    PROFILE,
    assess_heading_scope,
    heading_scope,
    require_heading_scope,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_reference_student_exam import load_inputs
from tests.rsi.test_first_touch_stage_diagnostics import outcome
from tests.rsi.test_planner_navigation_pulse_scope import config as navigation_config


def config():
    cfg = navigation_config()
    cfg.pop("pulse_group_order")
    return dict(
        cfg,
        exam_profile=PROFILE,
        heading_group_order=[28, 76, 12],
        leg_reference_kind="ORIGINAL_BOUNDED_IK_WITH_HEADING_PROBE",
        original_ik_predecessor={},
    )


def receipts():
    s = heading_scope("sha256:" + "a" * 64)
    rows = [
        dict(j, outcome=outcome(None if j["group"] == 28 and not j["heading_enabled"] else 151))
        for j in s["jobs"]
    ]
    for r in rows:
        r["receipt_hash"] = hash_json(r)
    return s, rows


def test_one_factor_contact_heading_with_same_nonzero_approach_and_no_updates():
    require_heading_scope(config())
    s, rows = receipts()
    assert s["source_seeds"] == [20261078, 20261093, 20261077]
    assert [r["heading_enabled"] for r in rows] == [False, True] * 3
    assert [r["lateral_delta_m_s"] for r in rows] == [0.08] * 6
    assert s["off_original_array_count"] == 36
    assert s["on_pre_action_physical_prefix_frames"] == 30
    r = assess_heading_scope(s, rows)
    assert r["stages"]["original_m0_pairs"]["net_success_gain"] == 1
    assert r["all_failed_courses_retained"]
    assert not r["teacher_qualified"]
    assert assess_heading_scope(json.loads(json.dumps(s)), rows) == r


@pytest.mark.parametrize(
    "mutation", ["partial", "order", "wrong_group", "zero_pulse", "truthy", "seal", "scope"]
)
def test_partial_or_retagged_cohorts_cannot_stand_in_for_the_original_six(mutation):
    s, rows = receipts()
    if mutation == "partial":
        rows.pop()
    elif mutation == "order":
        rows.reverse()
    elif mutation == "wrong_group":
        rows[0]["group"] = 92
    elif mutation == "zero_pulse":
        rows[0]["lateral_delta_m_s"] = 0.0
    elif mutation == "truthy":
        rows[1]["heading_enabled"] = 1
        rows[1]["receipt_hash"] = hash_json(
            {k: v for k, v in rows[1].items() if k != "receipt_hash"}
        )
    elif mutation == "seal":
        rows[0]["receipt_hash"] = "wrong"
    else:
        s["new_optimizer_updates"] = 128
    with pytest.raises(ValueError):
        assess_heading_scope(s, rows)


@pytest.mark.parametrize(
    "key,value",
    [
        ("original_ik_predecessor", None),
        ("max_attempts", 7),
        ("neural_training_authorized", True),
        ("new_optimizer_updates", True),
        ("signed_predecessors", {}),
        ("matched_ablation", {}),
        ("heading_group_order", [28, 76, 92]),
        ("combined_body_substep_budget", 18000),
    ],
)
def test_new_mode_keeps_strict_budgets_and_no_model_or_fresh_mixing(key, value):
    c = config()
    c[key] = value
    with pytest.raises(ValueError):
        require_heading_scope(c)


@pytest.mark.parametrize("mode", [False, True, 1])
def test_dedicated_loader_cannot_bypass_frozen_source_validation(tmp_path, mode):
    c = dict(
        config(),
        source_root=str(tmp_path),
        core_root=str(tmp_path),
        source_head="not-this-source",
        output_root=str(tmp_path / "unused"),
    )
    p = tmp_path / "config.json"
    p.write_text(json.dumps(c))
    with pytest.raises(ValueError):
        load_inputs(p, heading_research=mode)
    assert not (tmp_path / "unused").exists()


@pytest.mark.parametrize("other", ["pulse_research", "original_ik_research"])
def test_heading_research_is_not_an_alias_for_an_old_zero_or_original_ik_exam(tmp_path, other):
    p = tmp_path / "config.json"
    p.write_text(json.dumps(config()))
    with pytest.raises(ValueError, match="explicit"):
        load_inputs(p, heading_research=True, **{other: True})
