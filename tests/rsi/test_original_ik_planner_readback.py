"""Readback metadata and physical parity guards, not new physical successes."""

import copy
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_original_ik_planner_pilot import intent_for, prefix_for
from scripts.rsi_read_original_ik_planner import require_complete
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_original_ik_planner_scope import receipts


def fixture():
    scope, rows = receipts()
    c = SimpleNamespace(
        cfg=dict(mode="SIM_ONLY"),
        scope=scope,
        candidate={"fixed": True},
        teachers={},
        signed_predecessors={"negative": "sealed", "positive": "sealed"},
        p=SimpleNamespace(PINS={}, HEADS={}),
    )
    declaration = dict(
        c.cfg,
        schema="soccer.rsi.original_ik_planner_six_declaration.v1",
        scope=scope,
        candidate=c.candidate,
        reused_teachers={},
        signed_predecessor_reviews=c.signed_predecessors,
        prepared_pins={},
        source_heads={},
    )
    declaration["report_hash"] = hash_json(declaration)
    complete = dict(
        schema="soccer.rsi.original_ik_planner_six_complete.v1",
        declaration_hash=declaration["report_hash"],
        results=rows,
        new_native_bodies=6,
        new_independent_cpu_bodies=6,
        native_substeps=18000,
        independent_cpu_substeps=18000,
        new_optimizer_updates=0,
        independent_terminal_readback_required=True,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    return c, declaration, complete


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "partial",
        "unbound",
        "reference_scope",
        "truthy",
        "optimized",
        "teacher",
        "missing_signed",
        "config",
    ],
)
def test_whole_native_and_independent_body_counts_and_intent_are_mandatory(mutation):
    c, d, complete = fixture()
    if mutation == "partial":
        complete["results"].pop()
    elif mutation == "unbound":
        complete["declaration_hash"] = "wrong"
    elif mutation == "reference_scope":
        d["scope"] = {}
    elif mutation == "truthy":
        complete["new_native_bodies"] = 6.0
    elif mutation == "optimized":
        complete["new_optimizer_updates"] = 128
    elif mutation == "teacher":
        complete["teacher_qualified"] = True
    elif mutation == "missing_signed":
        d["signed_predecessor_reviews"] = {}
    elif mutation == "config":
        d["mode"] = "REAL"
    if mutation is None:
        require_complete(c, d, complete)
    else:
        with pytest.raises(ValueError):
            require_complete(c, d, complete)


def test_ik_intent_is_distinct_from_reference_model_and_binds_before_body():
    scope, _ = receipts()
    intent = intent_for(scope["jobs"][1], "sha256:" + "a" * 64)
    assert intent["leg_reference_kind"] == "ORIGINAL_BOUNDED_IK"
    assert not any("model_hash" in k for k in intent)
    assert intent["report_hash"] == hash_json(
        {k: v for k, v in intent.items() if k != "report_hash"}
    )
    assert intent != intent_for(scope["jobs"][0], "sha256:" + "a" * 64)


def test_prefix_uses_authenticated_original_ik_teacher_not_nav_zero(tmp_path):
    scope, _ = receipts()
    t = {f"field_{i}": np.zeros((300, 1, i + 1)) for i in range(33)}
    path = tmp_path / "teacher.npz"
    np.savez_compressed(path, **t)
    teacher = dict(
        receipt_hash="teacher-receipt",
        witnesses=dict(trace=dict(path=str(path), sha256=digest(path))),
    )
    pins = {str(path): digest(path)}
    before = copy.deepcopy(t)
    result = prefix_for(scope["jobs"][0], t, teacher, pins)
    assert result["compared_frames"] == 300
    assert result["original_ik_teacher_receipt_hash"] == "teacher-receipt"
    t["field_0"][100] = 1.0
    assert prefix_for(scope["jobs"][1], t, teacher, pins)["compared_frames"] == 100
    with pytest.raises(ValueError):
        prefix_for(scope["jobs"][0], t, teacher, pins)
    pins[str(path)] = "wrong"
    with pytest.raises(ValueError):
        prefix_for(scope["jobs"][1], before, teacher, pins)
