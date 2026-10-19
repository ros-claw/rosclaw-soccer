"""Fixed producer/readback metadata, not executed C3 physics."""

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_foot_heading_pilot as pilot
from scripts.rsi_foot_heading_pilot import intent_for, prefix_for
from scripts.rsi_read_foot_heading import require_complete
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_foot_heading_parity import baseline
from tests.rsi.test_foot_heading_scope import receipts
from tests.rsi.test_original_ik_planner_scope import receipts as ik_receipts


def fixture():
    scope, rows = receipts()
    c = SimpleNamespace(
        cfg=dict(mode="SIM_ONLY"),
        scope=scope,
        candidate={"fixed": True},
        heading_baselines={},
        predecessor_summary={"whole": True},
        p=SimpleNamespace(PINS={}, HEADS={}),
    )
    declaration = dict(
        c.cfg,
        schema="soccer.rsi.foot_heading_six_declaration.v1",
        scope=scope,
        candidate=c.candidate,
        reused_positive_ik_baselines={},
        original_ik_predecessor_review=c.predecessor_summary,
        prepared_pins={},
        source_heads={},
    )
    declaration["report_hash"] = hash_json(declaration)
    complete = dict(
        schema="soccer.rsi.foot_heading_six_complete.v1",
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
        "scope",
        "truthy",
        "optimized",
        "teacher",
        "predecessor",
        "config",
        "baseline",
    ],
)
def test_readback_rejects_partial_or_retagged_declared_study(mutation):
    c, d, complete = fixture()
    if mutation == "partial":
        complete["results"].pop()
    elif mutation == "unbound":
        complete["declaration_hash"] = "wrong"
    elif mutation == "scope":
        d["scope"] = {}
    elif mutation == "truthy":
        complete["new_native_bodies"] = 6.0
    elif mutation == "optimized":
        complete["new_optimizer_updates"] = 128
    elif mutation == "teacher":
        complete["teacher_qualified"] = True
    elif mutation == "predecessor":
        d["original_ik_predecessor_review"] = {}
    elif mutation == "config":
        d["mode"] = "REAL"
    elif mutation == "baseline":
        d["reused_positive_ik_baselines"] = {"28": "unbound"}
    if mutation is None:
        require_complete(c, d, complete)
    else:
        with pytest.raises(ValueError):
            require_complete(c, d, complete)


def test_intent_binds_heading_and_positive_pulse_before_body():
    scope, _ = receipts()
    off, on = (intent_for(j, "declaration") for j in scope["jobs"][:2])
    assert off["pulse_contract_hash"] == on["pulse_contract_hash"]
    assert off["heading_contract_hash"] != on["heading_contract_hash"]
    assert off["leg_reference_kind"] == "ORIGINAL_BOUNDED_IK_WITH_HEADING_PROBE"
    assert not any("model_hash" in k for k in on)


def test_prefix_uses_authenticated_current_c2_positive_not_t33(tmp_path):
    scope, _ = receipts()
    t = baseline()
    path = tmp_path / "c2-positive.npz"
    np.savez_compressed(path, **t)
    source = dict(
        receipt_hash="c2",
        role="ik_pulse",
        group=28,
        lateral_delta_m_s=0.08,
        witnesses=dict(trace=dict(path=str(path), sha256=digest(path))),
    )
    pins = {str(path): digest(path)}
    r = prefix_for(scope["jobs"][0], t, source, pins)
    assert r["compared_frames"] == 300
    assert r["c2_positive_receipt_hash"] == "c2"
    t["joint_target_rad"][30] = 0.1
    assert prefix_for(scope["jobs"][1], t, source, pins)["compared_frames"] == 30
    with pytest.raises(ValueError):
        prefix_for(scope["jobs"][0], t, source, pins)
    pins[str(path)] = "wrong"
    with pytest.raises(ValueError):
        prefix_for(scope["jobs"][1], t, source, pins)


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "wrongworld",
        "wrongparent",
        "wrongcore",
        "b2",
        "plan",
        "seeds",
        "partial",
        "zeropulse",
        "closed",
        "source",
    ],
)
def test_prepare_requires_whole_same_c2_positive_and_explicit_both_requests(
    tmp_path, monkeypatch, mutation
):
    scope, _ = receipts()
    old_scope, old_rows = ik_receipts()
    parent = scope["source_plan"]["behavior_model_hash"]
    c = SimpleNamespace(
        source=tmp_path,
        core=tmp_path / "core",
        root=tmp_path / "unused",
        pins={},
        sealed=None,
        cfg=dict(
            original_ik_predecessor={},
            heading_group_order=[28, 76, 12],
            mesh_task_contact_world_hash="world",
        ),
        assets=SimpleNamespace(MEAN=parent),
        scope=scope,
        candidate=dict(parameters={}),
    )
    cohort = dict(
        producer=dict(
            mesh_task_contact_world_hash="world", behavior_model_hash=parent, core_root=str(c.core)
        ),
        declaration=dict(candidate=c.candidate, scope=old_scope),
        complete=dict(report_hash="complete", results=old_rows),
        audit=dict(report_hash="audit"),
    )
    previous = dict(
        cohort=cohort, contact=dict(report_hash="contact"), stage=dict(report_hash="stage")
    )
    calls = []

    def load(path, **kwargs):
        assert kwargs == {"heading_research": True}
        return c

    monkeypatch.setattr(pilot, "load_inputs", load)
    monkeypatch.setattr(pilot, "__file__", str(tmp_path / "scripts/rsi_foot_heading_pilot.py"))
    monkeypatch.setattr(pilot, "load_heading_predecessor", lambda *a, **kw: previous)
    original_request = pilot.require_heading_request

    def request(probe, **kwargs):
        calls.append(probe.enabled)
        original_request(probe, **kwargs)

    monkeypatch.setattr(pilot, "require_heading_request", request)
    if mutation in ("wrongworld", "wrongparent", "wrongcore"):
        cohort["producer"][
            {
                "wrongworld": "mesh_task_contact_world_hash",
                "wrongparent": "behavior_model_hash",
                "wrongcore": "core_root",
            }[mutation]
        ] = "wrong"
    elif mutation == "b2":
        cohort["declaration"]["candidate"] = {}
    elif mutation == "plan":
        cohort["declaration"]["scope"]["source_plan"] = {}
    elif mutation == "seeds":
        cohort["declaration"]["scope"]["source_seeds"] = []
    elif mutation == "partial":
        cohort["complete"]["results"].pop()
    elif mutation == "zeropulse":
        old_rows[1]["lateral_delta_m_s"] = 0.0
    elif mutation == "closed":
        c.root.mkdir()
    elif mutation == "source":
        monkeypatch.setattr(pilot, "__file__", str(tmp_path / "wrong.py"))
    if mutation is None:
        assert pilot.prepare(tmp_path / "config") is c
        assert calls == [False, True]
        assert list(c.heading_baselines) == [28, 76, 12]
        assert c.predecessor_summary["teacher_qualified"] is False
    else:
        with pytest.raises(ValueError):
            pilot.prepare(tmp_path / "config")
        assert calls == []
