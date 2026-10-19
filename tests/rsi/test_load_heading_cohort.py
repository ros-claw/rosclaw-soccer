"""Mocked metadata authentication, never additional physical evidence."""

import copy
import json

import pytest

from rosclaw_soccer.rsi.foot_heading_scope import assess_heading_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_load_heading_cohort as module
from tests.rsi.test_foot_heading_scope import config, receipts


def fixture(tmp_path, monkeypatch):
    root, source = tmp_path / "root", tmp_path / "source"
    parent = "sha256:" + "a" * 64
    producer = dict(
        config(),
        source_root=str(source),
        output_root=str(root),
        source_head="head",
        behavior_model_hash=parent,
    )
    cp = tmp_path / "config.json"
    cp.write_text(json.dumps(producer))
    scope, rows = receipts()
    for r in rows:
        r["qualification_alignment"] = {"primitive_pass": r["outcome"]["task_pass"]}
        r["receipt_hash"] = hash_json({k: v for k, v in r.items() if k != "receipt_hash"})
    assessment = assess_heading_scope(scope, rows)
    d = dict(
        producer,
        schema="soccer.rsi.foot_heading_six_declaration.v1",
        scope=scope,
        input_file_hashes={},
        prepared_pins={},
        report_hash="declaration",
    )
    complete = dict(
        schema="soccer.rsi.foot_heading_six_complete.v1",
        declaration_hash="declaration",
        new_native_bodies=6,
        new_independent_cpu_bodies=6,
        native_substeps=18000,
        independent_cpu_substeps=18000,
        new_optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        results=rows,
        assessment=assessment,
        report_hash="complete",
    )
    audit = dict(
        schema="soccer.rsi.foot_heading_six_terminal_readback.v1",
        complete_hash="complete",
        reader_source_hash="file",
        report_hash="audit",
        source_native_bodies=6,
        source_independent_cpu_bodies=6,
        prior_actual_cpu_ik_and_heading_reconstructions_authenticated=6,
        new_physical_steps=0,
        new_foundation_inferences=0,
        optimizer_updates=0,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        ledger_tip="tip",
        input_file_hashes={},
        assessment=assessment,
        rows=[
            dict(
                job=j,
                receipt_hash=r["receipt_hash"],
                outcome=r["outcome"],
                alignment=r["qualification_alignment"],
            )
            for j, r in zip(scope["jobs"], rows, strict=True)
        ],
    )
    cfg = dict(
        producer_config=str(cp),
        producer_config_file_hash="file",
        producer_unit="native.service",
        audit_unit="audit.service",
        producer_declaration=str(root / "declaration.json"),
        producer_complete=str(root / "complete.json"),
        source_terminal_readback=str(tmp_path / "audit.json"),
        source_terminal_readback_hash="audit",
        audit_source=str(source / "scripts/rsi_read_foot_heading.py"),
    )
    documents = {
        cfg["producer_declaration"]: d,
        cfg["producer_complete"]: complete,
        cfg["source_terminal_readback"]: audit,
    }
    documents.update({str(root / f"receipt-{i:04d}.json"): r for i, r in enumerate(rows)})
    monkeypatch.setattr(module, "digest", lambda _: "file")
    monkeypatch.setattr(module, "require_terminal", lambda _: None)
    monkeypatch.setattr(
        module.subprocess,
        "check_output",
        lambda args, **kw: "head\n" if "rev-parse" in args else "",
    )
    state = dict(completed={r["job"]: copy.deepcopy(r) for r in rows})
    monkeypatch.setattr(module, "read_journal", lambda *a: ("tip", state))
    return cfg, d, complete, audit, lambda p, key="report_hash": documents[str(p)]


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "cfg_hash",
        "reader",
        "audit_hash",
        "order",
        "partial",
        "unsafe_claim",
        "counter",
        "truthy",
        "unmatched",
        "journal",
        "pins",
    ],
)
def test_only_whole_heading_cohort_can_reach_contact_replay(tmp_path, monkeypatch, mutation):
    cfg, d, complete, audit, sealed = fixture(tmp_path, monkeypatch)
    if mutation == "cfg_hash":
        cfg["producer_config_file_hash"] = "wrong"
    elif mutation == "reader":
        cfg["audit_source"] = "/not-the-owned-reader.py"
    elif mutation == "audit_hash":
        cfg["source_terminal_readback_hash"] = "wrong"
    elif mutation == "order":
        complete["results"].reverse()
    elif mutation == "partial":
        complete["results"].pop()
    elif mutation == "unsafe_claim":
        complete["teacher_qualified"] = True
    elif mutation == "counter":
        audit["prior_actual_cpu_ik_and_heading_reconstructions_authenticated"] = 5
    elif mutation == "truthy":
        audit["new_physical_steps"] = False
    elif mutation == "unmatched":
        audit["rows"][1]["receipt_hash"] = "wrong"
    elif mutation == "journal":
        audit["ledger_tip"] = "wrong"
    elif mutation == "pins":
        d["input_file_hashes"]["source.py"] = "different"
    if mutation is None:
        assert module.load_closed_heading(cfg, sealed=sealed, pins={})["complete"] == complete
    else:
        with pytest.raises(ValueError):
            module.load_closed_heading(cfg, sealed=sealed, pins={})
