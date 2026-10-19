"""Mocked source orchestration, not physical qualification evidence."""

import copy
import json
import sys
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.original_ik_planner_diagnostics import ik_contact_readout
from rosclaw_soccer.rsi.original_ik_planner_scope import assess_ik_scope
from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import pulse_contact_readout
from rosclaw_soccer.rsi.planner_navigation_pulse_scope import assess_pulse_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_read_pulse_stage_pairs as reader
from tests.rsi.test_first_touch_stage_diagnostics import outcome
from tests.rsi.test_original_ik_planner_scope import receipts as ik_receipts
from tests.rsi.test_pre_planner_pulse_cohort import cohort, dense


@pytest.mark.parametrize(
    "mutation",
    [None, "membership", "metrics", "source", "hash", "running", "admission", "counts", "upstream"],
)
@pytest.mark.parametrize("original_ik", [False, True])
def test_whole_stage_readback_requires_original_cohort_and_dense_evidence(
    tmp_path, monkeypatch, mutation, original_ik
):
    p, d, c, a = cohort()
    if original_ik:
        d["scope"], c["results"] = ik_receipts()
    a["report_hash"] = "audit"
    for r in c["results"]:
        g, role = r["group"], r["role"]
        r["outcome"] = outcome(
            None
            if g == 28 or (g == 76 and role in ("pulse", "ik_pulse"))
            else 151
            if g == 76
            else 231,
            weak=g == 76,
        )
        if original_ik:
            r["receipt_hash"] = hash_json({k: v for k, v in r.items() if k != "receipt_hash"})
    assess = assess_ik_scope if original_ik else assess_pulse_scope
    c["assessment"] = assess(d["scope"], c["results"])
    replay, metrics = dense(d, c)
    measured = (ik_contact_readout if original_ik else pulse_contact_readout)(d, c, replay, metrics)
    if mutation == "membership":
        replay["rows"].reverse()
    elif mutation == "metrics":
        metrics["rows"][0]["original_outcome"] = {}

    def persist(name, doc):
        doc = copy.deepcopy(doc)
        doc["report_hash"] = hash_json({k: v for k, v in doc.items() if k != "report_hash"})
        path = tmp_path / name
        path.write_text(json.dumps(doc))
        return str(path), doc["report_hash"]

    rp, rh = persist("replay.json", replay)
    mp, mh = persist("metrics.json", metrics)
    witness = tmp_path / "original_source.py"
    witness.write_text("# frozen source\n")
    contact = dict(
        schema="soccer.rsi.original_ik_planner_contact_complete.v1"
        if original_ik
        else "soccer.rsi.pre_planner_pulse_contact_complete.v1",
        upstream_readback_hash="other" if mutation == "upstream" else "audit",
        replay_complete_hash=rh,
        metrics_hash=mh,
        measured_contact_readout=measured,
        new_fixed_control_replay_substeps=18000,
        new_closed_loop_policy_trials=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        total_metrics_endpoints=6,
        teacher_qualified=False,
        dataset_admission_authorized=mutation == "admission",
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        input_file_hashes={str(witness): reader.digest(witness)},
    )
    if mutation == "counts":
        contact["new_neural_inferences"] = 1
    cp, ch = persist("contact.json", contact)
    if mutation == "source":
        witness.write_text("# modified source\n")
    output = tmp_path / "stage.json"
    cfg = dict(
        mode="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        source_root=str(Path(reader.__file__).resolve().parents[1]),
        source_head="frozen",
        output=str(output),
        cohort={"test": "whole-cohort-loader"},
        contact=dict(
            unit="contact.service",
            complete=cp,
            complete_hash="other" if mutation == "hash" else ch,
            replay_complete=rp,
            metrics=mp,
        ),
    )
    if original_ik:
        cfg["stage_profile"] = "original-ik-pulse"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(cfg))
    monkeypatch.setattr(
        reader.subprocess,
        "check_output",
        lambda argv, **kw: "frozen" if argv[1] == "rev-parse" else "",
    )

    def load(spec, **kwargs):
        assert spec == {"test": "whole-cohort-loader"}
        assert callable(kwargs["sealed"])
        return dict(producer=p, declaration=d, complete=c, audit=a)

    monkeypatch.setattr(
        reader, "load_closed_original_ik" if original_ik else "load_closed_pulse", load
    )

    def terminal(unit):
        assert unit == "contact.service"
        if mutation == "running":
            raise ValueError("not terminal")

    monkeypatch.setattr(reader, "require_terminal", terminal)
    monkeypatch.setattr(sys, "argv", ["reader", "--config", str(config_path)])
    if mutation is None:
        reader.main()
        result = json.loads(output.read_text())
        assert result["schema"] == (
            "soccer.rsi.original_ik_first_touch_stage_pairs_readback.v1"
            if original_ik
            else "soccer.rsi.pulse_first_touch_stage_pairs_readback.v1"
        )
        diag = result["stage_diagnostics"]
        assert diag["contact_regressions"] == 1
        assert diag["original_m0_pairs"]["net_success_gain"] == 0
        assert result["source_native_bodies"] == 6
        assert result["new_physical_steps"] == result["new_neural_inferences"] == 0
        assert not result["teacher_qualified"]
        assert result["input_file_hashes"][str(witness)] == reader.digest(witness)
        old = output.read_bytes()
        with pytest.raises(ValueError, match="unused"):
            reader.main()
        assert output.read_bytes() == old
    else:
        with pytest.raises(ValueError):
            reader.main()
        assert not output.exists()
        assert output.with_suffix(".started.failure.json").exists()


@pytest.mark.parametrize(
    "mutation", [None, "native_running", "audit_running", "dense_running", "override", "truthy"]
)
def test_future_hashes_bind_only_after_all_original_processes_close(monkeypatch, mutation):
    cfg = dict(
        bind_completed_source_hashes=True,
        cohort=dict(
            producer_unit="native", audit_unit="audit", source_terminal_readback="audit.json"
        ),
        contact=dict(unit="dense", complete="dense.json"),
    )
    original = copy.deepcopy(cfg)
    terminals, reads = [], []

    def terminal(unit):
        if mutation == unit + "_running":
            raise ValueError("not terminal")
        terminals.append(unit)

    monkeypatch.setattr(reader, "require_terminal", terminal)

    def sealed(path):
        assert terminals == ["native", "audit", "dense"]
        reads.append(path)
        return dict(report_hash="sealed-" + path)

    if mutation == "override":
        cfg["contact"]["complete_hash"] = "pinned"
    elif mutation == "truthy":
        cfg["bind_completed_source_hashes"] = 1
    if mutation is None:
        cohort_spec, contact_spec = reader.completed_inputs(cfg, sealed=sealed)
        assert cohort_spec["source_terminal_readback_hash"] == "sealed-audit.json"
        assert contact_spec["complete_hash"] == "sealed-dense.json"
        assert cfg == original
    else:
        with pytest.raises(ValueError):
            reader.completed_inputs(cfg, sealed=sealed)
        assert not reads
