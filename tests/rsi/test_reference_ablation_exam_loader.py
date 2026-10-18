"""Synthetic prior cohort plus mocked numeric-fit authentication, no physics."""

import json

import pytest

from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout
from scripts import rsi_load_reference_ablation_exam as loader
from scripts.rsi_stage_release_contact_replay import digest
from tests.rsi.test_reference_student_exam_diagnostics import cohort, dense


@pytest.mark.parametrize(
    "tamper",
    [
        None,
        "unit",
        "config_hash",
        "recursive",
        "partial",
        "other_fit",
        "dense",
        "reader",
        "input",
        "grant",
    ],
)
def test_ablation_waits_for_whole_prior_unchanged_exam(tmp_path, monkeypatch, tamper):
    pc, d, c, a = cohort()
    _, _, replay, metrics = dense()
    paths = {k: tmp_path / k for k in ("fc", "pc", "cc", "readback", "dense", "reader", "metrics")}
    pc.update(
        output_root=str(tmp_path / "old"),
        fit_configuration="original-fit-config",
        fit_configuration_file_hash="old-fit-bytes",
    )
    d.update(pc)
    cfg = dict(pc)
    fc = dict(output_root=str(tmp_path / "fit"))
    cc = dict(
        producer_config=str(paths["pc"]),
        readback=str(paths["readback"]),
        complete_path=str(paths["dense"]),
        audit_source=str(paths["reader"]),
        replay_output_root=str(tmp_path / "replay"),
        metrics_output=str(paths["metrics"]),
    )
    paths["reader"].write_text("synthetic-reader")
    paths["pc"].write_text(json.dumps(pc))
    paths["fc"].write_text(json.dumps(fc))
    paths["cc"].write_text(json.dumps(cc))
    spec = dict(
        fit_configuration=str(paths["fc"]),
        fit_configuration_file_hash=digest(paths["fc"]),
        fit_unit="fit.service",
        predecessor_configuration=str(paths["pc"]),
        predecessor_configuration_file_hash=digest(paths["pc"]),
        predecessor_native_unit="native.service",
        predecessor_audit_unit="audit.service",
        predecessor_contact_unit="dense.service",
        predecessor_readback=str(paths["readback"]),
        predecessor_dense_complete=str(paths["dense"]),
        predecessor_contact_configuration=str(paths["cc"]),
        predecessor_contact_configuration_file_hash=digest(paths["cc"]),
    )
    cfg["matched_ablation"] = spec
    a.update(report_hash="audit", reader_source_hash=digest(paths["reader"]), input_file_hashes={})
    d.update(input_file_hashes={}, fit_stage_hash="old-fit-stage")
    replay["report_hash"], metrics["report_hash"] = "replay", "metrics"
    dc = dict(
        upstream_readback_hash="audit",
        paired_results=c["paired_results"],
        replay_complete_hash="replay",
        metrics_hash="metrics",
        measured_contact_readout=measured_readout(d, c, replay, metrics),
        new_fixed_control_replay_substeps=39000,
        new_closed_loop_policy_trials=0,
        promotion_authorized=False,
        hardware_authorized=False,
        teacher_qualified=False,
    )
    stage, fit = (
        dict(input_file_hashes={}),
        dict(test="numeric fit already authenticated separately"),
    )
    files = {
        str(tmp_path / "fit/complete.json"): stage,
        str(tmp_path / "fit/numeric-fit.json"): fit,
        str(tmp_path / "old/declaration.json"): d,
        str(tmp_path / "old/complete.json"): c,
        str(paths["readback"]): a,
        str(paths["dense"]): dc,
        str(tmp_path / "replay/complete.json"): replay,
        str(paths["metrics"]): metrics,
    }
    required = []

    def terminal(unit):
        required.append(unit)
        if tamper == "unit" and unit == "audit.service":
            raise ValueError("nonterminal")

    monkeypatch.setattr(loader, "require_terminal", terminal)
    monkeypatch.setattr(loader, "require_matched_fit", lambda *args, **kw: d["models"]["student"])
    if tamper == "config_hash":
        spec["fit_configuration_file_hash"] = "wrong"
    elif tamper == "recursive":
        pc["matched_ablation"] = {}
        paths["pc"].write_text(json.dumps(pc))
        spec["predecessor_configuration_file_hash"] = digest(paths["pc"])
    elif tamper == "partial":
        c["results"].pop()
    elif tamper == "other_fit":
        d["fit_stage_hash"] = "other"
    elif tamper == "dense":
        dc["replay_complete_hash"] = "other"
    elif tamper == "reader":
        a["reader_source_hash"] = "changed"
    elif tamper == "input":
        stage["input_file_hashes"][str(paths["reader"])] = "changed"
    elif tamper == "grant":
        dc["promotion_authorized"] = True
    pins = {}

    def run():
        return loader.load_ablation(
            cfg,
            sealed=lambda p: files[str(p)],
            digest=digest,
            pins=pins,
            original_stage=dict(report_hash="old-fit-stage"),
            original_fit=dict(numeric_model=d["models"]["student"]),
        )

    if tamper:
        with pytest.raises(ValueError):
            run()
    else:
        assert run() == (d["models"]["student"], stage, fit)
        assert required == ["fit.service", "native.service", "audit.service", "dense.service"]
        assert pins[str(paths["cc"])] == digest(paths["cc"])
