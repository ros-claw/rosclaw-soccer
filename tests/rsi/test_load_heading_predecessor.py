"""Mocked closure authentication only; never physical improvement evidence."""

from pathlib import Path

import pytest

from scripts import rsi_load_heading_predecessor as module


def fixture(tmp_path, monkeypatch):
    spec = dict(
        stage_unit="stage.service",
        stage_readback=str(tmp_path / "stage.json"),
        stage_file_hash="file",
        stage_readback_hash="stage",
    )
    cohort = dict(
        declaration={}, complete=dict(report_hash="native"), audit=dict(report_hash="audit")
    )
    measured = dict(
        assessment=dict(stages={"unchanged": True}), pulse_endpoints_with_nonfoot_over1n=0
    )
    stage = dict(
        schema="soccer.rsi.original_ik_first_touch_stage_pairs_readback.v1",
        report_hash="stage",
        configuration=dict(
            source_root=str(tmp_path),
            source_head="head",
            output=spec["stage_readback"],
            stage_profile="original-ik-pulse",
            mode="SIM_ONLY",
        ),
        input_file_hashes={"source.py": "file"},
        effective_cohort_spec={},
        effective_contact_spec=dict(
            unit="contact.service",
            complete="contact.json",
            complete_hash="contact",
            replay_complete="replay.json",
            metrics="metrics.json",
        ),
        native_complete_hash="native",
        native_audit_hash="audit",
        contact_complete_hash="contact",
        stage_diagnostics=measured["assessment"]["stages"],
        pulse_dense_nonfoot_endpoints=0,
        all_six_receipts_and_failures_retained=True,
        source_native_bodies=6,
        derived_courses=3,
        source_scene_clusters=3,
        new_physical_steps=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        receiving_skill_qualified=False,
        successor_readiness_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    contact = dict(
        schema="soccer.rsi.original_ik_planner_contact_complete.v1",
        report_hash="contact",
        upstream_readback_hash="audit",
        replay_complete_hash="replay",
        metrics_hash="metrics",
        measured_contact_readout=measured,
        new_fixed_control_replay_substeps=18000,
        new_closed_loop_policy_trials=0,
        new_neural_inferences=0,
        new_optimizer_updates=0,
        total_metrics_endpoints=6,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        input_file_hashes={"dense.py": "file"},
    )
    docs = {
        spec["stage_readback"]: stage,
        "contact.json": contact,
        "replay.json": dict(report_hash="replay"),
        "metrics.json": dict(report_hash="metrics"),
    }
    units = []
    monkeypatch.setattr(module, "require_terminal", units.append)
    monkeypatch.setattr(module, "digest", lambda _: "file")
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda args, **kw: "head" if "rev-parse" in args else ""
    )
    monkeypatch.setattr(module, "load_closed_original_ik", lambda *a, **kw: cohort)
    monkeypatch.setattr(module, "ik_contact_readout", lambda *a: measured)
    return spec, stage, contact, lambda p: docs[str(p)], units


@pytest.mark.parametrize(
    "mutation",
    [
        None,
        "stage_hash",
        "file_hash",
        "profile",
        "truthy",
        "partial",
        "authority",
        "source",
        "native",
        "audit",
        "dense_hash",
        "dense_counter",
        "dense_authority",
        "stages",
        "nonfoot",
        "pins",
        "dense_pins",
        "symlink",
    ],
)
def test_whole_closed_source_dense_and_stage_cannot_be_bypassed(tmp_path, monkeypatch, mutation):
    spec, stage, contact, sealed, units = fixture(tmp_path, monkeypatch)
    if mutation == "stage_hash":
        spec["stage_readback_hash"] = "wrong"
    elif mutation == "file_hash":
        spec["stage_file_hash"] = "wrong"
    elif mutation == "profile":
        stage["configuration"]["stage_profile"] = "zero-reference-pulse"
    elif mutation == "truthy":
        stage["new_physical_steps"] = False
    elif mutation == "partial":
        stage["source_native_bodies"] = 5
    elif mutation == "authority":
        stage["teacher_qualified"] = True
    elif mutation == "source":
        stage["configuration"]["source_head"] = "wrong"
    elif mutation in ("native", "audit"):
        stage["native_complete_hash" if mutation == "native" else "native_audit_hash"] = "wrong"
    elif mutation == "dense_hash":
        contact["metrics_hash"] = "wrong"
    elif mutation == "dense_counter":
        contact["new_fixed_control_replay_substeps"] = 18000.0
    elif mutation == "dense_authority":
        contact["neural_training_authorized"] = True
    elif mutation == "stages":
        stage["stage_diagnostics"] = {}
    elif mutation == "nonfoot":
        stage["pulse_dense_nonfoot_endpoints"] = 1
    elif mutation in ("pins", "dense_pins"):
        (stage if mutation == "pins" else contact)["input_file_hashes"]["changed.py"] = "wrong"
    elif mutation == "symlink":
        Path(spec["stage_readback"]).symlink_to(tmp_path / "missing")
    pins = {}
    if mutation is None:
        result = module.load_heading_predecessor(spec, sealed=sealed, pins=pins)
        assert result["contact"] is contact
        assert units == ["stage.service", "contact.service"]
        assert pins == {"source.py": "file", "dense.py": "file"}
    else:
        with pytest.raises(ValueError):
            module.load_heading_predecessor(spec, sealed=sealed, pins=pins)
