"""Synthetic upstream authority/physics and optimizer; no real training claim.

Exercises actual once-only orchestration, files, physical difference re-read,
whole measured selection, source splitting and finite fit invocation.
"""

import copy
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.rsi.positive_lateral_diagnostics import measured_contact_readout
from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_fit_pre_parent_reference as stage
from scripts import rsi_read_reference_student_three as reference_audit
from scripts import rsi_reference_student_three_pilot as pilot
from tests.rsi.test_positive_lateral_diagnostics import dense_fixture
from tests.rsi.test_reference_student_three_pipeline import fixture as reference_fixture


def config():
    return dict(
        fit_profile="positive-sixteen-reference-imitation-v1",
        upstream_units=list(stage.UPSTREAM_UNITS),
        offline_numeric_fit_authorized=True,
        fit_steps=128,
        fit_seed=stage.FIT_SEED,
        new_physical_budget=0,
        new_foundation_inferences=0,
        maximum_optimizer_steps=128,
        mode="SIM_ONLY",
        hardware_authorized=False,
        promotion_authorized=False,
        private_fresh_accessed=False,
        teacher_auto_admission=False,
        validation_checkpoint_selection=False,
        online_rl_claimed=False,
    )


@pytest.mark.parametrize(
    "changed", [None, "units", "extra_step", "bool", "fit_off", "online", "hardware", "validation"]
)
def test_exact_offline_numeric_protocol_cannot_be_widened(changed):
    cfg = config()
    if changed == "units":
        cfg["upstream_units"].pop()
    elif changed == "extra_step":
        cfg["fit_steps"] = 129
    elif changed == "bool":
        cfg["new_physical_budget"] = False
    elif changed == "fit_off":
        cfg["offline_numeric_fit_authorized"] = False
    elif changed in ("online", "hardware", "validation"):
        cfg[
            {
                "online": "online_rl_claimed",
                "hardware": "hardware_authorized",
                "validation": "validation_checkpoint_selection",
            }[changed]
        ] = True
    if changed:
        with pytest.raises(ValueError):
            stage.require_scope(cfg)
    else:
        stage.require_scope(cfg)


def fixture(tmp_path, monkeypatch, condition):
    refdir = tmp_path / "reference"
    refdir.mkdir()
    refcfg, refpath, _, _ = reference_fixture(refdir, monkeypatch, "normal")
    refcfg["behavior_model_hash"] = "sha256:" + "a" * 64
    refpath.write_text(json.dumps(refcfg))
    monkeypatch.setattr(sys, "argv", ["pilot", "--config", str(refpath)])
    pilot.main()
    refread = refdir / "readback.json"
    monkeypatch.setattr(
        sys, "argv", ["audit", "--config", str(refpath), "--unit", "fake", "--output", str(refread)]
    )
    reference_audit.main()
    rroot = Path(refcfg["output_root"])
    source = Path(stage.__file__).resolve().parents[1]
    root = tmp_path / "positive"
    root.mkdir()

    def save(path, value, key="report_hash"):
        if key:
            value[key] = hash_json({k: v for k, v in value.items() if k != key})
        path.write_text(json.dumps(value))
        return value

    declaration, complete, replay, metrics = dense_fixture()
    declaration.update(prepared_pins={}, input_file_hashes={})
    save(root / "declaration.json", declaration)
    trace = root / "shared_trace.npz"
    trace.write_bytes((rroot / "attempt-0000/physical_trace.npz").read_bytes())
    for i, (job, r) in enumerate(zip(declaration["jobs"], complete["results"], strict=True)):
        r.update(job, witnesses=dict(trace=dict(path=str(trace), sha256=stage.digest(trace))))
        save(root / f"receipt-{i:04d}.json", r, "receipt_hash")
        for m in (replay["rows"][i], metrics["rows"][i]):
            m["source_receipt_hash"] = r["receipt_hash"]
    save(root / "complete.json", complete)
    audit_path = root / "audit.json"
    audit = save(
        audit_path,
        dict(
            reader_source_hash=stage.digest(
                source / "scripts/rsi_read_mesh_contact_native_pilot.py"
            )
        ),
    )
    if condition == "rejected":
        metrics["rows"][1]["metrics"]["first_nonfoot_robot_contact_substep"] = 1000
    replay_root = root / "replay"
    replay_root.mkdir()
    save(replay_root / "complete.json", replay)
    metrics_path = root / "metrics.json"
    save(metrics_path, metrics)
    measured = measured_contact_readout(declaration, complete, replay, metrics)
    positive_path = root / "config.json"
    save(
        positive_path,
        dict(
            output_root=str(root),
            mesh_task_contact_world_hash=refcfg["mesh_task_contact_world_hash"],
            behavior_model_hash="sha256:" + "a" * 64,
        ),
        key=None,
    )
    contact_path = root / "contact-config.json"
    contact = dict(
        producer_config=str(positive_path),
        readback=str(audit_path),
        complete_path=str(root / "contact-complete.json"),
        replay_output_root=str(replay_root),
        metrics_output=str(metrics_path),
        audit_source=str(source / "scripts/rsi_read_mesh_contact_native_pilot.py"),
    )
    save(contact_path, contact, key=None)
    save(
        Path(contact["complete_path"]),
        dict(
            upstream_readback_hash=audit["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            measured_contact_readout=measured,
            paired_development=complete["paired_development"],
        ),
    )
    cfg = dict(
        config(),
        source_root=str(source),
        source_head="test-head",
        core_root=str(tmp_path),
        output_root=str(tmp_path / "fit"),
        deadline_unix=10**20,
        reference_readback=str(refread),
    )
    for key, path in (
        ("positive_configuration", positive_path),
        ("positive_contact_configuration", contact_path),
        ("reference_configuration", refpath),
    ):
        cfg[key], cfg[key + "_file_hash"] = str(path), stage.digest(path)
    path = tmp_path / "fit-config.json"
    save(path, cfg, key=None)
    # Original cohort's independent numerical qualification is mocked here;
    # its real whole-reader contract has separate full tests.
    monkeypatch.setattr(stage, "require_cohort", lambda *a: complete["paired_development"])
    checked_units = []
    monkeypatch.setattr(stage, "require_terminal", checked_units.append)

    def observation(t):
        x = np.zeros((300, 135), dtype=np.float64)
        x[:, -1] = (300 - np.arange(300)) / 300
        y = np.zeros((300, 12), dtype=np.float64)
        y[30:, 4] = 0.08
        return dict(observation=x, same_state_reference_target_rad=y)

    monkeypatch.setattr(stage, "trace_observations", observation)
    calls = []

    def mock_fit(batch, **kw):
        calls.append((copy.deepcopy(batch), kw))
        return dict(
            schema="test.synthetic_optimizer_not_real_fit",
            report_hash=hash_json("fake"),
            learning_receipt=dict(completed_optimizer_steps=128),
        )

    monkeypatch.setattr(stage, "fit_numeric_reference_student", mock_fit)
    if condition == "witness":
        trace.write_bytes(trace.read_bytes() + b" ")
    return cfg, path, calls, checked_units


@pytest.mark.parametrize("condition", ["normal", "preflight", "rejected", "witness", "fit_failure"])
def test_once_only_fit_or_refusal_and_original_witness_integrity(tmp_path, monkeypatch, condition):
    cfg, path, calls, checked = fixture(tmp_path, monkeypatch, condition)
    if condition == "fit_failure":

        def failing(*a, **kw):
            raise ValueError("mock optimizer failure")

        monkeypatch.setattr(stage, "fit_numeric_reference_student", failing)
    monkeypatch.setattr(
        sys,
        "argv",
        ["fit", "--config", str(path)] + (["--preflight"] if condition == "preflight" else []),
    )
    root = Path(cfg["output_root"])
    if condition in ("witness", "fit_failure"):
        with pytest.raises(ValueError, match="witness|mock optimizer"):
            stage.main()
        assert not (root / "complete.json").exists()
        if condition == "fit_failure":
            assert (root / "failure.json").exists() and (root / "started.json").exists()
            with pytest.raises(ValueError, match="unused output"):
                stage.main()
        return
    stage.main()
    assert checked == stage.UPSTREAM_UNITS
    if condition == "preflight":
        assert not calls and not root.exists()
        return
    result = json.loads((root / "complete.json").read_text())
    assert result["all_source_failures_retained"]
    assert not result["own_body_student_evaluated"] and not result["teacher_qualified"]
    if condition == "rejected":
        assert not calls and result["optimizer_updates"] == 0
        assert result["status"] == "REFUSED_BY_ORIGINAL_PHYSICAL_EVIDENCE"
    else:
        assert len(calls) == 1 and result["optimizer_updates"] == 128
        assert calls[0][1]["steps"] == 128 and calls[0][1]["seed"] == stage.FIT_SEED
        assert set(calls[0][0]["source_seeds"]) == {20261077, 20261078, 20261093, 20261096}
    with pytest.raises(ValueError, match="unused output"):
        stage.main()
