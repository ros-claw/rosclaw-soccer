"""Whole staged source/decision contracts with synthetic upstream evidence.

No physical or neural engines are called, and no successful G1 skill is claimed.
"""

import json
import sys
from pathlib import Path

import pytest

from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout
from scripts import rsi_compare_matched_references as stage
from tests.rsi.test_reference_exam_contact_stage import fixture as original_fixture
from tests.rsi.test_reference_positive_zero import dense as zero_dense
from tests.rsi.test_reference_student_exam_diagnostics import dense as student_dense


def fixture(tmp_path, monkeypatch):
    specs = {}
    for role, zero in (("zero", True), ("student", False)):
        folder = tmp_path / role
        folder.mkdir()
        pc, d, c, a, replay_cfg, save, _ = original_fixture(folder, zero)
        _, _, replay, metrics = (zero_dense if zero else student_dense)()
        for i, receipt in enumerate(c["results"]):
            for rows in (replay["rows"], metrics["rows"]):
                rows[i]["source_receipt_hash"] = receipt["receipt_hash"]
        replay_root = Path(replay_cfg["output_root"])
        replay_root.mkdir()
        save(replay_root / "complete.json", replay)
        metrics_path = folder / "metrics.json"
        save(metrics_path, metrics)
        readout = measured_readout(d, c, replay, metrics)
        completed = folder / "dense-complete.json"
        save(
            completed,
            dict(
                schema="soccer.rsi.reference_exam_contact_complete.v1",
                upstream_readback_hash=a["report_hash"],
                replay_complete_hash=replay["report_hash"],
                metrics_hash=metrics["report_hash"],
                paired_results=c["paired_results"],
                measured_contact_readout=readout,
                new_fixed_control_replay_substeps=replay_cfg["new_substep_budget"],
                total_metrics_endpoints=len(c["results"]),
                **{
                    k: 0
                    for k in (
                        "new_closed_loop_policy_trials",
                        "new_neural_inferences",
                        "new_optimizer_updates",
                    )
                },
                **{
                    k: False
                    for k in (
                        "hardware_authorized",
                        "promotion_authorized",
                        "dataset_admission_authorized",
                        "neural_training_authorized",
                        "teacher_qualified",
                        "fresh_exam",
                    )
                },
            ),
        )
        cc = dict(
            source_head=pc["source_head"],
            producer_unit=f"{role}-native",
            audit_unit=f"{role}-audit",
            producer_config=replay_cfg["producer_config"],
            producer_config_file_hash=replay_cfg["producer_config_file_hash"],
            audit_source=replay_cfg["audit_source"],
            readback=replay_cfg["source_terminal_readback"],
            complete_path=str(completed),
            replay_output_root=str(replay_root),
            metrics_output=str(metrics_path),
            new_substep_budget=replay_cfg["new_substep_budget"],
        )
        cp = folder / "contact-config.json"
        save(cp, cc, key=None)
        specs[role] = dict(
            contact_configuration=str(cp),
            contact_configuration_file_hash=stage.digest(cp),
            contact_unit=f"{role}-contact",
        )
    source = Path(stage.__file__).resolve().parents[1]
    cfg = dict(
        profile="matched-positive-zero-teacher-student-comparison-v1",
        mode="SIM_ONLY",
        source_root=str(source),
        source_head="test-stage-head",
        output_root=str(tmp_path / "result"),
        deadline_unix=10**20,
        cohorts=specs,
        **{
            k: False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "neural_training_authorized",
                "dataset_admission_authorized",
                "private_fresh_accessed",
            )
        },
    )
    cp = tmp_path / "comparison-config.json"
    cp.write_text(json.dumps(cfg))
    checked = []
    monkeypatch.setattr(stage, "require_terminal", checked.append)
    monkeypatch.setattr(
        stage.subprocess,
        "check_output",
        lambda cmd, **kw: (
            ("test-stage-head" if Path(kw["cwd"]) == source else "mock-original-head")
            if cmd == ["git", "rev-parse", "HEAD"]
            else ""
        ),
    )
    return cfg, cp, checked


@pytest.mark.parametrize("mode", ["normal", "preflight", "late_mutation"])
def test_whole_original_both_cohorts_close_once_without_physics_or_optimization(
    tmp_path, monkeypatch, mode
):
    cfg, cp, checked = fixture(tmp_path, monkeypatch)
    root = Path(cfg["output_root"])
    monkeypatch.setattr(
        sys,
        "argv",
        ["compare", "--config", str(cp)] + (["--preflight"] if mode == "preflight" else []),
    )
    if mode == "late_mutation":
        actual = stage.compare_matched_references

        def mutate(*args):
            result = actual(*args)
            Path(cfg["cohorts"]["zero"]["contact_configuration"]).write_text("{}")
            return result

        monkeypatch.setattr(stage, "compare_matched_references", mutate)
        with pytest.raises(ValueError, match="changed during comparison"):
            stage.main()
        assert (root / "failure.json").exists() and not (root / "complete.json").exists()
    else:
        stage.main()
        if mode == "preflight":
            assert not root.exists()
            return
        report = json.loads((root / "complete.json").read_text())
        assert report["comparison"]["comparisons"]["zero_vs_student"]["pairs"] == 8
        assert report["new_optimizer_updates"] == report["new_physical_steps"] == 0
        assert report["original_task_labels_unchanged"] and not report["promotion_authorized"]
        assert checked == [
            f"{r}-{s}" for r in ("zero", "student") for s in ("native", "audit", "contact")
        ]
    with pytest.raises(ValueError, match="unused frozen"):
        stage.main()


@pytest.mark.parametrize(
    "tamper",
    [
        "contact_config",
        "producer",
        "source_head",
        "reader",
        "audit_flag",
        "dense",
        "owned_receipt",
        "witness",
    ],
)
def test_bad_original_evidence_cannot_produce_comparison(tmp_path, monkeypatch, tamper):
    cfg, cp, _ = fixture(tmp_path, monkeypatch)
    spec = cfg["cohorts"]["zero"]
    contact = Path(spec["contact_configuration"])
    cc = json.loads(contact.read_text())

    def reseal(path, value, key="report_hash"):
        value[key] = stage.hash_json({k: v for k, v in value.items() if k != key})
        Path(path).write_text(json.dumps(value))

    if tamper == "contact_config":
        contact.write_text("{}")
    elif tamper == "producer":
        Path(cc["producer_config"]).write_text("{}")
    elif tamper == "source_head":
        cc["source_head"] = "borrowed"
        contact.write_text(json.dumps(cc))
        spec["contact_configuration_file_hash"] = stage.digest(contact)
        cp.write_text(json.dumps(cfg))
    elif tamper == "reader":
        Path(cc["audit_source"]).write_text("changed reader")
    elif tamper == "audit_flag":
        a = json.loads(Path(cc["readback"]).read_text())
        a["matched_zero_all300_reference_increments_exactly_zero"] = False
        reseal(cc["readback"], a)
    elif tamper == "dense":
        d = json.loads(Path(cc["complete_path"]).read_text())
        d["new_optimizer_updates"] = 1
        reseal(cc["complete_path"], d)
    elif tamper == "owned_receipt":
        producer = json.loads(Path(cc["producer_config"]).read_text())
        receipt = Path(producer["output_root"]) / "receipt-0001.json"
        r = json.loads(receipt.read_text())
        r["outcome"]["task_pass"] = False
        reseal(receipt, r, "receipt_hash")
    elif tamper == "witness":
        a = json.loads(Path(cc["readback"]).read_text())
        a["input_file_hashes"] = {str(contact): "sha256:" + "0" * 64}
        reseal(cc["readback"], a)
        # Preserve other relations so the actual witness byte check is reached.
        d = json.loads(Path(cc["complete_path"]).read_text())
        d["upstream_readback_hash"] = a["report_hash"]
        reseal(cc["complete_path"], d)
    monkeypatch.setattr(sys, "argv", ["compare", "--config", str(cp)])
    with pytest.raises(ValueError):
        stage.main()
    assert not Path(cfg["output_root"]).exists()
