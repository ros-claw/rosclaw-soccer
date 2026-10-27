import json
import sys

import numpy as np

from scripts import rsi_audit_relay_balance_selector as reviewer
from scripts import rsi_train_relay_balance_selector as training
from scripts.rsi_diagnose_pass_contact_timing import digest


def test_offline_group_split_trains_and_never_authorizes_motor(monkeypatch, tmp_path):
    dataset, audit = tmp_path / "data", tmp_path / "audit"
    dataset.mkdir()
    audit.mkdir()
    data_path = dataset / "dataset.npz"
    x = np.random.default_rng(1).normal(size=(12, 17)).astype(np.float32)
    group = np.repeat(np.arange(3), 4)
    success = np.zeros((12, 4), bool)
    failure = np.zeros((12, 4), bool)
    kick = np.zeros((12, 4), bool)
    success[:, 0], kick[:, 0] = True, True
    np.savez(data_path, features=x, context=group, success=success, failure=failure, kick=kick)
    receipt = dataset / "complete.json"
    receipt.write_text(
        json.dumps(
            dict(
                status="PAIRED_RELAY_BALANCE_DECISIONS_EXPORTED_NOT_TRAINED",
                dataset_hash=digest(data_path),
                handoff_decisions=12,
                no_handoff_conditions=0,
            )
        )
    )
    audit_report = audit / "complete.json"
    audit_report.write_text(
        json.dumps(
            dict(
                status="PAIRED_RELAY_BALANCE_DECISIONS_READBACK_NOT_REINTEGRATED",
                dataset_hash=digest(receipt),
            )
        )
    )
    monkeypatch.setattr(training, "clean", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "train",
            "--dataset",
            str(dataset),
            "--dataset-hash",
            digest(receipt),
            "--audit",
            str(audit),
            "--audit-hash",
            digest(audit_report),
            "--output",
            str(tmp_path / "trained"),
            "--expected-head",
            "fixture",
        ],
    )
    training.main()
    report = json.loads((tmp_path / "trained" / "complete.json").read_text())
    assert report["optimizer_updates"] == 4500
    assert report["physics_evaluated_as_controller"] is False
    assert report["promotion_authorized"] is False
    assert len(report["rows"]) == 9
    assert all(
        row["training_decisions"] == 8 and row["heldout_decisions"] == 4 for row in report["rows"]
    )
    model_root = tmp_path / "trained"
    monkeypatch.setattr(reviewer, "clean", lambda _: None)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "audit",
            "--source",
            str(model_root),
            "--source-hash",
            digest(model_root / "complete.json"),
            "--dataset",
            str(dataset),
            "--dataset-hash",
            digest(receipt),
            "--output",
            str(tmp_path / "audit-report.json"),
            "--expected-head",
            "fixture",
        ],
    )
    reviewer.main()
    replay = json.loads((tmp_path / "audit-report.json").read_text())
    assert len(replay["rows"]) == 9
    assert replay["physics_evaluated_as_controller"] is False
