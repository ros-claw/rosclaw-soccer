import json
import sys
from types import SimpleNamespace

from scripts import rsi_run_receiver_center_candidate_audits as runner


def test_insufficient_training_has_no_candidate_audit(monkeypatch, tmp_path):
    for name in ("dataset", "training", "admission"):
        (tmp_path / name).mkdir()
        (tmp_path / name / "complete.json").write_text("{}")
    result = dict(
        status="CENTER_RECEIVER_INSUFFICIENT_NO_TRAINING",
        source_head="producer",
        dataset_hash="pinned",
        audit_hash="pinned",
        actor_updates=0,
        critic_updates=0,
        promotion_authorized=False,
        hardware_authorized=False,
        goal_complete=False,
    )
    (tmp_path / "training/complete.json").write_text(json.dumps(result))
    (tmp_path / "audit").write_text("{}")
    monkeypatch.setattr(runner, "clean", lambda _: None)
    monkeypatch.setattr(runner, "digest", lambda _: "pinned")
    calls = []

    def run(command, **_kwargs):
        calls.append(command)
        return SimpleNamespace(
            stdout="LoadState=loaded\nActiveState=inactive\nResult=success\nExecMainStatus=0\n"
        )

    monkeypatch.setattr(runner.subprocess, "run", run)
    output = tmp_path / "out"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run",
            "--dataset",
            str(tmp_path / "dataset"),
            "--data-audit",
            str(tmp_path / "audit"),
            "--admission",
            str(tmp_path / "admission"),
            "--training",
            str(tmp_path / "training"),
            "--output",
            str(output),
            "--training-unit",
            "rosclaw-receiver-center-three-seed-training-test.service",
            "--producer-head",
            "producer",
            "--admission-hash",
            "pinned",
            "--expected-head",
            "head",
        ],
    )
    runner.main()
    assert (
        json.loads((output / "complete.json").read_text())["status"]
        == "CENTER_RECEIVER_NO_CANDIDATE_TO_AUDIT"
    )
    assert len(calls) == 1
