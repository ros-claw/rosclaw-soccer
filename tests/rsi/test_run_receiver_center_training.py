import json
import sys
from types import SimpleNamespace

from scripts import rsi_run_receiver_center_training as runner


def test_insufficient_data_records_no_optimizer(monkeypatch, tmp_path):
    dataset = tmp_path / "data"
    dataset.mkdir()
    quality = {
        "research_training_data_sufficient": False,
        "positive_conditions": 0,
        "negative_conditions": 8,
    }
    (dataset / "complete.json").write_text(json.dumps({"research_data_sufficiency": quality}))
    audit = tmp_path / "audit.json"
    audit.write_text(
        json.dumps(
            dict(
                status="RECEIVER_ONPOLICY_READBACK_AND_DENSITY_NOT_TRAINED",
                source_head="audit-head",
                dataset_hash="pinned",
                reward_profile="center-strict-full-history-v1",
                research_data_sufficiency=quality,
                exact_inputs_actions_rewards_and_source_replayed=True,
                optimizer_authorized=False,
                promotion_authorized=False,
                hardware_authorized=False,
            )
        )
    )
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
            str(dataset),
            "--audit",
            str(audit),
            "--anchor-admission",
            str(tmp_path / "anchor"),
            "--anchor-dataset",
            str(tmp_path / "old"),
            "--output",
            str(output),
            "--audit-unit",
            "rosclaw-receiver-center-data-audit-test.service",
            "--audit-head",
            "audit-head",
            "--anchor-admission-hash",
            "pinned",
            "--anchor-dataset-hash",
            "pinned",
            "--expected-head",
            "head",
        ],
    )
    runner.main()
    result = json.loads((output / "complete.json").read_text())
    assert result["status"] == "CENTER_RECEIVER_INSUFFICIENT_NO_TRAINING"
    assert result["actor_updates"] == 0 and result["critic_updates"] == 0
    assert len(calls) == 1
