"""Synthetic process-control tests; never launch a simulator or real worker."""

import hashlib
import json
import sys
from pathlib import Path

import pytest

from scripts import rsi_follow_native_continuation as module


def fixture(tmp_path, monkeypatch):
    source = tmp_path / "source"
    source.mkdir()
    monkeypatch.chdir(source)
    learner = tmp_path / "learner"
    learner.mkdir()
    receipt = dict(
        status="NATIVE_OPTION_CONTINUATION132_REQUIRES_FULL_EXAM",
        end_episode=132,
        resume_checkpoint_hash="cp",
        resume_source_hash="parent",
        resume_proof_hash="proof",
        input_file_hashes={},
        results=[dict(episode=i) for i in range(20, 132)],
    )
    (learner / "complete.json").write_text(json.dumps(receipt))
    (learner / "declaration.json").write_text(json.dumps(dict(source_head="learner-head")))
    tasks = {}
    for task in ("train64", "dev64"):
        folder = tmp_path / task
        folder.mkdir()
        (folder / "complete.json").write_text("{}")
        h = "sha256:" + hashlib.sha256(b"{}").hexdigest()
        tasks[task] = dict(starts_root=str(folder), starts_hash=h)
        if task == "train64":
            tasks[task].update(baseline_root=str(folder), baseline_hash=h)
    plan = tmp_path / "plan.json"
    plan.write_text(
        json.dumps(
            dict(
                schema="soccer.rsi.continuation132_followup_plan.v1",
                worker_unit="learner.service",
                learner_root=str(learner),
                learner_source_head="learner-head",
                resume_checkpoint_hash="cp",
                resume_source_hash="parent",
                resume_proof_hash="proof",
                tasks=tasks,
                promotion_authorized=False,
            )
        )
    )
    output = tmp_path / "chain"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "follow",
            "--plan",
            str(plan),
            "--plan-hash",
            "sha256:" + hashlib.sha256(plan.read_bytes()).hexdigest(),
            "--expected-head",
            "head",
            "--output",
            str(output),
        ],
    )
    calls = []
    polls = []

    def read(command, **kwargs):
        if command[0] == "git":
            return b"" if command[-1] == "--porcelain" else "head\n"
        assert command[0] == "systemctl" and "learner.service" in command
        polls.append(command)
        # Even with complete.json present, wait for real worker terminal state.
        live = len(polls) == 1
        return "\n".join(
            [
                "MainPID=" + ("42" if live else "0"),
                "LoadState=loaded",
                "ActiveState=active",
                "SubState=" + ("running" if live else "exited"),
                "ExecMainStatus=0",
            ]
        )

    def execute(command, **kwargs):
        assert kwargs["check"] is True and kwargs["env"]["CUDA_VISIBLE_DEVICES"] == ""
        assert command[0] == sys.executable and command[1] == "-m"
        calls.append(command)
        path = Path(command[command.index("--output") + 1])
        if command[2] == "scripts.rsi_exam_native_contact_policy":
            assert (
                command[command.index("--checkpoint-intent") + 1] == "bounded-recovery-continuation"
            )
            path.mkdir()
            (path / "complete.json").write_text(json.dumps(dict(summary={"synthetic": True})))
        else:
            assert command[2] == "scripts.rsi_audit_finite_option_exam"
            path.write_text("{}")

    monkeypatch.setattr(module.subprocess, "check_output", read)
    monkeypatch.setattr(module.subprocess, "run", execute)
    monkeypatch.setattr(module.time, "sleep", lambda _: None)
    return output, learner, calls, polls


def test_waits_for_exit_then_runs_only_fixed_exams_and_audits(tmp_path, monkeypatch):
    output, _, calls, polls = fixture(tmp_path, monkeypatch)
    module.main()
    result = json.loads((output / "complete.json").read_text())
    assert len(polls) == 2 and len(calls) == 4
    assert result["status"] == "CONTINUATION132_EXAMS_AUDITED_NOT_PROMOTED"
    assert not result["promotion_authorized"] and not result["goal_complete"]


def test_incomplete_final_history_stops_without_restart_or_exam(tmp_path, monkeypatch):
    output, learner, calls, _ = fixture(tmp_path, monkeypatch)
    path = learner / "complete.json"
    value = json.loads(path.read_text())
    value["results"].pop()
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="fixed declared"):
        module.main()
    assert not calls
    assert json.loads((output / "failed.json").read_text())["training_restart_attempted"] is False


def test_failed_worker_cannot_be_hidden_by_a_complete_file(tmp_path, monkeypatch):
    output, _, calls, _ = fixture(tmp_path, monkeypatch)
    read = module.subprocess.check_output

    def failed(command, **kwargs):
        if command[0] == "git":
            return read(command, **kwargs)
        return "MainPID=0\nLoadState=loaded\nActiveState=failed\nSubState=failed\nExecMainStatus=1"

    monkeypatch.setattr(module.subprocess, "check_output", failed)
    with pytest.raises(ValueError, match="missing/failed"):
        module.main()
    assert not calls and (output / "failed.json").is_file()
