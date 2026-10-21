"""Queue flow fixtures only; these tests do not claim physics or CUDA success."""

import json
import sys
from pathlib import Path

import pytest

from scripts import rsi_continue_paid_successor_learning as queue
from scripts.rsi_native_context_recovery_common import digest


def fixture(
    tmp_path,
    monkeypatch,
    *,
    incomplete=False,
    empty_child=False,
    changed_source=False,
    progressive=False,
):
    data = tmp_path / "data"
    data.mkdir()
    (data / "complete.json").write_text(
        json.dumps(
            dict(
                source_head="data-head",
                status=(
                    "PAID_PROGRESSIVE_TRAIN_PARENT_FEATURES_CAUSALLY_RECOMPUTED"
                    if progressive
                    else "PAID_SUCCESSOR_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"
                ),
                curriculum_protocol="successor-six-teacher-v2",
            )
        )
    )
    output = tmp_path / "queue"
    args = [
        "queue",
        "--data-root",
        str(data),
        "--output",
        str(output),
        "--data-source-head",
        "data-head",
        "--expected-head",
        "source-head",
        "--seed",
        "0",
        "--gpu-uuid",
        "GPU-fixture",
    ]
    for name in ("train-root", "train-audit-root", "dev-root", "dev-audit-root"):
        args.extend(["--" + name, str(tmp_path / name)])
    if progressive:
        args.append("--progressive")
    monkeypatch.setattr(sys, "argv", args)
    state = {"mutated": False}

    def check(command):
        if "status" in command:
            return b""
        return ("changed" if state["mutated"] else "source-head") + "\n"

    # The runner requests bytes for status and text for rev-parse.
    monkeypatch.setattr(queue.subprocess, "check_output", lambda command, **_: check(command))
    calls = []
    training_status = "PAID_SUCCESSOR_NEURAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
    progressive_status = "PAID_PROGRESSIVE_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED"
    exam_status = "PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT"
    audit_status = "PAID_SUCCESSOR_STUDENT_ALL_NATIVE_PRIMITIVES_AND_OWN_NEURAL_AUDIT_PASS"
    statuses = {
        "scripts.rsi_train_paid_successor_student": training_status,
        "scripts.rsi_train_paid_progressive_student": progressive_status,
        "scripts.rsi_exam_paid_successor_student": exam_status,
        "scripts.rsi_audit_paid_successor_student": audit_status,
    }

    def child(command, **kwargs):
        calls.append((command, kwargs["env"].copy()))
        if empty_child:
            return
        module = command[command.index("-m") + 1]
        path = Path(command[command.index("--output") + 1])
        path.mkdir()
        result = dict(
            status=statuses[module],
            source_head="source-head",
            hardware_authorized=False,
            promotion_authorized=False,
        )
        if module in (
            "scripts.rsi_train_paid_successor_student",
            "scripts.rsi_train_paid_progressive_student",
        ):
            result.update(data_hash=digest(data / "complete.json"), config=dict(seed_index=0))
        (path / "complete.json").write_text(json.dumps(result))
        if changed_source:
            state["mutated"] = True

    monkeypatch.setattr(queue.subprocess, "run", child)
    if incomplete:
        (output / "trained").mkdir(parents=True)
    return calls, output


def test_fixed_learning_queue_chains_five_stages_and_never_promotes(tmp_path, monkeypatch):
    calls, output = fixture(tmp_path, monkeypatch)
    queue.main()
    assert len(calls) == 5
    assert [env["CUDA_VISIBLE_DEVICES"] for _, env in calls] == ["GPU-fixture", "", "", "", ""]
    assert all(c[0][0] == sys.executable and c[0][1:3] == ["-u", "-m"] for c in calls)
    final = json.loads((output / "complete.json").read_text())
    assert not final["goal_complete"] and not final["promotion_authorized"]
    queue.main()  # Revalidates closed stages, no duplicate child allocations.
    assert len(calls) == 5


def test_incomplete_stage_is_preserved_without_automatic_retry(tmp_path, monkeypatch):
    calls, output = fixture(tmp_path, monkeypatch, incomplete=True)
    with pytest.raises(ValueError, match="incomplete/ambiguous"):
        queue.main()
    assert not calls and (output / "trained").is_dir()


def test_exit_zero_without_evidence_cannot_unlock_physics(tmp_path, monkeypatch):
    calls, output = fixture(tmp_path, monkeypatch, empty_child=True)
    with pytest.raises(FileNotFoundError):
        queue.main()
    assert len(calls) == 1 and (output / "trained-started.json").exists()


def test_source_mutation_stops_before_allocating_next_stage(tmp_path, monkeypatch):
    calls, output = fixture(tmp_path, monkeypatch, changed_source=True)
    with pytest.raises(ValueError, match="immutable"):
        queue.main()
    assert len(calls) == 1 and (output / "trained" / "complete.json").exists()


def test_progressive_queue_uses_new_curriculum_and_same_full_physics_audits(tmp_path, monkeypatch):
    calls, output = fixture(tmp_path, monkeypatch, progressive=True)
    queue.main()
    assert len(calls) == 5
    assert "scripts.rsi_train_paid_progressive_student" in calls[0][0]
    assert "--staged" not in calls[0][0] and "--decoupled" not in calls[0][0]
    assert json.loads((output / "declaration.json").read_text())["progressive"] is True
    queue.main()
    assert len(calls) == 5
