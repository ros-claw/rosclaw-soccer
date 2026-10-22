import json
import subprocess
import sys

import pytest

from scripts import rsi_run_motor_dagger_stage as stage


def setup_stage(tmp_path, monkeypatch, fail_at=None):
    data = tmp_path / "data"
    data.mkdir()
    (data / "complete.json").write_text(
        json.dumps(
            dict(
                source_head="datahead",
                status="DUAL_MOTOR_ACTUAL_DAGGER_SEQUENCES_READY_NOT_TRAINED",
                new_corrections=2,
                records=[{}] * 24,
                hardware_authorized=False,
                promotion_authorized=False,
            )
        )
    )
    out = tmp_path / "stage"
    args = [
        "stage",
        "--data-root",
        str(data),
        "--data-producer-head",
        "datahead",
        "--warm-start-root",
        str(tmp_path / "warm"),
        "--warm-start-hash",
        "sha256:warm",
        "--rehearsal-root",
        str(tmp_path / "rehearsal"),
        "--rehearsal-hash",
        "sha256:old",
        "--expected-head",
        "head",
        "--gpu-uuid",
        "GPU-test",
        "--output",
        str(out),
    ]
    monkeypatch.setattr(sys, "argv", args)
    monkeypatch.setattr(
        stage.subprocess, "check_output", lambda argv, **kw: b"" if "status" in argv else "head\n"
    )
    calls = []
    statuses = [
        "DUAL_RECURRENT_MOTOR_ACTUALLY_TRAINED_NOT_PHYSICALLY_QUALIFIED",
        "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
        "DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
        "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS",
    ]

    def fake_run(argv, *, env, check):
        assert check is True
        calls.append((argv, env))
        if len(calls) == fail_at:
            raise subprocess.CalledProcessError(1, argv)
        from pathlib import Path

        folder = Path(argv[argv.index("--output") + 1])
        folder.mkdir()
        (folder / "complete.json").write_text(
            json.dumps(
                dict(
                    source_head="head",
                    status=statuses[len(calls) - 1],
                    hardware_authorized=False,
                    promotion_authorized=False,
                    old_protected_retained=False,
                    consumed_run_successes=0,
                )
            )
        )

    monkeypatch.setattr(stage.subprocess, "run", fake_run)
    return out, calls


def test_all_successors_execute_without_llm_or_threshold_relaxation(tmp_path, monkeypatch):
    out, calls = setup_stage(tmp_path, monkeypatch)
    stage.main()
    assert len(calls) == 4
    assert [x[1]["CUDA_VISIBLE_DEVICES"] for x in calls] == ["GPU-test", "", "", ""]
    receipt = json.loads((out / "complete.json").read_text())
    assert receipt["old_protected_retained"] is False
    assert receipt["promotion_authorized"] is False
    assert receipt["goal_complete"] is False
    assert receipt["reasoning_wakeup_provided_by_this_program"] is False
    assert len(receipt["completed"]) == 4


@pytest.mark.parametrize("fail_at", [1, 2, 3, 4])
def test_stage_failure_stops_successors_without_fake_retry(tmp_path, monkeypatch, fail_at):
    out, calls = setup_stage(tmp_path, monkeypatch, fail_at)
    with pytest.raises(subprocess.CalledProcessError):
        stage.main()
    assert len(calls) == fail_at
    assert not (out / "complete.json").exists()
    failed = json.loads((out / "failed.json").read_text())
    assert failed["automatic_retry"] is False
    assert len(failed["completed"]) == fail_at - 1


def test_wrong_dataset_producer_rejected_before_any_update(tmp_path, monkeypatch):
    out, calls = setup_stage(tmp_path, monkeypatch)
    path = tmp_path / "data" / "complete.json"
    record = json.loads(path.read_text())
    record["source_head"] = "unexpected"
    path.write_text(json.dumps(record))
    with pytest.raises(ValueError):
        stage.main()
    assert not calls and not out.exists()
