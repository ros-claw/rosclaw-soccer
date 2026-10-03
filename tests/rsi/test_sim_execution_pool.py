import argparse
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from scripts import rsi_collect_online_step_validation as pilot
from scripts import rsi_collect_protected_phase_bank_validation as bank
from scripts.rsi_sim_execution_pool import execute_shards


def compact_worker(job):
    return {"gpu": job["gpu"], "pid": os.getpid(), "value": job["value"]}


def failed_worker(job):
    if job["gpu"] == 2:
        raise ValueError("declared failure; must not skip or retry")
    return job["gpu"]


@pytest.mark.parametrize("spawn", [False, True])
def test_all_declared_shards_return_in_order_without_parent_gpu_state(spawn):
    jobs = [dict(gpu=i, value=10 - i) for i in range(4)]
    rows = execute_shards(compact_worker, jobs, spawn=spawn)
    assert [r["gpu"] for r in rows] == list(range(4))
    assert [r["value"] for r in rows] == [10, 9, 8, 7]
    assert all((r["pid"] != os.getpid()) == spawn for r in rows)


@pytest.mark.parametrize("spawn", [False, True])
def test_one_failure_aborts_instead_of_returning_selected_successes(spawn):
    with pytest.raises(ValueError, match="must not skip"):
        execute_shards(failed_worker, [dict(gpu=i) for i in range(4)], spawn=spawn)


@pytest.mark.parametrize("gpus", [[], [0], [0, 1, 2], [0, 1, 2, 2], [1, 0, 2, 3], [0, 1, 2, True]])
def test_wrong_or_duplicate_shard_declaration_rejected_before_execution(gpus):
    with pytest.raises(ValueError):
        execute_shards(compact_worker, [dict(gpu=i) for i in gpus], spawn=True)


def test_spawn_mode_is_explicit_boolean():
    with pytest.raises(ValueError):
        execute_shards(compact_worker, [dict(gpu=i) for i in range(4)], spawn=1)


def args(tmp_path):
    return argparse.Namespace(
        output_root=tmp_path,
        isaac_python=Path("fixture-isaac"),
        g1_usd=Path("fixture-body"),
        model_root=Path("fixture-models"),
        late_swing_policy=Path("fixture-swing"),
        core_root=Path("fixture-core"),
        step_model=Path("fixture-warm"),
        online_model=Path("fixture-candidate"),
        warm_model=Path("fixture-warm"),
        candidate_model=Path("fixture-candidate"),
        resume=False,
        compressed_reports=True,
        shared_model_reports=True,
    )


def reference():
    return dict(
        body_trace_hash="body",
        trace_hash="ball",
        asset_hash="asset",
        sonic_qualification_hash="sonic",
    )


def fake_outcome():
    return dict(
        clean_foot_only=True, forward_60_m=2.0, lateral_60_m=0.1, maximum_lateral_excursion_m=0.2
    )


def test_pilot_worker_keeps_three_same_controlled_arms_and_gpu(monkeypatch, tmp_path):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(report_hash=kwargs["arm"], **reference()), fake_outcome()

    monkeypatch.setattr(pilot, "_run", run)
    result = pilot.execute_online_course(
        dict(args=args(tmp_path), runner=Path("fixture-runner"), gpu=2, reference=reference())
    )
    assert [r["arm"] for r in calls] == ["reproduction", "warm", "online"]
    assert all(r["gpu"] == 2 and r["gain"] == 1.2 and r["negative_only"] for r in calls)
    assert all(r["compressed_report"] and r["shared_model_report"] for r in calls)
    assert (result["seed"], result["lane"]) == pilot.COURSES[2]
    assert calls[1]["motor_step"] == Path("fixture-warm")
    assert calls[2]["motor_step"] == Path("fixture-candidate")
    assert calls[1]["parent_report_override"] == calls[2]["parent_report_override"]


def test_changed_baseline_still_rejects_before_candidate(monkeypatch, tmp_path):
    calls = []

    def run(**kwargs):
        calls.append(kwargs["arm"])
        return dict(report_hash=kwargs["arm"], **reference()), fake_outcome()

    monkeypatch.setattr(pilot, "_run", run)
    old = reference()
    old["body_trace_hash"] = "changed"
    with pytest.raises(ValueError, match="frozen warm-start"):
        pilot.execute_online_course(
            dict(args=args(tmp_path), runner=Path("runner"), gpu=0, reference=old)
        )
    assert calls == ["reproduction", "warm"]


def test_bank_worker_keeps_every_course_and_timeout_and_does_not_hide_failure(
    monkeypatch, tmp_path
):
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        outcome = fake_outcome()
        if kwargs["arm"] == "candidate" and kwargs["seed"] == 106:
            outcome["clean_foot_only"] = False
        return dict(report_hash=kwargs["arm"], **reference()), outcome

    monkeypatch.setattr(bank, "_run", run)
    monkeypatch.setattr(bank, "_sealed", lambda _: reference())
    courses = [dict(seed=100 + i, lane=0, parent_report="fixture-parent") for i in range(52)]
    rows = bank.execute_bank_shard(
        dict(args=args(tmp_path), runner=Path("runner"), gpu=2, courses=courses, reuse=False)
    )
    assert [r["index"] for r in rows] == list(range(2, 52, 4))
    assert len(calls) == 39
    assert all(r["gpu"] == 2 and r["execution_timeout_s"] == 600 for r in calls)
    assert all(r["gain"] == 1.2 and r["negative_only"] for r in calls)
    assert [r["arm"] for r in calls[:3]] == ["reproduction", "warm", "candidate"]
    assert rows[1]["candidate"]["high_quality"] is False
    assert (tmp_path / "row-6.json").is_file()


def test_failed_native_log_cannot_be_overwritten_by_new_worker(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / "seed100-lane0-candidate-actor.log").write_text("native failed")
    with pytest.raises(ValueError, match="not overwrite"):
        bank.preserve_failed_attempt(tmp_path, 100, 0, "candidate", "actor")


def test_bank_exception_is_durable_before_ordered_pool_returns(monkeypatch, tmp_path):
    original = TimeoutError("native lifecycle timeout; no implicit retry")
    calls = []

    def failed(job):
        calls.append(job["gpu"])
        raise original

    monkeypatch.setattr(bank, "_execute_bank_shard", failed)
    (tmp_path / "row-2.json").write_text("partial recorded row, not an audit")
    job = dict(args=args(tmp_path), gpu=2)
    for _ in range(2):
        with pytest.raises(TimeoutError) as raised:
            bank.execute_bank_shard(job)
        assert raised.value is original
    artifacts = list(tmp_path.glob("shard-2-failure-*.json"))
    assert len(artifacts) == 2
    assert calls == [2, 2]  # Only the explicit invocations; no retry in either.
    for path in artifacts:
        record = json.loads(path.read_text())
        assert record["exception_type"] == "builtins.TimeoutError"
        assert "native lifecycle timeout" in record["traceback"]
        assert record["completed_row_indices"] == [2]
        assert record["completed_paths_are_not_independent_verification"] is True
        assert record["automatic_retry"] is False
        assert record["promotion_authorized"] is record["hardware_authorized"] is False


def test_journal_disk_failure_does_not_mask_original_exception(monkeypatch, tmp_path):
    original = ValueError("original source binding failure")

    def failed(job):
        raise original

    def disk_failed(*args):
        raise OSError("disk full")

    monkeypatch.setattr(bank, "_execute_bank_shard", failed)
    monkeypatch.setattr(bank, "write_once", disk_failed)
    with pytest.raises(ValueError) as raised:
        bank.execute_bank_shard(dict(args=args(tmp_path), gpu=0))
    assert raised.value is original
    assert "disk full" in original.__notes__[0]


def test_failed_later_shard_is_visible_while_first_shard_is_still_busy(monkeypatch, tmp_path):
    release = threading.Event()
    published = threading.Event()
    original_write = bank.write_once

    def work(job):
        if job["gpu"] == 0 and not release.wait(timeout=10):
            raise RuntimeError("fixture waiter timed out")
        if job["gpu"] == 1:
            raise TimeoutError("later shard failed first")
        return []

    def observe_write(path, value):
        original_write(path, value)
        published.set()

    monkeypatch.setattr(bank, "_execute_bank_shard", work)
    monkeypatch.setattr(bank, "write_once", observe_write)
    jobs = [dict(args=args(tmp_path), gpu=i) for i in range(4)]
    with ThreadPoolExecutor(max_workers=1) as coordinator:
        result = coordinator.submit(execute_shards, bank.execute_bank_shard, jobs, spawn=False)
        try:
            assert published.wait(timeout=10)
            assert not result.done()
            record = json.loads(next(tmp_path.glob("shard-1-failure-*.json")).read_text())
            assert record["exception_message"] == "later shard failed first"
        finally:
            release.set()
        with pytest.raises(TimeoutError, match="later shard"):
            result.result(timeout=10)
