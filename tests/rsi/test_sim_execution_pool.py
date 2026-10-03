import argparse
import os
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
