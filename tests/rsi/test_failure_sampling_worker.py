from pathlib import Path

import pytest

from scripts import rsi_failure_sampling_worker as worker


def make_job(tmp_path, gpu=0):
    (tmp_path / "logs").mkdir()
    return dict(
        gpu=gpu,
        arm="candidate",
        runner=Path("fixture-only.py"),
        courses=[dict(seed=100, lane=0, candidate=dict(report_hash="old-candidate"))],
        view_hashes=[f"view-{i}" for i in range(4)],
        args=dict(
            output_root=tmp_path,
            isaac_python=Path("not-executed"),
            g1_usd=Path("fixture.usd"),
            model_root=Path("fixture-models"),
            late_swing_policy=Path("fixture.json"),
            core_root=Path("fixture-core"),
            resume=False,
            warm_model=Path("fixture-mean.json"),
            bank_physics_root=Path("fixture-bank"),
            samples_per_course=4,
            compressed_sampling_models=True,
        ),
    )


def install_fixture(monkeypatch, *, corrupt=False):
    calls = []
    physical = dict(
        body_trace_hash="body",
        trace_hash="ball",
        asset_hash="asset",
        sonic_qualification_hash="sonic",
    )

    def sealed(path):
        return {**physical, "report_hash": "old-candidate"}

    def run(**kwargs):
        calls.append(kwargs)
        raw = {**physical, "report_hash": kwargs["arm"]}
        if corrupt and kwargs["arm"] == "greedy":
            raw["body_trace_hash"] = "changed-body"
        measured = dict(
            clean_foot_only=True,
            forward_60_m=1.5,
            lateral_60_m=0.15,
            lateral_over_forward_60=0.1,
            maximum_lateral_excursion_m=0.2,
        )
        return raw, measured

    monkeypatch.setattr(worker, "_sealed", sealed)
    monkeypatch.setattr(worker, "_run", run)
    return calls


def test_worker_preserves_all_ordered_samples_gpu_and_compressed_identity(monkeypatch, tmp_path):
    job = make_job(tmp_path)
    calls = install_fixture(monkeypatch)
    rows = worker.run_failure_worker(job)
    assert [c["arm"] for c in calls] == [
        "reproduction",
        "greedy",
        "sample-0",
        "sample-1",
        "sample-2",
        "sample-3",
    ]
    assert all(c["gpu"] == 0 for c in calls)
    assert [r["view_hash"] for r in rows[0]["samples"]] == job["view_hashes"]
    assert all(str(c["motor_step"]).endswith(".json.gz") for c in calls[2:])
    assert (tmp_path / "row-0.json").is_file()


def test_parent_body_identity_mismatch_blocks_every_exploratory_execution(monkeypatch, tmp_path):
    job = make_job(tmp_path)
    calls = install_fixture(monkeypatch, corrupt=True)
    with pytest.raises(ValueError, match="physical traces"):
        worker.run_failure_worker(job)
    assert len(calls) == 2
    assert not (tmp_path / "row-0.json").exists()


def test_shared_worker_uses_complete_compressed_reports_and_resolved_parent(monkeypatch, tmp_path):
    job = make_job(tmp_path)
    job["args"]["shared_sampling_models"] = True
    calls = install_fixture(monkeypatch)
    monkeypatch.setattr(worker, "resolve_physical_report", lambda p: p.with_name("report.json.gz"))
    worker.run_failure_worker(job)
    assert all(c["compressed_report"] and c["shared_model_report"] for c in calls)
    assert all(c["parent_report_override"].name == "report.json.gz" for c in calls[1:])


def test_failed_log_is_preserved_before_any_retry(monkeypatch, tmp_path):
    job = make_job(tmp_path)
    calls = install_fixture(monkeypatch)
    log = tmp_path / "logs/seed100-lane0-sample-0-actor.log"
    log.write_text("failure evidence")
    with pytest.raises(ValueError, match="explicitly archive"):
        worker.run_failure_worker(job)
    assert not calls
    assert log.read_text() == "failure evidence"


def test_invalid_gpu_boolean_is_rejected_before_execution(monkeypatch, tmp_path):
    calls = install_fixture(monkeypatch)
    with pytest.raises(ValueError, match="bounded declared"):
        worker.run_failure_worker(make_job(tmp_path, gpu=True))
    assert not calls
