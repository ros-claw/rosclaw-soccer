import pytest

from scripts import rsi_validate_consolidated_parent_successes as experiment


def test_selection_covers_all_new_parent_successes_not_rejected_child_failures(monkeypatch):
    monkeypatch.setattr(experiment, "qualified_memory_failure_rows", lambda *a: None)
    rows = [
        dict(seed=i, lane=0, warm=dict(high_quality=i == 0), candidate=dict(high_quality=i < 5))
        for i in range(7)
    ]
    assert experiment.newly_successful_rows(dict(rows=rows), {}) == rows[1:5]


def test_too_many_courses_are_rejected_not_silently_truncated(monkeypatch):
    monkeypatch.setattr(experiment, "qualified_memory_failure_rows", lambda *a: None)
    rows = [
        dict(warm=dict(high_quality=False), candidate=dict(high_quality=True)) for _ in range(5)
    ]
    with pytest.raises(ValueError, match="never truncate"):
        experiment.newly_successful_rows(dict(rows=rows), {})


def test_parent_rejection_blocks_course_selection(monkeypatch):
    def reject(*a):
        raise ValueError("new out of play")

    monkeypatch.setattr(experiment, "qualified_memory_failure_rows", reject)
    with pytest.raises(ValueError, match="out of play"):
        experiment.newly_successful_rows({}, {})


def row(hq=True, clean=True, lateral=3.0, z=0.7):
    return dict(
        high_quality=hq,
        clean_foot_only=clean,
        maximum_lateral_excursion_m=lateral,
        minimum_pelvis_z_m=z,
    )


def test_counts_report_restoration_without_calling_it_global_growth():
    measured = experiment.counts(
        [dict(qualified=row(), rejected=row(hq=False), consolidated=row())]
    )
    assert measured["qualified_high_quality"] == measured["consolidated_high_quality"] == 1
    assert measured["rejected_high_quality"] == 0
    assert measured["old_high_quality_loss"] == measured["new_out_of_play"] == 0
    assert "full_bank_passed" not in measured


def test_local_improvement_cannot_hide_new_out_or_loss():
    measured = experiment.counts(
        [dict(qualified=row(), rejected=row(hq=False), consolidated=row(hq=False, lateral=4.1))]
    )
    assert measured["old_high_quality_loss"] == measured["new_out_of_play"] == 1


def test_changed_historical_parent_blocks_every_actor_before_execution(tmp_path, monkeypatch):
    job = dict(
        row=dict(seed=42, lane=0),
        runner="fixture-runner",
        gpu=0,
        args=dict(
            output_root=tmp_path,
            parent_bank_root=tmp_path,
            isaac_python="fixture",
            g1_usd="fixture",
            model_root="fixture",
            late_swing_policy="fixture",
            core_root="fixture",
        ),
    )
    calls = []

    def run(**kwargs):
        calls.append(kwargs)
        return dict(body_trace_hash="new", trace_hash="same"), {}

    monkeypatch.setattr(experiment, "_run", run)
    monkeypatch.setattr(
        experiment, "_sealed", lambda _: dict(body_trace_hash="old", trace_hash="same")
    )
    with pytest.raises(ValueError, match="changed frozen parent physics"):
        experiment.execute_course(job)
    assert len(calls) == 1
