import copy
import sqlite3

import pytest

from rosclaw_soccer.rsi.research_loop_watch import assess, collect


def snapshot():
    return dict(
        service=dict(LoadState="loaded", ActiveState="active", MainPID="12", Result="success"),
        cpu_ticks=100,
        process_start_ticks=99,
        ledger=dict(tip="a", seq=1, phase="SIMULATING", completed=0),
    )


def test_running_is_not_completion_or_training():
    result = assess(snapshot(), None, now=100)
    assert result["phase"] == "RUNNING"
    for name in (
        "goal_complete",
        "outcome_verified",
        "ledger_integrity_verified",
        "automatic_retry",
    ):
        assert result[name] is False


def test_transient_observer_failure_does_not_poison_subsequent_observation():
    current = snapshot()
    current["ledger"] = None
    result = assess(current, {"phase": "OBSERVATION_FAILED"}, now=100)
    assert result["phase"] == "RUNNING"


def test_cpu_activity_does_not_hide_absence_of_scientific_commits():
    current = snapshot()
    previous = assess(current, None, now=100)
    current["cpu_ticks"] += 100
    result = assess(current, previous, now=1400)
    assert result["phase"] == "COMPUTING_WITHOUT_NEW_COMMIT"
    assert result["last_ledger_change_unix"] == 100
    assert result["last_cpu_change_unix"] == 1400


def test_idle_requires_investigation_not_automatic_physics_retry():
    current = snapshot()
    previous = assess(current, None, now=100)
    result = assess(current, previous, now=1400)
    assert result["phase"] == "NO_OBSERVED_CPU_PROGRESS"
    assert result["automatic_retry"] is False


@pytest.mark.parametrize("completed", [0, 3, 44])
def test_terminal_worker_always_leaves_next_agent_work(completed):
    current = snapshot()
    current["service"].update(ActiveState="inactive", MainPID="0")
    current["ledger"].update(
        completed=completed, phase="PILOT_COMPLETE_EXPANDED_VALIDATION_REQUIRED"
    )
    result = assess(current, None, now=100)
    assert result["next_agent_action"] == "REVIEW_LEDGER_AND_DECLARE_NEXT_SCIENTIFIC_STAGE"
    assert result["goal_complete"] is False


def test_failure_is_not_successful_idle_or_a_silent_restart():
    current = snapshot()
    current["service"].update(ActiveState="failed", MainPID="0", Result="exit-code")
    assert assess(current, None, now=100)["phase"] == "FAILED"


def test_pid_reuse_and_clock_rollback_do_not_fake_stall():
    current = snapshot()
    previous = assess(current, None, now=100)
    current["process_start_ticks"] += 1
    assert assess(current, previous, now=1400)["phase"] == "COMPUTING_WITHOUT_NEW_COMMIT"
    assert assess(current, previous, now=50)["phase"] == "RUNNING"


def test_new_ledger_tip_is_observed_without_promoting_evidence():
    current = snapshot()
    previous = assess(copy.deepcopy(current), None, now=100)
    current["ledger"]["tip"] = "b"
    current["cpu_ticks"] += 1
    result = assess(current, previous, now=1400)
    assert result["phase"] == "RUNNING"
    assert not result["ledger_integrity_verified"]


@pytest.mark.parametrize("clock", [float("nan"), float("inf")])
def test_finite_clock(clock):
    with pytest.raises(ValueError):
        assess(snapshot(), None, now=clock)


def test_collect_uses_read_only_sqlite_and_does_not_create_missing_ledger(tmp_path, monkeypatch):
    from types import SimpleNamespace

    def run(command, **kwargs):
        assert command[:3] == ["systemctl", "--user", "show"]
        assert kwargs["check"]
        return SimpleNamespace(
            stdout="LoadState=loaded\nMainPID=0\nActiveState=inactive\nResult=success\n"
        )

    monkeypatch.setattr("rosclaw_soccer.rsi.research_loop_watch.subprocess.run", run)
    assert collect("example.service", tmp_path)["ledger"] is None
    assert not list(tmp_path.iterdir())
    with sqlite3.connect(tmp_path / "campaign.sqlite") as db:
        db.execute("CREATE TABLE events (seq, digest, body)")
        db.execute(
            "INSERT INTO events VALUES (1, 'a', ?)",
            ('{"state":{"phase":"INIT","attempts":0,"completed":[],"active":null}}',),
        )
    data = (tmp_path / "campaign.sqlite").read_bytes()
    assert collect("example.service", tmp_path)["ledger"]["completed"] == 0
    assert (tmp_path / "campaign.sqlite").read_bytes() == data


@pytest.mark.parametrize("failure_name", ["failure.json", "failed.json"])
def test_serial_checkpoints_are_progress_not_verified_outcomes(tmp_path, monkeypatch, failure_name):
    from types import SimpleNamespace

    monkeypatch.setattr(
        "rosclaw_soccer.rsi.research_loop_watch.subprocess.run",
        lambda *a, **kw: SimpleNamespace(
            stdout="LoadState=loaded\nMainPID=0\nActiveState=inactive\nResult=success\n"
        ),
    )
    (tmp_path / "declaration.json").write_text("{}")
    first = collect("example.service", tmp_path)
    assert first["artifacts"]["checkpoint_count"] == 1
    previous = assess(first, None, now=100)
    (tmp_path / "complete.json").write_text("not even valid json; observer must not promote it")
    second = collect("example.service", tmp_path)
    result = assess(second, previous, now=1400)
    assert result["phase"] == "STAGE_ARTIFACT_READY_FOR_READBACK"
    assert result["last_ledger_change_unix"] == 1400
    assert not result["outcome_verified"] and not result["goal_complete"]
    assert not result["artifacts"]["contents_verified"]
    (tmp_path / failure_name).write_text("{}")
    assert assess(collect("example.service", tmp_path), result, now=1401)["phase"] == "FAILED"


def test_artifact_completion_does_not_override_active_worker_or_failed_exit():
    current = snapshot()
    current["artifacts"] = dict(tip="synthetic", complete_present=True, failure_present=False)
    assert assess(current, None, now=100)["phase"] == "RUNNING"
    current["service"].update(ActiveState="inactive", MainPID="0", Result="exit-code")
    assert assess(current, None, now=101)["phase"] == "FAILED"


def test_garbage_collected_oneshot_with_artifact_requires_history_not_relaunch():
    current = snapshot()
    current["service"].update(LoadState="not-found", ActiveState="inactive", MainPID="0")
    current["artifacts"] = dict(tip="synthetic", complete_present=True, failure_present=False)
    result = assess(current, None, now=100)
    assert result["phase"] == "UNLOADED_STAGE_ARTIFACT_REQUIRES_HISTORY_CHECK"
    assert result["next_agent_action"].endswith("NO_RELAUNCH")
    assert not result["automatic_retry"] and not result["outcome_verified"]


def test_activating_oneshot_with_actual_pid_is_running_not_only_transitioning():
    current = snapshot()
    current["service"]["ActiveState"] = "activating"
    assert assess(current, None, now=100)["phase"] == "RUNNING"


def test_waiting_launcher_child_cpu_is_observed_without_faking_scientific_progress():
    current = snapshot()
    current["service"].update(ActiveState="activating", CPUUsageNSec="1000000000")
    previous = assess(copy.deepcopy(current), None, now=100)
    current["service"]["CPUUsageNSec"] = "2000000000"
    result = assess(current, previous, now=1400)
    assert current["cpu_ticks"] == previous["cpu_ticks"]
    assert result["whole_unit_cpu_advanced_since_previous_observation"]
    assert result["cpu_advanced_since_previous_observation"]
    assert result["phase"] == "COMPUTING_WITHOUT_NEW_COMMIT"
    assert not result["goal_complete"] and not result["automatic_retry"]


def test_unit_cpu_accounting_does_not_confuse_pid_reuse_with_same_invocation():
    current = snapshot()
    current["service"]["CPUUsageNSec"] = "1000"
    previous = assess(copy.deepcopy(current), None, now=100)
    current["process_start_ticks"] += 1
    current["service"]["CPUUsageNSec"] = "100000"
    result = assess(current, previous, now=101)
    assert not result["whole_unit_cpu_advanced_since_previous_observation"]


def test_loaded_waiting_job_is_not_missing_work_or_completion():
    current = snapshot()
    current["service"].update(ActiveState="inactive", MainPID="0", Job="8531938")
    result = assess(current, None, now=100)
    assert result["phase"] == "QUEUED_DECLARED_JOB"
    assert result["next_agent_action"] == "VERIFY_PREDECESSOR_NO_DUPLICATE_LAUNCH"
    assert not result["automatic_retry"] and not result["goal_complete"]
