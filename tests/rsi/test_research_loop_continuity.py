import pytest

from rosclaw_soccer.rsi.research_loop_continuity import continuity_state


def dispatch(idle=True):
    return dict(no_live_declared_work=idle, live_units=[] if idle else ["job.service"])


def test_observer_heartbeats_do_not_hide_idle_queue():
    first = continuity_state(dispatch(), None, now=100)
    second = continuity_state(dispatch(), first, now=399)
    third = continuity_state(dispatch(), second, now=400)
    assert first["no_live_since_unix"] == second["no_live_since_unix"] == 100
    assert not second["successor_overdue"]
    assert third["successor_overdue"]
    assert third["state"] == "SCIENTIFIC_SUCCESSOR_OVERDUE"
    assert third["agent_attention_required"]
    for key in (
        "observer_tick_is_scientific_progress",
        "agent_wakeup_provided",
        "automatic_physics_retry",
        "goal_complete",
    ):
        assert third[key] is False


def test_real_live_work_resets_idle_clock_not_goal():
    first = continuity_state(dispatch(), None, now=100)
    live = continuity_state(dispatch(False), first, now=500)
    again = continuity_state(dispatch(), live, now=600)
    assert live["no_live_since_unix"] is None
    assert not live["goal_complete"]
    assert again["no_live_since_unix"] == 600


def test_live_job_does_not_hide_dispatch_review():
    current = dispatch(False) | dict(agent_attention_required=True)
    assert continuity_state(current, None, now=100)["agent_attention_required"]


def test_clock_rollback_is_explicit_not_fake_elapsed_time():
    first = continuity_state(dispatch(), None, now=100)
    second = continuity_state(dispatch(), first, now=50)
    assert second["state"] == "CLOCK_REVIEW_REQUIRED"
    assert second["idle_seconds"] == 0
    assert not second["successor_overdue"]


@pytest.mark.parametrize("now", [True, float("nan"), float("inf")])
def test_invalid_clock_rejected(now):
    with pytest.raises(ValueError):
        continuity_state(dispatch(), None, now=now)


@pytest.mark.parametrize("defect", [dict(no_live_declared_work=1), dict(live_units=["x"])])
def test_inconsistent_dispatch_rejected(defect):
    with pytest.raises(ValueError):
        continuity_state(dispatch() | defect, None, now=100)


@pytest.mark.parametrize(
    "prior",
    [
        dict(observed_unix=float("nan")),
        dict(no_live_since_unix=1),
        dict(no_live_since_unix=2, observed_unix=1),
    ],
)
def test_invalid_prior_clock_rejected(prior):
    with pytest.raises(ValueError):
        continuity_state(dispatch(), prior, now=100)


def test_real_watcher_persists_idle_deadline_across_atomic_reports(tmp_path, monkeypatch, capsys):
    import json

    from scripts import rsi_watch_research_loop as watcher

    output = tmp_path / "watch.json"
    monkeypatch.setattr(
        "sys.argv",
        ["watch", "--campaign", "closed.service", str(tmp_path), "--output", str(output)],
    )
    monkeypatch.setattr(
        watcher,
        "collect",
        lambda *args: dict(
            service=dict(LoadState="loaded", MainPID="0", ActiveState="active", Result="success"),
            artifacts=dict(tip="closed", complete_present=True),
        ),
    )
    monkeypatch.setattr(watcher.time, "time", lambda: 100)
    watcher.main()
    first = json.loads(output.read_text())
    monkeypatch.setattr(watcher.time, "time", lambda: 401)
    watcher.main()
    second = json.loads(output.read_text())
    assert first["continuity"]["no_live_since_unix"] == 100
    assert second["continuity"]["idle_seconds"] == 301
    assert second["continuity"]["successor_overdue"]
    assert not second["agent_wakeup_provided_by_this_script"]
    assert not second["goal_complete"]
    assert len(capsys.readouterr().out.splitlines()) == 2
