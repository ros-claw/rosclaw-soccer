import json

import pytest

from scripts import rsi_watch_research_loop as watcher


def pointer(tmp_path, campaigns=None):
    return dict(
        schema="soccer.agent.resume_pointer.v1",
        goal_status="active",
        observation_campaigns=campaigns
        if campaigns is not None
        else [dict(unit="closed.service", root=str(tmp_path))],
    )


def test_pointer_refresh_observes_successor_without_resetting_idle(tmp_path, monkeypatch, capsys):
    path, output = tmp_path / "next.json", tmp_path / "observer.json"
    path.write_text(json.dumps(pointer(tmp_path)))
    monkeypatch.setattr(
        "sys.argv",
        ["watch", "--resume-pointer", str(path), "--output", str(output)],
    )
    observed = []

    def collect(unit, root):
        observed.append((unit, root))
        return dict(
            unit=unit,
            service=dict(LoadState="loaded", MainPID="0", ActiveState="inactive", Result="success"),
            artifacts=dict(tip="closed", complete_present=True),
        )

    monkeypatch.setattr(watcher, "collect", collect)
    monkeypatch.setattr(watcher.time, "time", lambda: 100)
    watcher.main()
    first = json.loads(output.read_text())
    # Atomic replacement, as used by the real checkpoint publisher.
    replacement = tmp_path / "replacement.json"
    replacement.write_text(
        json.dumps(pointer(tmp_path, [dict(unit="audit.service", root=str(tmp_path / "audit"))]))
    )
    replacement.replace(path)
    monkeypatch.setattr(watcher.time, "time", lambda: 401)
    watcher.main()
    second = json.loads(output.read_text())
    assert observed == [("closed.service", tmp_path), ("audit.service", tmp_path / "audit")]
    assert first["resume_pointer_hash"] != second["resume_pointer_hash"]
    assert second["continuity"]["no_live_since_unix"] == 100
    assert second["continuity"]["successor_overdue"]
    assert second["dispatch"]["state"] == "AGENT_SUCCESSOR_REQUIRED"
    assert not second["resume_pointer_is_scientific_progress"]
    assert not second["agent_wakeup_provided_by_this_script"]
    assert not second["scientific_successor_automated"]
    assert not second["goal_complete"]
    assert len(capsys.readouterr().out.splitlines()) == 2


@pytest.mark.parametrize(
    "defect",
    [
        dict(schema="other"),
        dict(goal_status="complete"),
        dict(observation_campaigns=[dict(unit="bad.service; touch x", root="/code/evidence")]),
        dict(observation_campaigns=[dict(unit="--help.service", root="/code/evidence")]),
        dict(observation_campaigns=[dict(unit="a.service", root="../evidence")]),
        dict(observation_campaigns=[dict(unit="a.service", root="/code/../etc")]),
        dict(observation_campaigns=[dict(unit="a.service", root="/code", command="run")]),
        dict(observation_campaigns=[dict(unit="a.service", root=1)]),
        dict(observation_campaigns=[dict(unit="a.service", root="/code")] * 2),
        dict(observation_campaigns=[dict(unit=f"a{i}.service", root="/code") for i in range(129)]),
    ],
)
def test_invalid_pointer_rejected_without_observing_or_launching(tmp_path, defect):
    path = tmp_path / "next.json"
    path.write_text(json.dumps(pointer(tmp_path) | defect))
    with pytest.raises(ValueError):
        watcher.resume_campaigns(path)


def test_bounded_ordinary_file(tmp_path):
    path = tmp_path / "next.json"
    path.write_text(" " * 262145)
    with pytest.raises(ValueError):
        watcher.resume_campaigns(path)
    link = tmp_path / "link.json"
    link.symlink_to(path)
    with pytest.raises(ValueError):
        watcher.resume_campaigns(link)


def test_empty_queue_is_observable_as_idle_without_resetting_deadline(tmp_path, monkeypatch):
    path, output = tmp_path / "next.json", tmp_path / "observer.json"
    path.write_text(json.dumps(pointer(tmp_path, [])))
    monkeypatch.setattr(
        "sys.argv", ["watch", "--resume-pointer", str(path), "--output", str(output)]
    )
    monkeypatch.setattr(watcher.time, "time", lambda: 100)
    watcher.main()
    monkeypatch.setattr(watcher.time, "time", lambda: 401)
    watcher.main()
    report = json.loads(output.read_text())
    assert report["campaigns"] == {}
    assert report["dispatch"]["state"] == "AGENT_SUCCESSOR_REQUIRED"
    assert report["continuity"]["no_live_since_unix"] == 100
    assert report["continuity"]["successor_overdue"]
    assert not report["agent_wakeup_provided_by_this_script"]
