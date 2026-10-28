import json
from types import SimpleNamespace

import pytest

from scripts import rsi_watch_native_pass_audit as watch


def setup(monkeypatch, tmp_path, *, state, complete):
    source = tmp_path / "source"
    source.mkdir()
    if complete:
        (source / "complete.json").write_text(
            json.dumps(
                dict(
                    status="CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
                    source_head="train",
                    records=[{}] * 32,
                    promotion_authorized=False,
                )
            )
        )
    monkeypatch.setattr(watch, "clean", lambda _: None)
    monkeypatch.setattr(
        watch.sys,
        "argv",
        [
            "watch",
            "--source",
            str(source),
            "--output",
            str(tmp_path / "output"),
            "--unit",
            "rosclaw-team-sweep.service",
            "--source-head",
            "train",
            "--expected-head",
            "audit",
        ],
    )
    calls = []

    def run(cmd, **kwargs):
        calls.append(cmd)
        return SimpleNamespace(stdout=state, returncode=0)

    monkeypatch.setattr(watch.subprocess, "run", run)
    return calls


def test_complete_sweep_replays(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    watch.main()
    assert len(calls) == 2
    assert calls[1][1] == "scripts/rsi_audit_native_receiving_roles.py"


def test_failed_sweep_does_not_retry(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=failed\nResult=exit-code\nExecMainStatus=1\n",
        complete=False,
    )
    with pytest.raises(ValueError, match="without completion"):
        watch.main()
    assert len(calls) == 1


def test_priority_sweep_replays_only_matching_protocol(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    source.write_text(
        json.dumps(
            dict(
                status="CAPTURE_PRIORITY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY",
                source_head="train",
                records=[{}] * 32,
                promotion_authorized=False,
            )
        )
    )
    watch.sys.argv.append("--priority")
    watch.main()
    assert len(calls) == 2


def test_bilateral_sweep_replays_only_matching_protocol(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    report = json.loads(source.read_text())
    report["status"] = "CAPTURE_BILATERAL_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
    source.write_text(json.dumps(report))
    watch.sys.argv.append("--bilateral")
    watch.main()
    assert len(calls) == 2


def test_missing_unit_state_fails_closed(monkeypatch, tmp_path):
    calls = setup(monkeypatch, tmp_path, state="", complete=True)
    with pytest.raises(ValueError, match="cannot verify"):
        watch.main()
    assert len(calls) == 1


def test_long_horizon_sweep_replays_only_matching_protocol(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    report = json.loads(source.read_text())
    report["status"] = "CAPTURE_LONG_HORIZON_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
    source.write_text(json.dumps(report))
    watch.sys.argv.append("--long-horizon")
    watch.main()
    assert len(calls) == 2


def test_development_relay_routes_to_relay_replay(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    report = json.loads(source.read_text())
    report["status"] = "NEURAL_RELAY_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
    source.write_text(json.dumps(report))
    watch.sys.argv.append("--relay-development")
    watch.main()
    assert calls[1][1] == "scripts/rsi_audit_neural_relay_chain.py"


def test_balance_relay_routes_to_relay_replay(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    report = json.loads(source.read_text())
    report["status"] = "NEURAL_RELAY_BALANCE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
    source.write_text(json.dumps(report))
    watch.sys.argv.append("--relay-balance")
    watch.main()
    assert calls[1][1] == "scripts/rsi_audit_neural_relay_chain.py"


def test_expanded_balance_requires_all_64_worlds(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    report = json.loads(source.read_text())
    report["status"] = "NEURAL_RELAY_BALANCE_EXPANDED64_COLLECTED_PENDING_OWN_REPLAY"
    source.write_text(json.dumps(report))
    watch.sys.argv.append("--relay-balance-expanded")
    with pytest.raises(ValueError, match="unpromoted"):
        watch.main()
    report["records"] = [{}] * 64
    source.write_text(json.dumps(report))
    watch.main()
    assert calls[-1][1] == "scripts/rsi_audit_neural_relay_chain.py"


def test_lateral_development_requires_exact_unit_and_replay(monkeypatch, tmp_path):
    calls = setup(
        monkeypatch,
        tmp_path,
        state="ActiveState=inactive\nResult=success\nExecMainStatus=0\n",
        complete=True,
    )
    source = tmp_path / "source" / "complete.json"
    report = json.loads(source.read_text())
    report["status"] = "NEURAL_RELAY_RECEIVER_LATERAL_AUTHORITY32_PENDING_OWN_REPLAY"
    source.write_text(json.dumps(report))
    watch.sys.argv[watch.sys.argv.index("--unit") + 1] = (
        "rosclaw-relay-lateral-dev-b1-20261028.service"
    )
    watch.sys.argv.append("--relay-lateral-development")
    watch.main()
    assert calls[-1][1] == "scripts/rsi_audit_neural_relay_chain.py"
    watch.sys.argv[watch.sys.argv.index("--unit") + 1] = (
        "rosclaw-relay-lateral-dev-b3-20261028.service"
    )
    with pytest.raises(ValueError, match="bounded"):
        watch.main()
