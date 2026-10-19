"""Fixed successor process orchestration only, no physics or model allocation."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

from scripts import rsi_deferred_early_h300 as mod
from tests.rsi.test_early_impact_h300_scope import scope


def config():
    return dict(
        schema="soccer.rsi.deferred_early_h300.v1",
        mode="SIM_ONLY",
        automatic_retry=False,
        hardware_authorized=False,
        promotion_authorized=False,
        teacher_qualified=False,
        readback_unit="reader.service",
        readback_path="/fixture/readback.json",
        worker_config_template=dict(
            scope(), screen_readback="/fixture/readback.json", screen_readback_unit="reader.service"
        ),
    )


def test_fixed_scope():
    mod.validate_declaration(config())


@pytest.mark.parametrize(
    "key", ["automatic_retry", "hardware_authorized", "promotion_authorized", "teacher_qualified"]
)
def test_no_extra_authority_or_retry(key):
    d = config()
    d[key] = True
    with pytest.raises(ValueError):
        mod.validate_declaration(d)


@pytest.mark.parametrize("key", ["screen_readback_hash", "screen_readback_file_hash"])
def test_cannot_supply_unavailable_future_proof_hash(key):
    d = config()
    d["worker_config_template"][key] = "invented"
    with pytest.raises(ValueError):
        mod.validate_declaration(d)


def runtime(monkeypatch, tmp_path):
    d = config()
    source, worker, root = (tmp_path / name for name in ("source", "worker", "successor"))
    reader = tmp_path / "readback.json"
    d.update(
        source_root=str(source),
        source_head="bootstrap-head",
        output_root=str(root),
        readback_path=str(reader),
        readback_source_head="reader-head",
    )
    d["worker_config_template"].update(
        source_root=str(worker),
        source_head="worker-head",
        output_root=str(tmp_path / "physical"),
        screen_readback=str(reader),
    )
    data = dict(
        schema="soccer.rsi.early_feedback_impact_screen_readback.v1", source_head="reader-head"
    )
    data["report_hash"] = mod.hash_json(data)
    reader.write_text(json.dumps(data))
    path = tmp_path / "deferred.json"
    path.write_text(json.dumps(d))
    monkeypatch.setattr(mod, "__file__", str(source / "scripts/rsi_deferred_early_h300.py"))
    monkeypatch.setattr(mod, "clean_head", lambda *a: None)
    monkeypatch.setattr(sys, "argv", ["successor", "--config", str(path)])
    return d, root, reader


def test_failed_predecessor_launches_nothing_and_retains_attempt(monkeypatch, tmp_path):
    _, root, _ = runtime(monkeypatch, tmp_path)
    calls = []

    def fail(unit):
        raise ValueError("reader not successfully terminal")

    monkeypatch.setattr(mod, "require_terminal", fail)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="terminal"):
        mod.main()
    assert not calls
    failed = json.loads((root / "failure.json").read_text())
    assert not failed["physical_worker_started"] and not failed["automatic_retry"]
    with pytest.raises(ValueError, match="unused"):
        mod.main()


def test_actual_readback_hash_then_frozen_worker_preflight_then_only_fixed_run(
    monkeypatch, tmp_path
):
    cfg, root, reader = runtime(monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr(mod, "require_terminal", lambda unit: None)
    monkeypatch.setattr(mod.subprocess, "run", lambda command, **kw: commands.append(command))
    mod.main()
    assert len(commands) == 2
    assert commands[0] == [*commands[1], "--preflight"]
    assert commands[1][1] == str(
        Path(cfg["worker_config_template"]["source_root"]) / "scripts/rsi_early_impact_h300.py"
    )
    emitted = json.loads((root / "worker-config.json").read_text())
    assert emitted["screen_readback_file_hash"] == mod.digest(reader)
    assert emitted["source_head"] == "worker-head"
    assert (root / "worker-preflight-complete.json").is_file()
    done = json.loads((root / "command-complete.json").read_text())
    assert done["command_success_is_not_task_success"] and not done["teacher_qualified"]


def test_worker_preflight_refusal_blocks_physical_run(monkeypatch, tmp_path):
    _, root, _ = runtime(monkeypatch, tmp_path)
    commands = []
    monkeypatch.setattr(mod, "require_terminal", lambda unit: None)

    def refused(command, **kw):
        commands.append(command)
        raise subprocess.CalledProcessError(1, command)

    monkeypatch.setattr(mod.subprocess, "run", refused)
    with pytest.raises(subprocess.CalledProcessError):
        mod.main()
    assert len(commands) == 1 and commands[0][-1] == "--preflight"
    failed = json.loads((root / "failure.json").read_text())
    assert not failed["physical_worker_started"]
