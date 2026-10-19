"""Fixed successor orchestration with mocked external processes, no physics."""

import json
import sys
from pathlib import Path

import pytest

from scripts import rsi_early_feedback_successor as mod
from tests.rsi.test_early_feedback_screen_scope import scope


def config():
    return dict(
        mode="SIM_ONLY",
        profile="early80ms-zero-readback-then-fixed-screen-v1",
        hardware_authorized=False,
        promotion_authorized=False,
        teacher_qualified=False,
        training_authorized=False,
        automatic_retry=False,
        actor_future_event_input_allowed=False,
        new_optimizer_updates=0,
        zero_unit="zero.service",
        readback_unit="reader.service",
        screen_template=scope(),
    )


def test_declared_fixed_chain():
    mod.validate_config(config())


@pytest.mark.parametrize(
    "key",
    [
        "hardware_authorized",
        "promotion_authorized",
        "teacher_qualified",
        "training_authorized",
        "automatic_retry",
        "actor_future_event_input_allowed",
    ],
)
def test_rejects_authority_or_retries(key):
    cfg = config()
    cfg[key] = True
    with pytest.raises(ValueError):
        mod.validate_config(cfg)


@pytest.mark.parametrize(
    "key", ["source_head", "source_root", "output_root", "early_zero_readback"]
)
def test_template_cannot_override_identity_or_proof(key):
    cfg = config()
    cfg["screen_template"][key] = "override"
    with pytest.raises(ValueError):
        mod.validate_config(cfg)


def runtime(monkeypatch, tmp_path, stage):
    cfg = config()
    zero = tmp_path / "zero"
    zero.mkdir()
    source = tmp_path / "source"
    cfg.update(
        source_root=str(source),
        source_head="pinned-head",
        stage_root=str(tmp_path / "stages"),
        zero_root=str(zero),
        zero_config=str(tmp_path / "zero-config.json"),
    )
    Path(cfg["zero_config"]).write_text(json.dumps({"output_root": str(zero)}))
    cfg["zero_config_file_hash"] = mod.digest(cfg["zero_config"])
    file = tmp_path / "config.json"
    file.write_text(json.dumps(cfg))
    monkeypatch.setattr(mod, "__file__", str(source / "scripts/rsi_early_feedback_successor.py"))
    monkeypatch.setattr(sys, "argv", ["stage", "--stage", stage, "--config", str(file)])
    monkeypatch.setattr(
        mod.subprocess,
        "check_output",
        lambda args, **kw: "pinned-head" if args[1] == "rev-parse" else "",
    )
    return cfg, zero, Path(cfg["stage_root"])


def test_failed_prerequisite_runs_no_reader_or_models_and_is_not_retried(monkeypatch, tmp_path):
    _, _, root = runtime(monkeypatch, tmp_path, "readback")
    calls = []

    def failed(unit):
        raise ValueError("actual zero process not successfully terminal")

    monkeypatch.setattr(mod, "require_terminal", failed)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="not successfully terminal"):
        mod.main()
    assert not calls
    assert (root / "readback-started.json").exists()
    assert (root / "readback-failure.json").exists()
    with pytest.raises(FileExistsError):
        mod.main()
    assert not calls


def test_successful_reader_uses_fixed_script_and_actual_complete_bytes(monkeypatch, tmp_path):
    cfg, zero, root = runtime(monkeypatch, tmp_path, "readback")
    (zero / "complete.json").write_text("{}")
    seen = []
    monkeypatch.setattr(mod, "require_terminal", lambda unit: None)
    monkeypatch.setattr(mod.subprocess, "run", lambda command, **kw: seen.append((command, kw)))
    mod.main()
    command, kw = seen[0]
    assert command[1] == "scripts/rsi_read_zero_feedback.py" and "--early80" in command
    assert command[command.index("--complete-file-hash") + 1] == mod.digest(zero / "complete.json")
    assert kw["check"] is True and kw["cwd"] == Path(cfg["source_root"])
    assert kw["env"]["CUDA_VISIBLE_DEVICES"] == ""
    done = json.loads((root / "readback-command-complete.json").read_text())
    assert done["command_success_is_not_task_success"] and not done["teacher_qualified"]


def test_screen_cannot_run_until_distinct_reader_terminal(monkeypatch, tmp_path):
    cfg, _, root = runtime(monkeypatch, tmp_path, "screen")
    calls = []

    def terminal(unit):
        if unit == cfg["readback_unit"]:
            raise ValueError("reader still running")

    monkeypatch.setattr(mod, "require_terminal", terminal)
    monkeypatch.setattr(mod.subprocess, "run", lambda *a, **kw: calls.append(a))
    with pytest.raises(ValueError, match="reader still running"):
        mod.main()
    assert not calls and not (root / "screen-config.json").exists()
