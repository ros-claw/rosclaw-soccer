"""Early80ms integration scope/hash-binding admission, no physical execution."""

import json
from pathlib import Path

import pytest

from rosclaw_soccer.sim.contracts import hash_json
from scripts import rsi_replay_feedback_screen as mod
from tests.rsi.test_feedback_replay_scope import scope as old_scope


def scope():
    cfg = old_scope()
    cfg.update(
        profile="six-recorded-early-feedback80-replay-v1",
        branch_step_budget=3360,
        combined_step_budget=21360,
    )
    return cfg


def test_scope():
    mod.require_scope(scope(), early80=True)


@pytest.mark.parametrize("key", list(scope()))
def test_no_missing_or_widened_scope(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        mod.require_scope(cfg, early80=True)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        mod.require_scope(cfg, early80=True)


def declaration(monkeypatch, tmp_path):
    cfg = scope()
    source = Path(mod.__file__).resolve().parent.parent
    cfg.update(
        source_root=str(source),
        source_head="pinned",
        output_root=str(tmp_path / "out"),
        deadline_unix=9999999999,
        screen_unit="screen.service",
        bind_terminal_readback=True,
        readback_unit="reader.service",
        readback=str(tmp_path / "reader.json"),
        readback_source_head="expected-reader",
    )
    monkeypatch.setattr(mod, "require_terminal", lambda unit: None)
    monkeypatch.setattr(
        mod.subprocess,
        "check_output",
        lambda args, **kw: "pinned" if args[1] == "rev-parse" else "",
    )
    return cfg, tmp_path / "config.json"


@pytest.mark.parametrize("bad", ["hash_override", "file_hash_override", "integer_flag"])
def test_deferred_hashes_cannot_be_fabricated_or_untyped(monkeypatch, tmp_path, bad):
    cfg, path = declaration(monkeypatch, tmp_path)
    if bad == "hash_override":
        cfg["readback_hash"] = "invented"
    elif bad == "file_hash_override":
        cfg["readback_file_hash"] = "invented"
    else:
        cfg["bind_terminal_readback"] = 1
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="terminal-only"):
        mod.prepare(path, early80=True)
    assert not Path(cfg["output_root"]).exists()


def test_deferred_readback_must_have_expected_independent_source(monkeypatch, tmp_path):
    cfg, path = declaration(monkeypatch, tmp_path)
    d = {"source_head": "wrong-reader"}
    d["report_hash"] = hash_json(d)
    Path(cfg["readback"]).write_text(json.dumps(d))
    path.write_text(json.dumps(cfg))
    with pytest.raises(ValueError, match="reader source"):
        mod.prepare(path, early80=True)
    assert not Path(cfg["output_root"]).exists()
