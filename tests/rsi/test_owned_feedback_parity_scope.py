"""Bounded opt-in parity admission, not physics or teacher qualification."""

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts.rsi_owned_feedback_parity_inputs import require_scope
from tests.rsi.test_early_feedback_calibration import scope as original_scope


def scope():
    return dict(
        original_scope(),
        profile="six-early-owned-continuation-parity-v1",
        reuse_owned_updates_only_on_second_branch=True,
    )


def test_fixed_whole_six_no_learning_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_missing_or_widened_protocol_rejected(key):
    cfg = scope()
    del cfg[key]
    with pytest.raises(ValueError):
        require_scope(cfg)
    cfg = scope()
    value = cfg[key]
    cfg[key] = int(value) if type(value) is bool else float(value) if type(value) is int else "REAL"
    with pytest.raises(ValueError):
        require_scope(cfg)


def test_cli_dispatches_owned_parity_preflight_without_allocating(monkeypatch, capsys):
    from scripts import rsi_calibrate_feedback_branches as runner
    from scripts import rsi_owned_feedback_parity_inputs as inputs

    calls = []

    def prepare(path):
        calls.append(path)
        return SimpleNamespace(source=Path(runner.__file__).resolve().parent.parent)

    monkeypatch.setattr(inputs, "prepare", prepare)
    monkeypatch.setattr(
        sys, "argv", ["calibrate", "--config", "fixture.json", "--owned-zero-parity", "--preflight"]
    )
    runner.main()
    assert calls == [Path("fixture.json")]
    assert "SIX_OWNED_EARLY_ZERO_PARITY_READY_NO_FULL_MODELS_OR_PHYSICS" in capsys.readouterr().out
