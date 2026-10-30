"""Crash-safe publishing of incomplete PPO progress, not automatic resume."""

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.training import near_ball_residual_ppo as module


def test_progress_is_complete_json_before_and_after_update(tmp_path):
    module._persist_training_manifest(tmp_path, {"iterations": [], "evaluation": []})
    assert json.loads((tmp_path / "training.json").read_text()) == {
        "iterations": [],
        "evaluation": [],
    }
    updated = {"iterations": [{"generation": 1}], "evaluation": [{"safe": True}]}
    module._persist_training_manifest(tmp_path, updated)
    assert json.loads((tmp_path / "training.json").read_text()) == updated
    assert not list(tmp_path.glob(".training-*.tmp"))


def test_failed_replace_preserves_previous_progress(monkeypatch, tmp_path):
    module._persist_training_manifest(tmp_path, {"generation": 1})

    def fail_replace(*_args):
        raise OSError("injected replacement failure")

    monkeypatch.setattr(module.os, "replace", fail_replace)
    with pytest.raises(OSError, match="injected replacement failure"):
        module._persist_training_manifest(tmp_path, {"generation": 2})
    assert json.loads((tmp_path / "training.json").read_text()) == {"generation": 1}
    assert not list(tmp_path.glob(".training-*.tmp"))


def test_nonfinite_snapshot_rejected_without_destroying_previous(tmp_path):
    module._persist_training_manifest(tmp_path, {"generation": 1})
    with pytest.raises(ValueError):
        module._persist_training_manifest(tmp_path, {"reward": float("nan")})
    assert json.loads((tmp_path / "training.json").read_text()) == {"generation": 1}


def test_evaluation_failure_keeps_last_validated_case(monkeypatch, tmp_path):
    ids = tuple(f"agent.{i}" for i in range(8))
    cells = tuple(
        SimpleNamespace(
            agent_id=agent, growth_scope=SimpleNamespace(body_hash="sha256:" + "a" * 64)
        )
        for agent in ids
    )
    monkeypatch.setattr(
        module, "build_four_vs_four_fixture", lambda _assets: SimpleNamespace(cells=cells)
    )

    def collect(job):
        destination = module.Path(job[1])
        destination.mkdir()
        np.savez(destination / "primary.npz", sample=np.zeros(1))
        return str(destination / "probe.json")

    monkeypatch.setattr(module, "_collect", collect)
    report = {
        "exact_replay": True,
        "report_hash": "sha256:" + "b" * 64,
        "results": [{"safe": True}],
        "assessment": {},
        "causal_pass_feedback": [],
    }
    monkeypatch.setattr(module, "validate_probe", lambda _path: report)
    monkeypatch.setattr(
        module,
        "update_private_actors",
        lambda policy, *_args, **_kwargs: (replace(policy, generation=policy.generation + 1), []),
    )
    calls = []

    def evaluate(**_kwargs):
        calls.append(1)
        if len(calls) == 2:
            raise RuntimeError("injected evaluator failure")
        return report

    monkeypatch.setattr(module, "run_probe", evaluate)
    output = tmp_path / "run"
    with pytest.raises(RuntimeError, match="injected evaluator failure"):
        module.train(assets=tmp_path, output=output, iterations=1, duration=5.0)
    saved = json.loads((output / "training.json").read_text())
    assert len(saved["iterations"]) == 1
    assert len(saved["evaluation"]) == 1
    assert "manifest_hash" not in saved  # A partial checkpoint is not a final result.
