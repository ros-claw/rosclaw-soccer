"""Fourteen-trial orchestration with fake models/physics, not G1 evidence."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from tests.rsi.test_postcontact_release_protocol import arrays, scope


@pytest.mark.parametrize("failure_frame", [None, 5])
def test_whole14_scope_or_explicit_partial_failure(monkeypatch, tmp_path, failure_frame):
    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        causal_impact_replay,
        first_touch_outcome,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )
    from scripts import rsi_postcontact_release_authority as mod

    cfg = dict(
        scope(), deadline_unix=time.time() + 3600, mean_model_path="mean", mean_model_hash="mean"
    )
    original, _ = arrays("baseline")
    original = {k: v for k, v in original.items() if not k.startswith("postcontact_release_")}
    outcome = dict(task_pass=False, safe=True)
    p = SimpleNamespace(
        source=Path(mod.__file__).resolve().parent.parent,
        cfg=cfg,
        root=tmp_path / "output",
        pins={},
        original_trace=original,
        original_outcome=outcome,
        local_pair_hash="closed-pair",
        course=(2.675, 0.132, 0.319),
        model_root=Path("models"),
        world_hash="world",
        declaration=dict(candidate=dict(parameters={})),
        case=dict(
            job=dict(heading_enabled=False),
            physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
            receipt=dict(
                receipt_hash="source",
                witnesses=dict(
                    world=dict(path="world"),
                    commitment=dict(path="commitment"),
                    report=dict(path="report"),
                ),
            ),
        ),
    )
    monkeypatch.setattr(mod, "prepare", lambda path: p)
    monkeypatch.setattr(sys, "argv", ["release", "--config", "fixture.json"])
    assets = {
        "mean": dict(model_hash="mean"),
        "commitment": dict(foundation_hash="foundation", motor_policy_hash="policy"),
        "report": dict(
            stripped_document=dict(
                executed_motor_policy=dict(step_motor_proof=dict(model=dict(seed=1)))
            )
        ),
    }
    monkeypatch.setattr(json_artifact_io, "load_json_artifact", lambda path: assets[str(path)])
    read_text = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "MemAvailable: 100000000 kB\n"
            if str(path) == "/proc/meminfo"
            else read_text(path, *a, **kw)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )

    class Factory:
        def __init__(self, mean):
            assert mean == dict(model_hash="mean")

        def generate_and_bind_seed(self, *, seed):
            assert seed == 1
            return dict(policy_hash="policy"), object()

        def _stable(self):
            pass

    names = []

    class Replay:
        def __init__(self, model, **kwargs):
            assert kwargs["impact_enabled"] is kwargs["early_impact_enabled"] is False
            names.append(kwargs["release_action_name"])
            self.trace, self.review_data = arrays(kwargs["release_action_name"])
            self.attempted_steps = 0
            self.data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), ctrl=np.zeros(29))

        def step(self, frame):
            assert self.attempted_steps == frame * 10
            if frame == failure_frame:
                raise RuntimeError("synthetic release runner failure")
            self.attempted_steps += 10
            return SimpleNamespace(arrays={k: v[frame] for k, v in self.trace.items()})

        def review(self):
            assert self.attempted_steps == 3000
            return dict(self.review_data, native_integrated_steps=3000, forecast_integrated_steps=0)

    monkeypatch.setattr(
        recurrent_sampling_episode_factory, "RecurrentSamplingEpisodeFactory", Factory
    )
    monkeypatch.setattr(sonic_torch, "FrozenSonicG1Torch", lambda *a, **kw: object())
    monkeypatch.setattr(causal_impact_replay, "CausalImpactReplay", Replay)
    monkeypatch.setattr(
        first_touch_outcome,
        "label_first_touch",
        lambda trace: SimpleNamespace(to_dict=lambda: outcome),
    )
    if failure_frame is not None:
        with pytest.raises(RuntimeError, match="synthetic release"):
            mod.main()
        d = json.loads((p.root / "failure.json").read_text())
        assert d["completed_trials"] == 0 and d["current_attempted_steps"] == 50
        assert d["automatic_retry"] is False
        assert not (p.root / "complete.json").exists()
        assert (p.root / "failure-state.npz").is_file()
        with np.load(p.root / "failure-partial-trace.npz", allow_pickle=False) as a:
            assert a["force_n"].shape[0] == 5
        return
    mod.main()
    assert names == cfg["action_names"]
    d = json.loads((p.root / "complete.json").read_text())
    assert len(d["rows"]) == 14 and d["physical_contexts"] == 1
    assert d["new_native_steps"] == 42000
    assert d["executed_control_foundation_pairs"] == 4200
    assert d["new_optimizer_updates"] == d["forecast_steps"] == 0
    assert d["successful_action_names"] == []
    assert d["teacher_qualified"] is d["training_authorized"] is d["promotion_authorized"] is False
    assert len(list(p.root.glob("trial-*-trace.npz"))) == 14
