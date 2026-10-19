"""Fake forty-course orchestration; counts are not G1 physical evidence."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.foot_heading_parity import SHAPES
from rosclaw_soccer.rsi.postcontact_robustness_plan import declare_robustness40
from scripts import rsi_collect_public_four_arm as mod
from scripts.rsi_public_four_arm_inputs import require_scope


def scope():
    return dict(
        profile="public-postcontact-four-arm40-v1",
        mode="SIM_ONLY",
        source_cases=6,
        new_rollouts=160,
        native_step_budget=480000,
        forecast_step_budget=44800,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        baseline_rollouts_reused=0,
        reuse_owned_postcontact_updates=True,
        constant_forward_approach=False,
        collection_workers=1,
        teacher_qualified=False,
        training_authorized=False,
        private_fresh_accessed=False,
        promotion_authorized=False,
        hardware_authorized=False,
        automatic_retry=False,
    )


@pytest.mark.parametrize("key", list(scope()))
def test_fixed_whole_scope(key):
    require_scope(scope())
    c = scope()
    c[key] = 1 if type(c[key]) is bool else "wrong"
    with pytest.raises(ValueError):
        require_scope(c)


@pytest.mark.parametrize("failure_frame", [None, 5])
def test_all160_fresh_owned_bodies_or_fatal_partial(monkeypatch, tmp_path, failure_frame):
    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        causal_impact_replay,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )

    mean_hash = "sha256:" + "a" * 64
    plan = declare_robustness40(mean_model_hash=mean_hash)
    cfg = dict(
        scope(), mean_model_path="mean", mean_model_hash=mean_hash, deadline_unix=time.time() + 3600
    )
    anchor = dict(
        physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
        receipt=dict(
            receipt_hash="anchor", witnesses={k: dict(path=k) for k in ("commitment", "world")}
        ),
    )
    p = SimpleNamespace(
        cfg=cfg,
        source=Path(mod.__file__).resolve().parent.parent,
        root=tmp_path / "output",
        plan=plan,
        pins={},
        owned_parity_hash="parity",
        cases=[anchor] * 6,
        model_root=Path("models"),
        world_hash="world",
    )
    monkeypatch.setattr(mod, "prepare", lambda path: p)
    monkeypatch.setattr(sys, "argv", ["public40", "--config", "fixture"])
    assets = dict(mean=dict(model_hash=mean_hash), commitment=dict(foundation_hash="foundation"))
    monkeypatch.setattr(json_artifact_io, "load_json_artifact", lambda path: assets[str(path)])
    old_read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "MemAvailable: 100000000 kB\n"
            if str(path) == "/proc/meminfo"
            else old_read(path, *a, **kw)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )
    decoders, body_calls = [], []

    class Factory:
        def __init__(self, mean):
            assert mean["model_hash"] == mean_hash

        def generate_and_bind_seed(self, *, seed):
            decoder = object()
            decoders.append(decoder)  # keep alive so identity cannot be reused
            return dict(policy_hash=f"policy-{seed}"), decoder

        def _stable(self):
            pass

    base = {
        k: np.zeros(
            (300, 1, *shape),
            dtype=np.bool_
            if k == "recurrent_sampling_performed"
            else np.float32
            if k.startswith("foundation_neural_")
            else np.float64,
        )
        for k, shape in SHAPES.items()
    }
    base["pelvis_z_per_substep_m"][:] = 0.793

    class Replay:
        attempted_steps = 0

        def __init__(self, model, **kw):
            body_calls.append(kw)
            assert kw["impact_enabled"] is False
            assert kw.get("constant_forward_approach", False) is False
            self.trace = base
            self.post = kw.get("postcontact_exit_enabled", False)
            if self.post:
                assert kw["reuse_owned_postcontact_updates"] is True
                self.teacher = SimpleNamespace(forecaster=SimpleNamespace(attempted_steps=0))
            else:
                self.teacher = None

        def step(self, f):
            if f == failure_frame:
                raise RuntimeError("synthetic fatal public40")
            self.attempted_steps += 10
            return SimpleNamespace(arrays={k: v[f] for k, v in self.trace.items()})

        def review(self):
            assert self.attempted_steps == 3000
            return dict(
                native_integrated_steps=3000,
                forecast_integrated_steps=0,
                postcontact_exit_owned_updates_reused=self.post,
            )

    monkeypatch.setattr(
        mod, "check_exit_trace", lambda *a: dict(first_physically_applied_correction_frame=None)
    )
    monkeypatch.setattr(
        recurrent_sampling_episode_factory, "RecurrentSamplingEpisodeFactory", Factory
    )
    monkeypatch.setattr(sonic_torch, "FrozenSonicG1Torch", lambda *a, **kw: object())
    monkeypatch.setattr(causal_impact_replay, "CausalImpactReplay", Replay)
    # GC performance is not the subject of this orchestration test.
    monkeypatch.setattr(mod.gc, "collect", lambda: 0)
    if failure_frame is not None:
        with pytest.raises(RuntimeError, match="synthetic fatal"):
            mod.main()
        f = json.loads((p.root / "failure.json").read_text())
        assert f["completed_native_steps"] == f["completed_bodies"] == 0
        assert f["current_attempted_steps"] == 50 and not f["automatic_retry"]
        assert not (p.root / "complete.json").exists()
        return
    mod.main()
    c = json.loads((p.root / "complete.json").read_text())
    assert len(body_calls) == len({id(d) for d in decoders}) == len(c["rows"]) == 160
    assert c["new_native_steps"] == 480000 and len(c["pairs"]) == 80
    assert c["unique_initial_states"] == 40 and c["generator_seed_clusters"] == 10
    assert not c["teacher_qualified"] and not c["training_authorized"]
    for i in range(40):
        group = body_calls[i * 4 : (i + 1) * 4]
        assert len({g["expected_policy_hash"] for g in group}) == 1
        assert [g["heading"].enabled for g in group] == [False, False, True, True]
        assert [g.get("postcontact_exit_enabled", False) for g in group] == [
            False,
            True,
            False,
            True,
        ]
