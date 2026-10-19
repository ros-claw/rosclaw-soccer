"""Full-bank scope/selection/fake orchestration, never physical G1 evidence."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from scripts import rsi_postcontact_full_bank as mod
from scripts.rsi_postcontact_full_bank_inputs import require_scope
from tests.rsi.test_postcontact_release_protocol import arrays


def scope():
    return dict(
        profile="postcontact-full-h300-bank252-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        comparison_contexts=18,
        comparison_rollouts=252,
        reused_authority_context_index=1,
        reused_authority_rollouts=14,
        new_rollouts=238,
        native_step_budget=714000,
        forecast_step_budget=0,
        official_horizon_frames=300,
        post_contact_frames=60,
        maximum_release_duration_frames=3,
        new_optimizer_updates=0,
        baseline_rollouts_reused=18,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
        action_names=[*ACTION_NAMES, "baseline"],
    )


def test_full_fixed_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_scope_cannot_drop_or_change_bound(key):
    c = scope()
    c.pop(key)
    with pytest.raises(ValueError):
        require_scope(c)
    c = scope()
    v = c[key]
    c[key] = int(v) if type(v) is bool else float(v) if type(v) is int else "other"
    with pytest.raises(ValueError):
        require_scope(c)


def context():
    outcome = dict(task_pass=False, safe=True, forward_60_m=0.3, lateral_60_m=0.1)
    job = dict(
        index=0, partition="CONSUMED_LOCAL_DEVELOPMENT", source_index=0, baseline_outcome=outcome
    )
    rows = [
        dict(
            context_index=0,
            action_index=i,
            action_name=n,
            outcome=dict(outcome),
            normalized_action_norm_squared=0.0,
        )
        for i, n in enumerate([*ACTION_NAMES, "baseline"])
    ]
    return job, rows


def test_full_future_outcomes_only_generate_offline_labels_not_runtime_or_teacher_permission():
    j, r = context()
    r[2]["outcome"] = dict(task_pass=True, safe=True, forward_60_m=1.2, lateral_60_m=0.2)
    r[3]["outcome"] = dict(task_pass=True, safe=True, forward_60_m=1.4, lateral_60_m=0.1)
    d = mod.summarize_context(j, r)
    assert d["offline_preferred_action_index"] == 3 and d[
        "observed_safe_passing_action_indices"
    ] == [2, 3]
    assert d["selection_uses_full_simulated_future_outcomes"]
    assert not d["runtime_future_outcome_input_authorized"] and not d["training_authorized"]
    assert not d["teacher_qualified"]
    j["baseline_outcome"] = dict(task_pass=True, safe=True, forward_60_m=1.2, lateral_60_m=0.1)
    r[0]["outcome"] = r[-1]["outcome"] = j["baseline_outcome"]
    assert mod.summarize_context(j, r)["offline_preferred_action_index"] == 0


@pytest.mark.parametrize("bad", ["drop", "index", "name", "baseline", "unsafe_pass", "bool"])
def test_incomplete_or_invalid_bank_cannot_make_training_labels(bad):
    j, r = context()
    if bad == "drop":
        r.pop()
    elif bad == "index":
        r[2]["context_index"] = 1
    elif bad == "name":
        r[2]["action_name"] = "arbitrary"
    elif bad == "baseline":
        r[-1]["outcome"]["forward_60_m"] = 0.4
    elif bad == "unsafe_pass":
        r[3]["outcome"].update(task_pass=True, safe=False)
    else:
        r[0]["outcome"]["safe"] = 1
    with pytest.raises(ValueError):
        mod.summarize_context(j, r)


@pytest.mark.parametrize("failure_frame", [None, 5])
def test_fake252_comparisons238_new_bodies_and14_honest_reused_or_fatal_partial(
    monkeypatch, tmp_path, failure_frame
):
    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        causal_impact_replay,
        first_touch_outcome,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )

    cfg = dict(
        scope(), deadline_unix=time.time() + 3600, mean_model_path="mean", mean_model_hash="mean"
    )
    original, _ = arrays("baseline")
    original = {k: v for k, v in original.items() if not k.startswith("postcontact_release_")}
    outcome = context()[0]["baseline_outcome"]
    jobs = [
        dict(
            index=i,
            source_index=i % 6,
            partition="CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION",
            course=[2.675, 0.132, 0.319],
            baseline_hash="base",
            baseline_outcome=outcome,
            baseline_trace=original,
        )
        for i in range(18)
    ]
    reused = []
    for i, name in enumerate(cfg["action_names"]):
        t, r = arrays(name)
        file = tmp_path / f"reused-{i}.npz"
        np.savez_compressed(file, **t)
        reused.append(
            dict(
                arrays=dict(path=str(file), sha256=mod.digest(file)),
                original_policy_hash="policy",
                report_hash=f"already-executed-{i}",
                causal_review=dict(r, native_integrated_steps=3000, forecast_integrated_steps=0),
            )
        )
    case = dict(
        job=dict(heading_enabled=False),
        physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
        receipt=dict(
            receipt_hash="receipt",
            witnesses=dict(
                world=dict(path="world"),
                commitment=dict(path="commitment"),
                report=dict(path="report"),
            ),
        ),
    )
    p = SimpleNamespace(
        cfg=cfg,
        source=Path(mod.__file__).resolve().parent.parent,
        root=tmp_path / "out",
        pins={},
        jobs=jobs,
        cases=[case] * 6,
        model_root=Path("models"),
        world_hash="world",
        reused_authority=reused,
        reused_authority_hash="authority",
        declaration=dict(candidate=dict(parameters={})),
    )
    monkeypatch.setattr(mod, "prepare", lambda path: p)
    monkeypatch.setattr(mod.gc, "collect", lambda: None)
    monkeypatch.setattr(sys, "argv", ["bank", "--config", "fixture.json"])
    assets = dict(
        mean=dict(model_hash="mean"),
        commitment=dict(foundation_hash="foundation", motor_policy_hash="policy"),
        report=dict(
            stripped_document=dict(
                executed_motor_policy=dict(step_motor_proof=dict(model=dict(seed=1)))
            )
        ),
    )
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
            assert mean["model_hash"] == "mean"

        def generate_and_bind_seed(self, *, seed):
            return dict(policy_hash="policy"), object()

        def _stable(self):
            pass

    names = []

    class Replay:
        def __init__(self, model, **kw):
            assert not kw["impact_enabled"] and not kw["early_impact_enabled"]
            names.append(kw["release_action_name"])
            self.t, self.r = arrays(kw["release_action_name"])
            self.attempted_steps = 0
            self.data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), ctrl=np.zeros(29))

        def step(self, f):
            if f == failure_frame:
                raise RuntimeError("synthetic full-bank partial failure")
            self.attempted_steps += 10
            return SimpleNamespace(arrays={k: v[f] for k, v in self.t.items()})

        def review(self):
            return dict(self.r, native_integrated_steps=3000, forecast_integrated_steps=0)

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
        with pytest.raises(RuntimeError, match="synthetic full-bank"):
            mod.main()
        fail = json.loads((p.root / "failure.json").read_text())
        assert fail["completed_new_native_steps"] == 0 and fail["current_attempted_steps"] == 50
        assert fail["completed_new_rollouts"] == 0 and not fail["automatic_retry"]
        assert (p.root / "failure-state.npz").exists() and not (p.root / "complete.json").exists()
        return
    mod.main()
    c = json.loads((p.root / "complete.json").read_text())
    assert c["comparison_rollouts"] == 252 and c["comparison_contexts"] == 18
    assert c["reused_execution_count"] == 14 and c["new_rollouts"] == 238
    assert c["new_native_steps"] == 714000 and c["executed_control_foundation_pairs"] == 71400
    assert len(names) == 238 and len(list(p.root.glob("*-trace.npz"))) == 238
    assert all(r["execution_reused"] for r in c["contexts"][1]["rows"])
    assert c["new_optimizer_updates"] == c["forecast_steps"] == 0
    assert not c["teacher_qualified"] and not c["training_authorized"]
