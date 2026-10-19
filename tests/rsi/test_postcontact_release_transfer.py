"""Synthetic cohort and orchestration tests; no G1 competence claim."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_postcontact_release_transfer as mod
from scripts.rsi_postcontact_release_transfer_inputs import require_scope
from tests.rsi.test_postcontact_release_protocol import arrays


def scope():
    return dict(
        profile="postcontact-fixed-release-transfer18-v1",
        mode="SIM_ONLY",
        action_name="forward_minus",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        new_rollouts=18,
        native_step_budget=54000,
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
    )


def test_whole_fixed_scope():
    require_scope(scope())


@pytest.mark.parametrize("key", list(scope()))
def test_scope_missing_or_type_changed_rejected(key):
    c = scope()
    c.pop(key)
    with pytest.raises(ValueError):
        require_scope(c)
    c = scope()
    c[key] = (
        int(c[key]) if type(c[key]) is bool else float(c[key]) if type(c[key]) is int else "other"
    )
    with pytest.raises(ValueError):
        require_scope(c)


def rows():
    return [
        dict(
            index=i,
            partition="CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION",
            baseline_outcome=dict(task_pass=True, safe=True),
            outcome=dict(task_pass=True, safe=True),
        )
        for i in range(18)
    ]


def test_development_gain_never_qualifies_teacher_and_retention_veto():
    r = rows()
    r[0]["baseline_outcome"]["task_pass"] = False
    result = mod.paired_summary(r)
    assert result["local"]["rescues"] == 1 and result["fixed_action_locally_promising"]
    assert not result["teacher_qualified"] and not result["training_authorized"]
    r[-1]["outcome"]["task_pass"] = False
    assert not mod.paired_summary(r)["fixed_action_locally_promising"]
    r[-1]["outcome"]["task_pass"] = True
    r[2]["outcome"]["safe"] = False
    assert not mod.paired_summary(r)["fixed_action_locally_promising"]


@pytest.mark.parametrize("bad", ["drop", "order", "partition", "bool"])
def test_summary_incomplete_or_wrong_primitive_rejected(bad):
    r = rows()
    if bad == "drop":
        r.pop()
    elif bad == "order":
        r.reverse()
    elif bad == "partition":
        r[-1]["partition"] = "CONSUMED_LOCAL_DEVELOPMENT"
    else:
        r[0]["outcome"]["safe"] = 1
    with pytest.raises(ValueError):
        mod.paired_summary(r)


@pytest.mark.parametrize("failure_frame", [None, 5])
def test_complete18_or_preserved_fatal_partial(monkeypatch, tmp_path, failure_frame):
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
    baseline, _ = arrays("baseline")
    baseline = {k: v for k, v in baseline.items() if not k.startswith("postcontact_release_")}
    # Fake bodies remain byte-identical to fake baseline until first action.
    outcome = dict(task_pass=True, safe=True)
    jobs = [
        dict(
            r,
            source_index=i % 6,
            course=[2.675, 0.132, 0.319],
            baseline_hash="baseline",
            baseline_trace=baseline,
        )
        for i, r in enumerate(rows())
    ]
    for j in jobs:
        j.pop("outcome")
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
        root=tmp_path / "output",
        jobs=jobs,
        cases=[case] * 6,
        pins={},
        local_complete_hash="local",
        release_physics_hash="physical",
        declaration=dict(candidate=dict(parameters={})),
        world_hash="world",
        model_root=Path("models"),
    )
    monkeypatch.setattr(mod, "prepare", lambda path: p)
    monkeypatch.setattr(sys, "argv", ["transfer", "--config", "fixture.json"])
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
            if frame == failure_frame:
                raise RuntimeError("synthetic fatal transfer failure")
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
        with pytest.raises(RuntimeError, match="synthetic fatal"):
            mod.main()
        fail = json.loads((p.root / "failure.json").read_text())
        assert fail["current_attempted_steps"] == 50 and fail["completed_cases"] == 0
        assert not fail["automatic_retry"] and not (p.root / "complete.json").exists()
        assert (p.root / "failure-state.npz").is_file()
        return
    mod.main()
    result = json.loads((p.root / "complete.json").read_text())
    assert names == ["forward_minus"] * 18
    assert result["new_native_steps"] == 54000 and result["baseline_rollouts_reused"] == 18
    assert result["local"]["contexts"] == 12 and result["retention"]["contexts"] == 6
    assert result["new_optimizer_updates"] == result["forecast_steps"] == 0
    assert not result["teacher_qualified"] and not result["training_authorized"]
