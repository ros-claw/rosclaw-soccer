"""Fake whole18/36-prediction qualification, NOT94840 actual physical steps."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_qualify_full_task_branch as mod
from scripts.rsi_full_task_branch_inputs import require_scope


def scope():
    return dict(
        profile="full-task-zero-branch18-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        main_reconstruction_runs=18,
        main_native_step_budget=54000,
        prediction_runs=36,
        prediction_native_step_budget=40840,
        combined_native_step_budget=94840,
        official_horizon_frames=300,
        post_contact_frames=60,
        constant_forward_approach=True,
        new_optimizer_updates=0,
        new_task_gain_claimed=False,
        teacher_qualified=False,
        training_authorized=False,
        private_fresh_accessed=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )


@pytest.mark.parametrize("key", list(scope()))
def test_whole_immutable_scope(key):
    require_scope(scope())
    c = scope()
    c[key] = "wrong" if type(c[key]) is not bool else int(c[key])
    with pytest.raises(ValueError):
        require_scope(c)


@pytest.mark.parametrize("failure", [None, "prediction"])
def test_whole18_or_preserved_charged_partial(monkeypatch, tmp_path, failure):
    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        causal_impact_replay,
        full_task_branch,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )

    feet = [
        185,
        183,
        184,
        183,
        151,
        148,
        151,
        148,
        235,
        209,
        234,
        209,
        184,
        184,
        148,
        148,
        228,
        228,
    ]
    jobs = []
    for i, f in enumerate(feet):
        a = dict(
            force_n=np.zeros((300, 1, 6)),
            pelvis_z_per_substep_m=np.full((300, 1, 10), 0.793),
            ball_position_after_step_m=np.zeros((300, 1, 3)),
        )
        a["force_n"][f, 0, 0] = 2.0
        jobs.append(
            dict(
                index=i,
                source_index=i % 6,
                course=[2.5, 0.1, 0.3],
                source_case_hash="gold",
                original_policy_hash="policy",
                comparison_start_after_measured_foot_bin=f + 1,
                gold_trace=a,
            )
        )
    mature = tmp_path / "mature"
    mature.mkdir()
    (mature / "course-0-arm-0-complete.json").write_text("{}")
    cfg = dict(
        scope(),
        deadline_unix=time.time() + 3600,
        mean_model_path="mean",
        mean_model_hash="mean",
        mature_worker_root=str(mature),
    )
    case = dict(
        physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
        job=dict(heading_enabled=False),
        receipt=dict(witnesses={k: dict(path=k) for k in ("world", "commitment", "report")}),
    )
    p = SimpleNamespace(
        cfg=cfg,
        source=Path(mod.__file__).resolve().parent.parent,
        root=tmp_path / "output",
        pins={},
        jobs=jobs,
        cases=[case] * 6,
        gold_complete_hash="gold",
        model_root=Path("models"),
        world_hash="world",
    )
    monkeypatch.setattr(mod, "prepare", lambda path: p)
    monkeypatch.setattr(sys, "argv", ["qualify", "--config", "fixture"])
    assets = {
        "mean": {"model_hash": "mean"},
        "commitment": {"foundation_hash": "foundation", "motor_policy_hash": "policy"},
        "report": {
            "stripped_document": {
                "executed_motor_policy": {"step_motor_proof": {"model": {"seed": 1}}}
            }
        },
    }
    monkeypatch.setattr(json_artifact_io, "load_json_artifact", lambda path: assets[str(path)])
    old = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda path, *a, **kw: (
            "MemAvailable: 100000000 kB\n" if str(path) == "/proc/meminfo" else old(path, *a, **kw)
        ),
    )
    monkeypatch.setitem(
        sys.modules,
        "mujoco",
        SimpleNamespace(MjModel=SimpleNamespace(from_binary_path=lambda path: object())),
    )

    class Factory:
        def __init__(self, mean):
            pass

        def generate_and_bind_seed(self, *, seed):
            return dict(policy_hash="policy"), object()

        def _stable(self):
            pass

    parents = []

    class Parent:
        attempted_steps = 0

        def __init__(self, model, **kw):
            assert kw["constant_forward_approach"] is True and kw["impact_enabled"] is False
            self.trace = jobs[len(parents)]["gold_trace"]
            self.next_frame = 0
            parents.append(self)

        def step(self, f):
            assert f == self.next_frame
            self.next_frame += 1
            self.attempted_steps += 10
            return SimpleNamespace(arrays={k: v[f].copy() for k, v in self.trace.items()})

        def review(self):
            return dict(native_integrated_steps=3000, native_attempted_steps=3000)

    branch_calls = []

    class Branch:
        def __init__(self, parent, **kw):
            assert kw == dict(world_hash="world", action_name="baseline")
            self.parent, self.start, self.clock = parent, parent.next_frame, parent.next_frame
            self._owned = SimpleNamespace(attempted_steps=0)
            branch_calls.append(self)

        def step(self):
            assert self.parent.next_frame == self.start
            if failure == "prediction" and self.clock == self.start + 5:
                raise RuntimeError("synthetic branch failure")
            result = SimpleNamespace(
                arrays={k: v[self.clock].copy() for k, v in self.parent.trace.items()}
            )
            self.clock += 1
            self._owned.attempted_steps += 10
            return result

        def review(self):
            assert self.clock == 300 and self.parent.next_frame == self.start
            return dict(
                new_native_integrated_steps=(300 - self.start) * 10,
                new_native_attempted_steps=(300 - self.start) * 10,
            )

    monkeypatch.setattr(
        recurrent_sampling_episode_factory, "RecurrentSamplingEpisodeFactory", Factory
    )
    monkeypatch.setattr(sonic_torch, "FrozenSonicG1Torch", lambda *a, **kw: object())
    monkeypatch.setattr(causal_impact_replay, "CausalImpactReplay", Parent)
    monkeypatch.setattr(full_task_branch, "FullTaskBranch", Branch)
    monkeypatch.setattr(mod.gc, "collect", lambda: 0)
    if failure:
        with pytest.raises(RuntimeError, match="synthetic branch"):
            mod.main()
        d = json.loads((p.root / "failure.json").read_text())
        assert (
            d["confirmed_returned_main_steps"] == 1860
            and d["confirmed_returned_prediction_steps"] == 50
        )
        assert not d["automatic_retry"] and not (p.root / "complete.json").exists()
        return
    mod.main()
    d = json.loads((p.root / "complete.json").read_text())
    assert len(parents) == 18 and len(branch_calls) == 36 and len(d["rows"]) == 18
    assert d["new_main_native_steps"] == 54000 and d["new_prediction_native_steps"] == 40840
    assert (
        d["combined_new_native_steps"] == 94840 and d["executed_control_foundation_pairs"] == 9484
    )
    assert not d["nonzero_action_prediction_qualified"] and not d["teacher_qualified"]
