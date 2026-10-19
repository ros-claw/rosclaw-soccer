"""Fake50320-step orchestration, NOT new G1 physical or learning evidence."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_qualify_navigation_recovery as mod
from scripts.rsi_navigation_recovery_inputs import require_scope


def scope():
    return dict(
        profile="full-task-navigation-recovery4-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        diagnostic_context_indices=[0, 8, 10, 12],
        action_names=["baseline", "nav_brake_0_7", "nav_brake_0"],
        main_reconstruction_runs=12,
        main_native_step_budget=36000,
        prediction_runs=16,
        prediction_native_step_budget=14320,
        combined_native_step_budget=50320,
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
def test_fixed_no_fresh_no_learning_scope(key):
    require_scope(scope())
    cfg = scope()
    cfg[key] = "wrong" if type(cfg[key]) is not bool else int(cfg[key])
    with pytest.raises(ValueError):
        require_scope(cfg)


@pytest.mark.parametrize("failure", [None, "prediction_count", "prediction_array"])
def test_whole_four_contexts_or_preserved_failure(monkeypatch, tmp_path, failure):
    import mujoco

    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        full_task_branch,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )

    jobs = []
    for i, foot in zip([0, 8, 10, 12], [185, 235, 234, 184], strict=True):
        a = dict(
            force_n=np.zeros((300, 1, 6)),
            pelvis_z_per_substep_m=np.full((300, 1, 10), 0.79),
            ball_position_after_step_m=np.zeros((300, 1, 3)),
            joint_target_rad=np.full((300, 1, 29), 1.4),
        )
        a["force_n"][foot, 0, 0] = 2
        jobs.append(
            dict(
                index=i,
                foot=foot,
                source_index=0,
                course=[2.5, 0.1, 0.3],
                source_case_hash="gold",
                comparison_start_after_measured_foot_bin=foot + 1,
                gold_trace=a,
            )
        )
    cfg = dict(
        scope(), deadline_unix=time.time() + 3600, mean_model_path="mean", mean_model_hash="mean"
    )
    p = SimpleNamespace(
        cfg=cfg,
        source=Path(mod.__file__).resolve().parent.parent,
        root=tmp_path / "out",
        pins={},
        jobs=jobs,
        model_root=Path("fake"),
        world_hash="world",
        cases=[dict(receipt=dict(witnesses=dict(world=dict(path="world"))))],
    )

    class Actor:
        def __init__(self, job):
            self.job = job
            self.next_frame = self.attempted_steps = self.integrated_steps = 0
            self.option = None

        def step(self, f):
            assert f == self.next_frame
            self.next_frame += 1
            self.attempted_steps += 10
            self.integrated_steps += 10
            arrays = {k: a[f].copy() for k, a in self.job["gold_trace"].items()}
            if self.option is not None:
                arrays["joint_target_rad"][:] = (
                    0.7 if self.option.action_name == "nav_brake_0_7" else 0.0
                )
            return SimpleNamespace(arrays=arrays)

        def install_navigation_recovery(self, option):
            assert self.next_frame == self.job["foot"] + 1 and self.option is None
            self.option = option

        def review(self):
            return dict(
                native_attempted_steps=self.attempted_steps,
                native_integrated_steps=self.integrated_steps,
            )

    class Branch:
        def __init__(self, actor, *, world_hash, action_name):
            from rosclaw_soccer.rsi.born_navigation_recovery import BornNavigationRecovery

            self.start = actor.next_frame
            self.action = action_name
            self._owned = Actor(actor.job)
            self._owned.next_frame = self.start
            if action_name != "baseline":
                self._owned.install_navigation_recovery(
                    BornNavigationRecovery(self.start, action_name)
                )

        def step(self):
            out = self._owned.step(self._owned.next_frame)
            if failure == "prediction_array" and self.action == "nav_brake_0_7":
                out.arrays["joint_target_rad"] += 0.01
            return out

        def review(self):
            return dict(
                new_native_attempted_steps=(300 - self.start) * 10,
                new_native_integrated_steps=0
                if failure == "prediction_count"
                else (300 - self.start) * 10,
            )

    class Factory:
        def __init__(self, _):
            pass

        def _stable(self):
            pass

    monkeypatch.setattr(mod, "prepare", lambda _: p)
    monkeypatch.setattr(mod, "build_actor", lambda _, job, *a: Actor(job))
    monkeypatch.setattr(mod.gc, "collect", lambda: None)
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **kw: (
            "MemAvailable: 100000000 kB\n"
            if str(self) == "/proc/meminfo"
            else original_read(self, *a, **kw)
        ),
    )
    monkeypatch.setattr(mujoco, "MjModel", SimpleNamespace(from_binary_path=lambda _: None))
    monkeypatch.setattr(sonic_torch, "FrozenSonicG1Torch", lambda *a, **kw: None)
    monkeypatch.setattr(json_artifact_io, "load_json_artifact", lambda _: dict(model_hash="mean"))
    monkeypatch.setattr(
        recurrent_sampling_episode_factory, "RecurrentSamplingEpisodeFactory", Factory
    )
    monkeypatch.setattr(full_task_branch, "FullTaskBranch", Branch)
    monkeypatch.setattr(sys, "argv", ["run", "--config", str(tmp_path / "cfg.json")])
    if failure:
        with pytest.raises(ValueError):
            mod.main()
        result = json.loads((p.root / "failure.json").read_text())
        assert result["automatic_retry"] is False
        assert not (p.root / "complete.json").exists()
    else:
        mod.main()
        result = json.loads((p.root / "complete.json").read_text())
        assert len(result["rows"]) == 4
        assert result["combined_new_native_steps"] == 50320
        assert result["new_optimizer_updates"] == 0
        assert result["teacher_qualified"] is False


original_read = Path.read_text
