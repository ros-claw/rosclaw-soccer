"""Fake whole14/50960 accounting, no actual G1 physics or neural inference."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_qualify_born_release as entry
from scripts import rsi_qualify_navigation_recovery as bank
from scripts.rsi_born_release_inputs import ACTIONS, require_scope


def scope():
    return dict(
        profile="full-task-born-release14-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        diagnostic_context_index=8,
        action_names=ACTIONS,
        main_reconstruction_runs=14,
        main_native_step_budget=42000,
        prediction_runs=14,
        prediction_native_step_budget=8960,
        combined_native_step_budget=50960,
        official_horizon_frames=300,
        post_contact_frames=60,
        constant_forward_approach=True,
        born_release_capture=True,
        new_optimizer_updates=0,
        new_task_gain_claimed=False,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )


@pytest.mark.parametrize("key", list(scope()))
def test_exact_one_context_bounded_scope(key):
    cfg = scope()
    require_scope(cfg)
    cfg[key] = "wrong" if type(cfg[key]) is not bool else int(cfg[key])
    with pytest.raises(ValueError):
        require_scope(cfg)


@pytest.mark.parametrize("failure", [None, "count", "future_array"])
def test_whole14_or_preserved_partial(monkeypatch, tmp_path, failure):
    import mujoco

    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        full_task_branch,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )

    gold = dict(
        force_n=np.zeros((300, 1, 6)),
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.79),
        ball_position_after_step_m=np.zeros((300, 1, 3)),
        joint_target_rad=np.zeros((300, 1, 29)),
    )
    gold["force_n"][235, 0, 0] = 2.0
    job = dict(
        index=8,
        source_index=0,
        course=[2.5, 0.1, 0.3],
        source_case_hash="gold",
        comparison_start_after_measured_foot_bin=236,
        gold_trace=gold,
    )
    cfg = dict(
        scope(), deadline_unix=time.time() + 3600, mean_model_path="mean", mean_model_hash="mean"
    )
    p = SimpleNamespace(
        cfg=cfg,
        jobs=[job],
        root=tmp_path / "out",
        pins={},
        world_hash="world",
        source=Path(entry.__file__).resolve().parent.parent,
        model_root=Path("fake"),
        cases=[dict(receipt=dict(witnesses=dict(world=dict(path="fake"))))],
    )

    class Actor:
        def __init__(self):
            self.next_frame = self.attempted_steps = self.integrated_steps = 0
            self.action = None

        def install_born_release(self, *, world_hash, action_name):
            assert self.next_frame == 236 and self.action is None
            self.action = action_name

        def step(self, f):
            assert f == self.next_frame
            self.next_frame += 1
            self.attempted_steps += 10
            self.integrated_steps += 10
            a = {k: v[f].copy() for k, v in gold.items()}
            offset = np.zeros((1, 29))
            if self.action not in (None, "baseline"):
                a["ball_position_after_step_m"][0, 0] = (f - 235) * 0.025
                if f < 239:
                    offset[0, 5] = 0.1
            a.update(
                born_release_requested_offset_nm=offset,
                born_release_applied_offset_nm=np.repeat(offset[:, None, :], 10, axis=1),
                born_release_forecast_steps=np.zeros((1, 1), dtype=np.int64),
            )
            return SimpleNamespace(arrays=a)

        def review(self):
            return dict(
                native_attempted_steps=self.attempted_steps,
                native_integrated_steps=self.integrated_steps,
            )

    class Branch:
        def __init__(self, actor, *, world_hash, action_name):
            self.start = actor.next_frame
            self._owned = Actor()
            self._owned.next_frame = self.start
            if action_name != "baseline":
                self._owned.install_born_release(
                    world_hash=world_hash, action_name=action_name.split(":")[1]
                )

        def step(self):
            step = self._owned.step(self._owned.next_frame)
            if failure == "future_array" and self._owned.action not in (None, "baseline"):
                step.arrays["joint_target_rad"] += 0.01
            return step

        def review(self):
            return dict(
                new_native_attempted_steps=(300 - self.start) * 10,
                new_native_integrated_steps=0 if failure == "count" else (300 - self.start) * 10,
            )

    class Factory:
        def __init__(self, _):
            pass

        def _stable(self):
            pass

    read = Path.read_text
    monkeypatch.setattr(
        Path,
        "read_text",
        lambda self, *a, **kw: (
            "MemAvailable: 100000000 kB\n" if str(self) == "/proc/meminfo" else read(self, *a, **kw)
        ),
    )
    monkeypatch.setattr(entry, "prepare", lambda _: p)
    monkeypatch.setattr(bank, "build_actor", lambda *a, **kw: Actor())
    monkeypatch.setattr(bank.gc, "collect", lambda: None)
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
            entry.main()
        assert not (p.root / "complete.json").exists()
        assert json.loads((p.root / "failure.json").read_text())["automatic_retry"] is False
    else:
        entry.main()
        result = json.loads((p.root / "complete.json").read_text())
        assert result["combined_new_native_steps"] == 50960
        assert len(result["rows"]) == 1 and len(result["rows"][0]["actuals"]) == 14
        assert result["new_optimizer_updates"] == 0 and result["teacher_qualified"] is False
