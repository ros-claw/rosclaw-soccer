"""Bounded whole-cohort orchestration and current-body law checks, synthetic."""

import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from scripts import rsi_constant_forward_approach as mod
from tests.rsi.test_postcontact_release_protocol import arrays


def scope():
    return dict(
        profile="constant-forward-approach18-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        new_rollouts=18,
        native_step_budget=54000,
        forecast_step_budget=0,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        baseline_rollouts_reused=18,
        constant_forward_approach=True,
        forward_cap_m_s=1.4,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )


@pytest.mark.parametrize("key", list(scope()))
def test_strict_scope(key):
    mod.require_scope(scope())
    c = scope()
    c.pop(key)
    with pytest.raises(ValueError):
        mod.require_scope(c)


def trace():
    from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
    from rosclaw_soccer.rsi.original_ik_planner_pulse import OriginalIkPlannerPulse

    a, _ = arrays("baseline")
    a = {k: v for k, v in a.items() if not k.startswith("postcontact_release_")}
    a["canonical_qpos"] = np.zeros((300, 1, 43))
    a["canonical_qvel"] = np.zeros((300, 1, 41))
    a["root_pose_xyzw_m"] = np.zeros((300, 1, 7))
    a["ball_position_before_step_m"] = np.full((300, 1, 3), 0.5)
    a["ball_linear_velocity_before_step_m_s"] = np.zeros((300, 1, 3))
    a["force_n"] = np.zeros((300, 1, 6))
    a["force_n"][50, 0, 0] = 2.0
    for k in (
        "approach_requested_navigation_command",
        "navigation_command",
        "ik_planner_pulse_baseline_command",
        "ik_planner_pulse_requested_residual",
        "ik_planner_pulse_applied_command",
    ):
        a[k] = np.zeros((300, 1, 3))
    from rosclaw_soccer.rsi.constant_forward_approach import apply_constant_forward

    for f in range(300):
        raw, projected = BoundedFirstTouchControl().navigation(
            gap_x=0.5, gap_y=0.5, ball_vx=0.0, before_contact=f <= 50
        )
        raw, projected = apply_constant_forward(raw, projected, before_contact=f <= 50, cap=1.4)
        residual, applied = OriginalIkPlannerPulse(0.08).compose(f, projected)
        for k, v in (
            ("approach_requested_navigation_command", raw),
            ("navigation_command", applied),
            ("ik_planner_pulse_baseline_command", projected),
            ("ik_planner_pulse_requested_residual", residual),
            ("ik_planner_pulse_applied_command", applied),
        ):
            a[k][f, 0] = v
    return a


def test_law_audit_rejects_lateral_or_pulse_changes():
    a = trace()
    assert mod.approach_review(a, a)["first_changed_projected_command_frame"] is None
    b = {k: v.copy() for k, v in a.items()}
    a["navigation_command"][100, 0, 1] += 0.01
    with pytest.raises(ValueError, match="navigation audit"):
        mod.approach_review(a, b)


@pytest.mark.parametrize("failure_frame", [None, 5])
def test_full18_or_preserved_failure(monkeypatch, tmp_path, failure_frame):
    from rosclaw_soccer.providers.g1 import sonic_torch
    from rosclaw_soccer.rsi import (
        causal_impact_replay,
        first_touch_outcome,
        json_artifact_io,
        recurrent_sampling_episode_factory,
    )

    a = trace()
    outcome = dict(task_pass=True, safe=True)
    jobs = [
        dict(
            index=i,
            partition="CONSUMED_LOCAL_DEVELOPMENT" if i < 12 else "CONSUMED_ORIGINAL_RETENTION",
            source_index=i % 6,
            course=[2.5, 0.1, 0.3],
            baseline_hash="baseline",
            baseline_outcome=outcome,
            baseline_trace=a,
        )
        for i in range(18)
    ]
    case = dict(
        job=dict(heading_enabled=False),
        physics=dict(canonical_pd_kp=[1.0] * 29, canonical_pd_kd=[0.1] * 29),
        receipt=dict(
            receipt_hash="receipt",
            witnesses={k: dict(path=k) for k in ("world", "commitment", "report")},
        ),
    )
    p = SimpleNamespace(
        cfg=dict(
            scope(),
            deadline_unix=time.time() + 3600,
            mean_model_path="mean",
            mean_model_hash="mean",
        ),
        source=Path(mod.__file__).resolve().parent.parent,
        root=tmp_path / "output",
        jobs=jobs,
        cases=[case] * 6,
        pins={},
        declaration=dict(candidate=dict(parameters={})),
        world_hash="world",
        model_root=Path("models"),
    )
    monkeypatch.setattr(mod, "transfer_prepare", lambda path, **kw: p)
    monkeypatch.setattr(sys, "argv", ["probe", "--config", "fixture"])
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

    class Factory:
        def __init__(self, mean):
            assert mean["model_hash"] == "mean"

        def generate_and_bind_seed(self, *, seed):
            assert seed == 1
            return dict(policy_hash="policy"), object()

        def _stable(self):
            pass

    class Replay:
        def __init__(self, model, **kw):
            assert kw["constant_forward_approach"] is True and kw["impact_enabled"] is False
            assert "postcontact_exit_enabled" not in kw and "release_action_name" not in kw
            self.attempted_steps = 0
            self.data = SimpleNamespace(qpos=np.zeros(43), qvel=np.zeros(41), ctrl=np.zeros(29))

        def step(self, f):
            if f == failure_frame:
                raise RuntimeError("synthetic failure")
            self.attempted_steps += 10
            return SimpleNamespace(arrays={k: v[f] for k, v in a.items()})

        def review(self):
            return dict(
                native_integrated_steps=3000,
                forecast_integrated_steps=0,
                constant_forward_approach=True,
                original_precontact_forward_law_unchanged=False,
            )

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
        with pytest.raises(RuntimeError, match="synthetic failure"):
            mod.main()
        f = json.loads((p.root / "failure.json").read_text())
        assert f["completed_cases"] == 0 and f["current_attempted_steps"] == 50
        assert not f["automatic_retry"] and not (p.root / "complete.json").exists()
        return
    mod.main()
    result = json.loads((p.root / "complete.json").read_text())
    assert result["new_native_steps"] == 54000 and len(result["rows"]) == 18
    assert result["forecast_steps"] == result["new_optimizer_updates"] == 0
    assert not result["teacher_qualified"] and not result["training_authorized"]
