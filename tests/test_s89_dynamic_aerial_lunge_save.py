from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.skills.team.shared_world import G1GoalkeeperConfig
from rosclaw_soccer.training import dynamic_aerial_lunge_save
from rosclaw_soccer.training.dynamic_aerial_lunge_save import (
    DynamicAerialLungeSaveConfig,
    dynamic_aerial_lunge_kwargs,
    evaluate_dynamic_aerial_lunge_save,
)


def test_dynamic_aerial_lunge_contract_is_bounded_and_honest() -> None:
    config = DynamicAerialLungeSaveConfig()
    assert config.activation_ceiling == "SIM_ONLY"
    assert config.commercial_use_allowed is False
    assert config.arm_scale == 0.0
    assert config.landing_capture_enabled is False
    assert config.outward_punch_force_scale == 0.0
    with pytest.raises(ValueError, match="arm skill"):
        replace(config, arm_scale=0.25)
    with pytest.raises(ValueError, match="non-commercial SIM_ONLY"):
        replace(config, commercial_use_allowed=True)
    with pytest.raises(ValueError, match="landing capture duration"):
        replace(config, landing_capture_sec=0.1)
    with pytest.raises(ValueError, match="outward punch"):
        replace(config, outward_punch_force_scale=0.76)
    with pytest.raises(ValueError, match="impact guard lead"):
        replace(config, joint_guard_impact_lead_sec=0.081)


def test_dynamic_lunge_requires_bound_sources(tmp_path: Path) -> None:
    missing = tmp_path / "missing"
    with pytest.raises(ValueError, match="dive source"):
        dynamic_aerial_lunge_kwargs(
            striker_actor_path=missing,
            goalkeeper_actor_path=missing,
            gmt_model_path=missing,
            gmt_skill_path=missing,
            dive_source_checkout=missing,
        )


def test_goalkeeper_dynamic_fields_preserve_default_authority() -> None:
    config = G1GoalkeeperConfig()
    assert config.balanced_dive_activation_lead_sec == 0.0
    assert config.balanced_dive_initial_phase == 0.0
    assert config.balanced_dive_lower_body_scale == 1.0
    assert config.balanced_dive_arm_scale == 1.0
    assert config.actor_bimanual_punch_vertical_force_scale == 0.0
    assert config.balanced_dive_landing_capture_enabled is False
    assert config.post_contact_stabilization_enabled is False


def test_dynamic_lunge_prefers_mujoco_root_velocity(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dynamic_aerial_lunge_save,
        "evaluate_three_role_aerial_save",
        lambda **_: {"passed": True, "gates": {"physical_save": True}},
    )
    time = np.arange(0.0, 2.02, 0.02, dtype=np.float64)
    pelvis = np.zeros((time.size, 7), dtype=np.float64)
    pelvis[:, 2] = 0.77
    pelvis[:, 3] = 1.0
    pelvis[10:21, 1] = np.linspace(0.0, 0.15, 11)
    blend = np.zeros_like(time)
    blend[10:21] = 1.0
    root_velocity = np.zeros((time.size, 6), dtype=np.float64)
    root_velocity[10:21, 1] = 0.60
    result = SimpleNamespace(
        goalkeeper_balanced_dive_seed_hash="sha256:seed",
        goalkeeper_balanced_dive_peak_blend=1.0,
    )
    report = evaluate_dynamic_aerial_lunge_save(  # type: ignore[arg-type]
        result=result,
        trajectory={
            "time": time,
            "goalkeeper_pelvis_pose": pelvis,
            "goalkeeper_root_velocity": root_velocity,
            "goalkeeper_balanced_dive_blend": blend,
        },
        config=DynamicAerialLungeSaveConfig(),
    )
    assert report["passed"] is True
    assert report["dynamic_metrics"]["velocity_authority"] == "MUJOCO_QVEL"
    assert report["dynamic_metrics"]["lunge_peak_lateral_speed_mps"] == pytest.approx(0.60)
