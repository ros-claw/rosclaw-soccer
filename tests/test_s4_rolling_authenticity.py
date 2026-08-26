from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.physics.rolling_authenticity import (
    RollingAuthenticityThresholds,
    audit_pass_rolling_physics,
    measure_rolling_authenticity,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def _trace(*, rolling: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    time = np.linspace(0.0, 1.0, 51)
    radius = 0.11
    speed = 1.0
    pose = np.zeros((len(time), 7), dtype=np.float64)
    pose[:, 0] = speed * time
    pose[:, 2] = radius
    pose[:, 3] = 1.0
    velocity = np.zeros((len(time), 6), dtype=np.float64)
    velocity[:, 0] = speed
    if rolling:
        velocity[:, 4] = speed / radius
    return time, pose, velocity


def test_rolling_metric_distinguishes_roll_from_slide() -> None:
    time, pose, velocity = _trace(rolling=True)
    rolling, _ = measure_rolling_authenticity(
        time=time,
        ball_pose=pose,
        ball_velocity=velocity,
        ball_radius_m=0.11,
    )
    _, _, sliding_velocity = _trace(rolling=False)
    sliding, _ = measure_rolling_authenticity(
        time=time,
        ball_pose=pose,
        ball_velocity=sliding_velocity,
        ball_radius_m=0.11,
    )

    assert rolling.passed
    assert rolling.median_slip_ratio == pytest.approx(0.0)
    assert not sliding.passed
    assert sliding.median_slip_ratio == pytest.approx(1.0)


def test_rolling_metric_rejects_insufficient_ground_samples() -> None:
    time, pose, velocity = _trace(rolling=True)
    pose[:, 2] = 1.0
    with pytest.raises(ValueError, match="insufficient"):
        measure_rolling_authenticity(
            time=time,
            ball_pose=pose,
            ball_velocity=velocity,
            ball_radius_m=0.11,
        )


def test_rolling_thresholds_fail_closed() -> None:
    with pytest.raises(ValueError, match="p95"):
        RollingAuthenticityThresholds(
            maximum_median_slip_ratio=0.20,
            maximum_p95_slip_ratio=0.10,
        )


def test_rolling_audit_output_must_be_external(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="new path outside"):
        audit_pass_rolling_physics(
            asset_root=tmp_path / "assets",
            source_evidence_path=tmp_path / "evidence.json",
            output_dir=tmp_path / "inside",
            source_checkout=tmp_path,
        )


@pytest.mark.integration
def test_stadium_compiles_dimensionally_separate_ball_damping() -> None:
    mujoco = pytest.importorskip("mujoco")
    asset_root = Path("/code/rosclaw/phase4_references/RoboNaldo/RoboNaldo_Deploy")
    if not asset_root.is_dir():
        pytest.skip("external qualified RoboNaldo assets are unavailable")
    goal = G1TrainingGoalSpec()
    model = build_g1_stadium_model(asset_root, goal)
    ball_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    dof = int(model.jnt_dofadr[ball_joint])

    np.testing.assert_allclose(model.dof_damping[dof : dof + 3], 0.02)
    np.testing.assert_allclose(model.dof_damping[dof + 3 : dof + 6], 0.00002)
