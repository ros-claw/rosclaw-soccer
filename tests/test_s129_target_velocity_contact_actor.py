from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.target_velocity_contact_actor import (
    fit_g1_target_velocity_contact_actor,
    load_g1_target_velocity_contact_actor,
    project_g1_target_velocity_contact_actor,
    save_g1_target_velocity_contact_actor,
)
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world

_HASH = "sha256:" + "1" * 64


def _actor():
    target = np.asarray(
        [
            (forward, lateral, vertical)
            for forward in (5.0, 7.0)
            for lateral in (-3.0, 3.0)
            for vertical in (-1.0, 3.0)
            for _ in range(4)
        ],
        dtype=np.float64,
    )
    velocity = np.column_stack(
        (
            np.linspace(-1.0, 2.0, len(target)),
            np.linspace(1.5, -1.5, len(target)),
            np.sin(np.linspace(0.0, 2.0, len(target))),
        )
    )
    gains = np.asarray((20.0, 18.0, 24.0))
    force = gains * (target - velocity)
    return fit_g1_target_velocity_contact_actor(
        target_velocity_xyz_mps=target,
        foot_velocity_xyz_mps=velocity,
        teacher_force_xyz_n=force,
        body_hash=_HASH,
        implementation_hash="sha256:" + "2" * 64,
        source_evidence_hashes=("sha256:" + "3" * 64,),
        training_trajectory_count=8,
        failed_trajectory_count=3,
        maximum_foot_ball_distance_m=0.50,
        start_policy_frame=230,
        end_policy_frame=335,
    )


def test_target_velocity_actor_learns_three_axes_and_projects_torque() -> None:
    actor = _actor()
    jacobian = np.zeros((3, 35), dtype=np.float64)
    jacobian[:, 6:9] = np.eye(3)
    effect = project_g1_target_velocity_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=np.zeros(35, dtype=np.float64),
        target_velocity_xyz_mps=np.asarray((6.0, 2.0, 1.0)),
        actor=actor,
    )

    assert effect.active
    assert effect.target_supported
    assert effect.force_xyz_n == pytest.approx((120.0, 36.0, 24.0), abs=0.1)
    assert effect.torque[:3] == pytest.approx(effect.force_xyz_n, abs=1.0e-9)
    assert np.count_nonzero(effect.torque[3:]) == 0
    assert actor.distillation_rmse_n < 0.01


def test_target_velocity_actor_fails_closed_outside_learned_envelope() -> None:
    actor = _actor()
    effect = project_g1_target_velocity_contact_actor(
        jacobian_position=np.zeros((3, 35), dtype=np.float64),
        generalized_velocity=np.zeros(35, dtype=np.float64),
        target_velocity_xyz_mps=np.asarray((8.0, 0.0, 0.0)),
        actor=actor,
    )

    assert not effect.active
    assert not effect.target_supported
    assert np.count_nonzero(effect.torque) == 0


def test_zero_target_preserves_teacher_axis_disable_semantics() -> None:
    actor = _actor()
    jacobian = np.zeros((3, 35), dtype=np.float64)
    jacobian[:, 6:9] = np.eye(3)
    effect = project_g1_target_velocity_contact_actor(
        jacobian_position=jacobian,
        generalized_velocity=np.zeros(35, dtype=np.float64),
        target_velocity_xyz_mps=np.asarray((6.0, 0.0, 0.0)),
        actor=actor,
    )

    assert effect.target_supported
    assert effect.force_xyz_n[0] > 0.0
    np.testing.assert_array_equal(effect.force_xyz_n[1:], 0.0)
    np.testing.assert_array_equal(effect.torque[1:], 0.0)


def test_target_velocity_actor_is_content_bound(tmp_path: Path) -> None:
    path = tmp_path / "actor.json"
    actor = _actor()
    save_g1_target_velocity_contact_actor(actor, path)
    assert load_g1_target_velocity_contact_actor(path).actor_hash == actor.actor_hash

    path.write_text(
        path.read_text().replace('"distillation_rmse_n":', '"x": 1, "distillation_rmse_n":')
    )
    with pytest.raises(TypeError):
        load_g1_target_velocity_contact_actor(path)

    with pytest.raises(ValueError, match="SIM-only contract"):
        replace(actor, hardware_authorized=True)


def test_shared_world_requires_actor_and_committed_target_together(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "rosclaw_soccer.skills.team.shared_world.qualify_g1_assets",
        lambda _path: SimpleNamespace(asset_root=tmp_path, require_eligible=lambda: None),
    )
    with pytest.raises(ValueError, match="configured together"):
        simulate_shared_world(
            tmp_path / "missing-assets",
            shooter_target_velocity_contact_actor_path=tmp_path / "actor.json",
        )
    with pytest.raises(ValueError, match="configured together"):
        simulate_shared_world(
            tmp_path / "missing-assets",
            shooter_target_foot_velocity_xyz_mps=(5.0, 0.0, 0.0),
        )
