from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.mosaic_agility_prior import (
    blend_g1_mosaic_agility_target,
    blend_g1_mosaic_agility_velocity,
    derive_g1_mosaic_agility_prior,
    load_g1_mosaic_agility_prior,
)
from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
)


def _dataset(root: Path) -> None:
    (root / "README.md").write_text(
        "---\nlicense: cdla-permissive-2.0\n---\n",
        encoding="utf-8",
    )
    output = root / "G1" / "optical_mocap"
    output.mkdir(parents=True)
    for skill in ("SE52", "SE56", "SE57", "SE63"):
        frames = 80
        velocity = np.zeros((frames, 29), dtype=np.float32)
        velocity[35:46, np.asarray(MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF[12:])] = 2.0
        np.savez(
            output / f"g1_{skill}_stageii.npz",
            fps=np.asarray((50,)),
            joint_pos=np.zeros((frames, 29), dtype=np.float32),
            joint_vel=velocity,
            body_lin_vel_w=np.zeros((frames, 30, 3), dtype=np.float32),
        )


def test_mosaic_prior_is_content_bound_and_sim_only(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)
    output = tmp_path / "evidence" / "prior.json"
    prior = derive_g1_mosaic_agility_prior(
        mosaic_root=root,
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )

    assert output.is_file()
    assert prior.prior_hash.startswith("sha256:")
    assert prior.activation_ceiling == "SIM_ONLY"
    assert not prior.promotion_authorized
    assert len(prior.selected_events) == 4
    assert {event.skill_id for event in prior.selected_events} == {
        "SE52",
        "SE56",
        "SE57",
        "SE63",
    }
    assert load_g1_mosaic_agility_prior(output) == prior


def test_mosaic_prior_loader_rejects_content_tampering(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)
    output = tmp_path / "evidence" / "prior.json"
    derive_g1_mosaic_agility_prior(
        mosaic_root=root,
        output_path=output,
        source_checkout=tmp_path / "checkout",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    payload["maximum_velocity_correction_rad_s"][12] = 999.0
    output.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="content hash mismatch"):
        load_g1_mosaic_agility_prior(output)


def test_mosaic_velocity_teacher_is_leg_isolated_and_bounded(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)
    prior = derive_g1_mosaic_agility_prior(
        mosaic_root=root,
        output_path=tmp_path / "evidence" / "prior.json",
        source_checkout=tmp_path / "checkout",
    )

    adapted, delta, active = blend_g1_mosaic_agility_velocity(
        target_velocity=np.zeros(29, dtype=np.float64),
        prior=prior,
        policy_frame=253,
        contact_policy_frame=253,
        control_dt_sec=0.02,
        blend=0.05,
        joint_scales=(0.0,) * 12 + (1.0,) * 17,
    )

    assert active
    np.testing.assert_array_equal(delta[:12], np.zeros(12))
    np.testing.assert_array_equal(adapted[:12], np.zeros(12))
    assert np.max(np.abs(delta)) <= 0.10 + 1e-12
    assert np.count_nonzero(delta[12:]) == 17


def test_mosaic_pose_teacher_returns_to_zero_at_both_endpoints(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)
    prior = derive_g1_mosaic_agility_prior(
        mosaic_root=root,
        output_path=tmp_path / "evidence" / "prior.json",
        source_checkout=tmp_path / "checkout",
        teacher_skill_id="SE63",
    )
    scales = (0.0,) * 12 + (1.0,) * 17

    _, start, start_active = blend_g1_mosaic_agility_target(
        target=np.zeros(29),
        prior=prior,
        policy_frame=243,
        contact_policy_frame=253,
        control_dt_sec=0.02,
        blend=0.50,
        joint_scales=scales,
    )
    _, middle, middle_active = blend_g1_mosaic_agility_target(
        target=np.zeros(29),
        prior=prior,
        policy_frame=253,
        contact_policy_frame=253,
        control_dt_sec=0.02,
        blend=0.50,
        joint_scales=scales,
    )
    _, end, end_active = blend_g1_mosaic_agility_target(
        target=np.zeros(29),
        prior=prior,
        policy_frame=263,
        contact_policy_frame=253,
        control_dt_sec=0.02,
        blend=0.50,
        joint_scales=scales,
    )

    assert not start_active
    assert middle_active
    assert not end_active
    np.testing.assert_allclose(start, 0.0, atol=1e-15)
    np.testing.assert_allclose(end, 0.0, atol=1e-15)
    np.testing.assert_array_equal(middle[:12], np.zeros(12))
    assert np.max(np.abs(middle)) <= 0.125 + 1e-12


def test_mosaic_prior_rejects_missing_license(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)
    (root / "README.md").write_text("no license declaration\n", encoding="utf-8")

    with pytest.raises(ValueError, match="CDLA"):
        derive_g1_mosaic_agility_prior(
            mosaic_root=root,
            output_path=tmp_path / "prior.json",
            source_checkout=tmp_path / "checkout",
        )


def test_mosaic_semantic_teacher_is_preselected_and_content_bound(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)
    output = tmp_path / "semantic-prior.json"

    prior = derive_g1_mosaic_agility_prior(
        mosaic_root=root,
        output_path=output,
        source_checkout=tmp_path / "checkout",
        teacher_skill_id="SE63",
    )

    assert prior.teacher_skill_id == "SE63"
    assert prior.schema_version == "rosclaw.growth.g1_mosaic_agility_prior.v3"
    assert load_g1_mosaic_agility_prior(output) == prior


def test_mosaic_semantic_teacher_rejects_undeclared_skill(tmp_path: Path) -> None:
    root = tmp_path / "mosaic"
    root.mkdir()
    _dataset(root)

    with pytest.raises(ValueError, match="not declared"):
        derive_g1_mosaic_agility_prior(
            mosaic_root=root,
            output_path=tmp_path / "semantic-prior.json",
            source_checkout=tmp_path / "checkout",
            teacher_skill_id="SE99",
        )
