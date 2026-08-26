from __future__ import annotations

import json
from dataclasses import asdict, replace

import pytest

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_BODY_NAMES,
    MOSAIC_G1_ISAACLAB_JOINT_NAMES,
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
)
from rosclaw_soccer.growth.mosaic_getup import (
    G1MosaicGMTGetUpSkill,
    load_g1_mosaic_gmt_getup_skill,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES


def _skill() -> G1MosaicGMTGetUpSkill:
    count = 180
    digest = "sha256:" + "1" * 64
    return G1MosaicGMTGetUpSkill(
        checkpoint_hash=digest,
        checkpoint_contract_hash=digest,
        body_hash=digest,
        physics_scene_hash=digest,
        dataset_readme_hash=digest,
        source_hash=digest,
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        source_relative_path="G1/inertial_mocap/getup.npz",
        source_fps=50.0,
        source_start_frame=0,
        source_standing_frame=150,
        source_stop_frame=count,
        relative_times_sec=tuple(index / 50.0 for index in range(count)),
        raw_joint_position_rad=tuple((0.0,) * 29 for _ in range(count)),
        raw_joint_velocity_rad_s=tuple((0.0,) * 29 for _ in range(count)),
        aligned_torso_quaternion_wxyz=tuple((1.0, 0.0, 0.0, 0.0) for _ in range(count)),
        initial_pelvis_height_m=0.25,
        initial_upright_projection=0.0,
        final_pelvis_height_m=0.72,
        final_upright_projection=0.95,
    )


def test_mosaic_getup_skill_is_content_bound_and_no_pickle(tmp_path) -> None:
    skill = _skill()
    payload = asdict(skill)
    payload.update(
        joint_names=list(G1_DDS_JOINT_NAMES),
        raw_joint_names=list(MOSAIC_G1_ISAACLAB_JOINT_NAMES),
        raw_body_names=list(MOSAIC_G1_ISAACLAB_BODY_NAMES),
        skill_hash=skill.skill_hash,
    )
    path = tmp_path / "getup.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    loaded = load_g1_mosaic_gmt_getup_skill(path)
    assert loaded == skill
    assert loaded.duration_sec == pytest.approx(3.58)
    payload["final_pelvis_height_m"] = 0.73
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="content hash"):
        load_g1_mosaic_gmt_getup_skill(path)


def test_mosaic_getup_skill_rejects_wrong_time_scale_and_hardware() -> None:
    skill = _skill()
    with pytest.raises(ValueError, match="invalid or unqualified"):
        replace(skill, relative_times_sec=tuple(index / 100.0 for index in range(180)))
    with pytest.raises(ValueError, match="invalid or unqualified"):
        replace(skill, hardware_authorized=True)
