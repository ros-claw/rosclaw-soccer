from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.growth.mosaic_g1_contract import (
    MOSAIC_G1_ISAACLAB_TO_CANONICAL_BODY,
    MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF,
    MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
    canonicalize_mosaic_g1_bodies,
    canonicalize_mosaic_g1_joints,
)
from rosclaw_soccer.growth.mosaic_overhead_reach_prior import (
    G1MosaicOverheadReachEvent,
    G1MosaicOverheadReachPrior,
    blend_g1_mosaic_overhead_reach_target,
    load_g1_mosaic_overhead_reach_prior,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.training.goalkeeper_targeted_dive_mjwarp import (
    GoalkeeperTargetedDiveRLConfig,
)

_HASH = "sha256:" + "0" * 64


def _prior() -> G1MosaicOverheadReachPrior:
    times = np.linspace(-0.56, 0.66, 62)
    position = np.zeros((62, 29), dtype=np.float64)
    peak = int(np.argmin(np.abs(times)))
    position[:, 15] = -2.0 * np.exp(-np.square(times / 0.25))
    position[:, 22] = -2.0 * np.exp(-np.square(times / 0.25))
    velocity = np.gradient(position, times, axis=0)
    events = tuple(
        G1MosaicOverheadReachEvent(
            center_frame=100 + 100 * index,
            center_time_sec=2.0 + 2.0 * index,
            bilateral_hand_height_relative_pelvis_m=0.58,
            left_hand_world_height_m=1.36,
            right_hand_world_height_m=1.37,
            leg_velocity_rms_rad_s=0.08,
            root_planar_speed_mps=0.03,
        )
        for index in range(8)
    )
    assert abs(position[peak, 15]) > 1.9
    return G1MosaicOverheadReachPrior(
        dataset_readme_hash=_HASH,
        source_hash=_HASH,
        semantic_contract_hash=MOSAIC_G1_SEMANTIC_CONTRACT_HASH,
        physics_scene_hash=_HASH,
        body_hash=_HASH,
        joint_names=G1_DDS_JOINT_NAMES,
        reference_times_sec=tuple(float(value) for value in times),
        whole_body_position_reference_rad=tuple(
            tuple(float(value) for value in row) for row in position
        ),
        whole_body_velocity_reference_rad_s=tuple(
            tuple(float(value) for value in row) for row in velocity
        ),
        selected_events=events,
        forward_kinematics_mean_error_m=1.0e-7,
        forward_kinematics_maximum_error_m=8.0e-7,
        reference_peak_bilateral_hand_height_m=1.36,
    )


def test_mosaic_raw_orders_are_explicitly_canonicalized() -> None:
    joints = np.arange(29, dtype=np.float64).reshape(1, 29)
    bodies = np.arange(30 * 3, dtype=np.float64).reshape(1, 30, 3)

    np.testing.assert_array_equal(
        canonicalize_mosaic_g1_joints(joints)[0],
        joints[0, np.asarray(MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF)],
    )
    np.testing.assert_array_equal(
        canonicalize_mosaic_g1_bodies(bodies)[0],
        bodies[0, np.asarray(MOSAIC_G1_ISAACLAB_TO_CANONICAL_BODY)],
    )
    assert MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF[11] == 18
    assert MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF[15] == 11


def test_overhead_prior_is_content_bound_and_tamper_evident(tmp_path: Path) -> None:
    prior = _prior()
    path = tmp_path / "prior.json"
    path.write_text(json.dumps(prior.to_dict()), encoding="utf-8")

    assert load_g1_mosaic_overhead_reach_prior(path) == prior
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["whole_body_position_reference_rad"][20][15] = 99.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="invalid|mismatch"):
        load_g1_mosaic_overhead_reach_prior(path)


def test_overhead_blend_is_high_ball_only_smooth_and_group_bounded() -> None:
    prior = _prior()
    base = np.zeros(29, dtype=np.float64)
    inactive, inactive_delta, inactive_gate = blend_g1_mosaic_overhead_reach_target(
        target=base,
        prior=prior,
        time_to_arrival_sec=0.0,
        target_height_m=1.0,
        blend=1.0,
    )
    active, delta, gate = blend_g1_mosaic_overhead_reach_target(
        target=base,
        prior=prior,
        time_to_arrival_sec=0.0,
        target_height_m=1.35,
        blend=0.8,
    )
    endpoint, endpoint_delta, endpoint_gate = blend_g1_mosaic_overhead_reach_target(
        target=base,
        prior=prior,
        time_to_arrival_sec=0.56,
        target_height_m=1.35,
        blend=0.8,
    )

    np.testing.assert_array_equal(inactive, base)
    np.testing.assert_array_equal(inactive_delta, np.zeros(29))
    assert inactive_gate == 0.0
    np.testing.assert_array_equal(active[:12], np.zeros(12))
    assert gate == pytest.approx(0.8)
    assert 1.5 < abs(delta[15]) < 1.7
    np.testing.assert_array_equal(endpoint, base)
    np.testing.assert_array_equal(endpoint_delta, np.zeros(29))
    assert endpoint_gate == 0.0


def test_targeted_dive_overhead_prior_contract_is_fail_closed(tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.json"
    prior_path.write_text("{}", encoding="utf-8")
    config = GoalkeeperTargetedDiveRLConfig(
        overhead_reach_prior_path=str(prior_path.resolve()),
        overhead_reach_blend=0.8,
    )

    assert config.overhead_reach_blend == 0.8
    with pytest.raises(ValueError, match="requires a prior"):
        GoalkeeperTargetedDiveRLConfig(overhead_reach_blend=0.8)
    with pytest.raises(ValueError, match="two runtime reach teachers"):
        GoalkeeperTargetedDiveRLConfig(
            overhead_reach_prior_path=str(prior_path.resolve()),
            overhead_reach_blend=0.8,
            runtime_reach_blend=0.2,
        )
