from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.goalkeeper_dive_option import (
    GoalkeeperControlledDiveMonitor,
    GoalkeeperDiveDirection,
    GoalkeeperDiveOptionConfig,
    GoalkeeperDivePhase,
    build_balanced_dive_imitation_seed,
    load_official_goalkeeper_dive_atlas,
    mirror_g1_joint_positions,
)


def _step(
    monitor: GoalkeeperControlledDiveMonitor,
    *,
    request: bool,
    shot: int = 1,
    lateral_error: float = 0.8,
    pelvis: float = 0.79,
    upright: float = 1.0,
    linear: float = 0.0,
    angular: float = 0.0,
    landing: bool = False,
    forbidden: bool = False,
):
    return monitor.step(
        option_request=np.asarray([request], dtype=np.bool_),
        shot_index=np.asarray([shot], dtype=np.int64),
        lateral_intercept_error_m=np.asarray([lateral_error], dtype=np.float64),
        pelvis_height_m=np.asarray([pelvis], dtype=np.float64),
        upright_projection=np.asarray([upright], dtype=np.float64),
        root_linear_speed_mps=np.asarray([linear], dtype=np.float64),
        root_angular_speed_rad_s=np.asarray([angular], dtype=np.float64),
        permitted_landing_contact=np.asarray([landing], dtype=np.bool_),
        forbidden_body_contact=np.asarray([forbidden], dtype=np.bool_),
    )


def test_dive_contract_is_content_bound_and_sim_only() -> None:
    config = GoalkeeperDiveOptionConfig()
    assert config.config_hash.startswith("sha256:")
    assert config.maximum_option_steps == 70
    assert config.recovery_hold_steps == 10
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="settings"):
        replace(config, dive_minimum_upright_projection=-0.1)


def test_dive_monitor_only_accepts_far_active_threats() -> None:
    monitor = GoalkeeperControlledDiveMonitor(1)
    centre = _step(monitor, request=True, lateral_error=0.2)
    assert centre.phase[0] == GoalkeeperDivePhase.READY
    idle = _step(monitor, request=True, shot=0)
    assert idle.phase[0] == GoalkeeperDivePhase.READY
    far = _step(monitor, request=True, lateral_error=-0.8)
    assert bool(far.option_started_event[0])
    assert far.phase[0] == GoalkeeperDivePhase.TAKEOFF


def test_controlled_dive_temporarily_allows_tilt_but_requires_recovery() -> None:
    config = GoalkeeperDiveOptionConfig(control_dt_sec=0.05, recovery_hold_sec=0.10)
    monitor = GoalkeeperControlledDiveMonitor(1, config)
    _step(monitor, request=True)
    flight = _step(
        monitor,
        request=False,
        pelvis=0.48,
        upright=0.45,
        linear=1.0,
        angular=2.0,
    )
    assert flight.phase[0] == GoalkeeperDivePhase.FLIGHT
    assert bool(flight.posture_exception_granted[0])
    assert not bool(flight.unsafe[0])
    landing = _step(
        monitor,
        request=False,
        pelvis=0.42,
        upright=0.35,
        linear=0.6,
        angular=1.0,
        landing=True,
    )
    assert landing.phase[0] == GoalkeeperDivePhase.LANDING
    _step(monitor, request=False)
    recovered = _step(monitor, request=False)
    assert recovered.phase[0] == GoalkeeperDivePhase.COMPLETE
    assert bool(recovered.recovered_event[0])
    assert monitor.phase[0] == GoalkeeperDivePhase.READY
    assert monitor.completed_dives[0] == 1


def test_dive_monitor_fails_closed_on_envelope_or_unrecovered_second_shot() -> None:
    monitor = GoalkeeperControlledDiveMonitor(1)
    _step(monitor, request=True)
    unsafe = _step(
        monitor,
        request=False,
        pelvis=0.20,
        upright=-0.2,
        linear=2.8,
        angular=7.0,
    )
    assert unsafe.phase[0] == GoalkeeperDivePhase.FAILED
    assert bool(unsafe.unsafe[0])

    monitor.reset()
    _step(monitor, request=True)
    _step(monitor, request=False, pelvis=0.48, upright=0.45, linear=1.0, angular=2.0)
    next_shot = _step(
        monitor,
        request=False,
        shot=2,
        pelvis=0.48,
        upright=0.45,
        linear=1.0,
        angular=2.0,
    )
    assert next_shot.phase[0] == GoalkeeperDivePhase.FAILED
    assert bool(next_shot.unsafe[0])


def test_dive_monitor_rejects_forbidden_landing_contact() -> None:
    monitor = GoalkeeperControlledDiveMonitor(1)
    _step(monitor, request=True)
    result = _step(
        monitor,
        request=False,
        pelvis=0.48,
        upright=0.45,
        linear=1.0,
        angular=2.0,
        landing=True,
        forbidden=True,
    )
    assert bool(result.unsafe[0])
    assert not bool(result.posture_exception_granted[0])


def test_official_dive_atlas_maps_21_dof_clips_to_g1_29_dof() -> None:
    pytest.importorskip("torch")
    checkout = Path("/code/rosclaw/rosclaw_football/repos/Humanoid-Goalkeeper")
    if not checkout.is_dir():
        pytest.skip("pinned Humanoid-Goalkeeper checkout is unavailable")
    atlas = load_official_goalkeeper_dive_atlas(checkout=checkout)

    assert atlas.atlas_hash.startswith("sha256:")
    assert atlas.training_use_only
    assert not atlas.commercial_use_allowed
    assert {clip.direction for clip in atlas.clips} == set(GoalkeeperDiveDirection)
    for clip in atlas.clips:
        assert clip.joint_position_rad.shape[1] == 29
        assert clip.joint_velocity_rad_s.shape == clip.joint_position_rad.shape
        assert clip.duration_sec > 6.0
        assert not clip.joint_position_rad.flags.writeable
        # The source has 21 joints; waist roll/pitch and wrists stay at the
        # explicit official ready pose until a separately qualified model owns them.
        fixed = clip.joint_position_rad[:, (13, 14, 19, 20, 21)]
        assert np.count_nonzero(np.ptp(fixed, axis=0)) == 0


def test_balanced_dive_seed_mirrors_the_stronger_bounded_source_window() -> None:
    pytest.importorskip("torch")
    checkout = Path("/code/rosclaw/rosclaw_football/repos/Humanoid-Goalkeeper")
    if not checkout.is_dir():
        pytest.skip("pinned Humanoid-Goalkeeper checkout is unavailable")
    atlas = load_official_goalkeeper_dive_atlas(checkout=checkout)
    seed = build_balanced_dive_imitation_seed(atlas)

    assert seed.seed_hash.startswith("sha256:")
    assert seed.joint_position_rad.shape == (2, 71, 29)
    assert seed.root_displacement_m.shape == (2, 71, 3)
    assert seed.source_lateral_displacement_m > 0.40
    assert seed.source_start_frame == 193
    assert seed.source_end_frame == 263
    assert np.allclose(
        seed.joint_position_rad[1],
        mirror_g1_joint_positions(seed.joint_position_rad[0]),
    )
    assert np.allclose(
        mirror_g1_joint_positions(seed.joint_position_rad[1]),
        seed.joint_position_rad[0],
    )
    assert np.allclose(
        seed.root_displacement_m[0, :, 1],
        -seed.root_displacement_m[1, :, 1],
    )
    assert not seed.joint_position_rad.flags.writeable
    assert not seed.root_displacement_m.flags.writeable


def test_low_dive_seed_selects_downward_motion_without_losing_lateral_reach() -> None:
    pytest.importorskip("torch")
    checkout = Path("/code/rosclaw/rosclaw_football/repos/Humanoid-Goalkeeper")
    if not checkout.is_dir():
        pytest.skip("pinned Humanoid-Goalkeeper checkout is unavailable")
    atlas = load_official_goalkeeper_dive_atlas(checkout=checkout)
    lateral = build_balanced_dive_imitation_seed(atlas)
    low = build_balanced_dive_imitation_seed(atlas, window_profile="low_vertical_dip")

    assert low.window_profile == "low_vertical_dip"
    assert low.source_lateral_displacement_m >= 0.30
    assert np.min(low.root_displacement_m[0, :, 2]) < np.min(
        lateral.root_displacement_m[0, :, 2]
    )
    assert low.source_start_frame == 178
    assert low.source_end_frame == 248
