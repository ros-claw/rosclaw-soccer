from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.recovery_snapshot import RecoverySnapshot
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryBridgeTrial,
    RecoveryEntryMatcher,
    RecoveryEntrySearchConfig,
    RecoveryPerturbationConfig,
    RecoveryReferenceMotion,
    body_gravity_vector,
    build_recovery_bridge_schedule,
    build_recovery_perturbation_holdout,
    select_recovery_bridge_trial,
)

_HASH = "sha256:" + "a" * 64


def _snapshot() -> RecoverySnapshot:
    qpos = np.zeros(36, dtype=np.float64)
    qpos[2] = 0.10
    # 90 degrees about body y: body gravity points along +x.
    qpos[3:7] = (2**-0.5, 0.0, -(2**-0.5), 0.0)
    return RecoverySnapshot(
        episode_seed=51,
        environment_index=0,
        control_step=10,
        stage="RECOVERY_ENTRY",
        save_kind="BODY",
        posture_cluster="PRONE",
        qpos=qpos,
        qvel=np.zeros(35),
        applied_action=np.zeros(29),
        ball_position_m=np.zeros(3),
        ball_velocity_mps=np.zeros(3),
        target_position_m=np.zeros(3),
        left_foot_supported=False,
        right_foot_supported=False,
        failed=True,
        body_hash=_HASH,
        physics_scene_hash=_HASH,
        source_policy_hash=_HASH,
        source_config_hash=_HASH,
    )


def _motion(*, motion_id: str, gravity_sign: float = 1.0) -> RecoveryReferenceMotion:
    qpos = np.zeros((220, 36), dtype=np.float64)
    qvel = np.zeros((220, 35), dtype=np.float64)
    qpos[:, 3] = 1.0
    qpos[:80, 2] = 0.12
    angle = gravity_sign * np.pi / 2.0
    qpos[:80, 3] = np.cos(angle / 2.0)
    qpos[:80, 5] = -np.sin(angle / 2.0)
    qpos[80:, 2] = 0.75
    qpos[80:, 3] = 1.0
    qpos[80:, 5] = 0.0
    return RecoveryReferenceMotion(
        motion_id=motion_id,
        qpos=qpos,
        qvel=qvel,
        source_hash=_HASH,
    )


def _trial(*, match, succeeded: bool, dilation: int, peak: float) -> RecoveryBridgeTrial:
    return RecoveryBridgeTrial(
        snapshot_hash=_snapshot().snapshot_hash,
        match=match,
        teacher_policy_hash=_HASH,
        time_dilation=dilation,
        succeeded=succeeded,
        final_stable_sec=2.0 if succeeded else 0.0,
        executed_sec=18.0,
        peak_root_angular_speed_rad_s=peak,
        final_pelvis_height_m=0.72 if succeeded else 0.20,
        finite_state=True,
        ready_handoff_triggered=succeeded,
    )


def test_body_gravity_is_yaw_invariant_and_rejects_bad_quaternion() -> None:
    gravity = body_gravity_vector((2**-0.5, 0.0, -(2**-0.5), 0.0))
    assert gravity == pytest.approx((-1.0, 0.0, 0.0), abs=1e-12)
    with pytest.raises(ValueError, match="normalized"):
        body_gravity_vector((2.0, 0.0, 0.0, 0.0))


def test_reference_npz_loader_never_requires_object_metadata(tmp_path: Path) -> None:
    motion = _motion(motion_id="prone_teacher")
    path = tmp_path / "prone_teacher.npz"
    np.savez(
        path,
        qpos=motion.qpos,
        qvel=motion.qvel,
        frequency=np.asarray(50.0),
        metadata=np.asarray({"unsafe": "object"}, dtype=object),
    )
    loaded = RecoveryReferenceMotion.from_npz(path)
    assert loaded.motion_id == "prone_teacher"
    assert loaded.qpos.shape == (220, 36)


def test_matcher_requires_fallen_entry_with_upright_future() -> None:
    config = RecoveryEntrySearchConfig(
        candidate_stride_frames=2,
        minimum_future_offset_frames=10,
        maximum_future_offset_frames=200,
        successor_hold_frames=50,
        nonmaximum_spacing_frames=20,
    )
    matcher = RecoveryEntryMatcher(
        (
            _motion(motion_id="matching_prone"),
            _motion(motion_id="opposite_prone", gravity_sign=-1.0),
        ),
        config=config,
    )
    matches = matcher.match(_snapshot(), maximum_matches=2)
    assert matches[0].motion_id == "matching_prone"
    assert matches[0].gravity_distance == pytest.approx(0.0, abs=1e-12)
    assert matches[0].successor_end_frame > matches[0].entry_frame
    assert len(matches) == 2
    assert abs(matches[0].entry_frame - matches[1].entry_frame) >= 20


def test_bridge_selection_prefers_physics_then_less_intervention() -> None:
    matcher = RecoveryEntryMatcher(
        (_motion(motion_id="matching_prone"),),
        config=RecoveryEntrySearchConfig(
            minimum_future_offset_frames=10,
            maximum_future_offset_frames=200,
            successor_hold_frames=50,
        ),
    )
    match = matcher.match(_snapshot(), maximum_matches=1)[0]
    failed_close = _trial(match=match, succeeded=False, dilation=1, peak=0.1)
    passed_slow = _trial(match=match, succeeded=True, dilation=2, peak=4.0)
    passed_fast = _trial(match=match, succeeded=True, dilation=1, peak=8.0)
    assert select_recovery_bridge_trial((failed_close, passed_slow, passed_fast)) is passed_fast
    lower_peak = replace(passed_fast, peak_root_angular_speed_rad_s=2.0)
    schedule = build_recovery_bridge_schedule((failed_close, passed_slow, lower_peak))
    assert schedule["passed_snapshot_count"] == 1
    assert schedule["development_pass_rate"] == 1.0
    assert schedule["selected_trials"][0]["time_dilation"] == 1
    assert schedule["claim_boundary"] == "PRIVILEGED_TEACHER_DEVELOPMENT_NOT_DEPLOYABLE_GATE"


def test_bridge_trial_cannot_claim_success_without_ready_handoff() -> None:
    matcher = RecoveryEntryMatcher(
        (_motion(motion_id="matching_prone"),),
        config=RecoveryEntrySearchConfig(
            minimum_future_offset_frames=10,
            maximum_future_offset_frames=200,
            successor_hold_frames=50,
        ),
    )
    match = matcher.match(_snapshot(), maximum_matches=1)[0]
    with pytest.raises(ValueError, match="invalid"):
        RecoveryBridgeTrial(
            snapshot_hash=_snapshot().snapshot_hash,
            match=match,
            teacher_policy_hash=_HASH,
            time_dilation=1,
            succeeded=True,
            final_stable_sec=2.0,
            executed_sec=10.0,
            peak_root_angular_speed_rad_s=1.0,
            final_pelvis_height_m=0.7,
            finite_state=True,
            ready_handoff_triggered=False,
        )


def test_perturbation_holdout_is_deterministic_bounded_and_unseen() -> None:
    snapshot = _snapshot()
    config = RecoveryPerturbationConfig(samples_per_snapshot=3)
    first = build_recovery_perturbation_holdout((snapshot,), config=config)
    second = build_recovery_perturbation_holdout((snapshot,), config=config)
    assert len(first) == 3
    assert [item[1].perturbation_hash for item in first] == [
        item[1].perturbation_hash for item in second
    ]
    assert len({item[0].snapshot_hash for item in first}) == 3
    for perturbed, record in first:
        assert perturbed.snapshot_hash != snapshot.snapshot_hash
        assert record.base_snapshot_hash == snapshot.snapshot_hash
        assert record.perturbed_snapshot_hash == perturbed.snapshot_hash
        assert record.joint_position_linf_rad <= config.joint_position_half_width_rad
        assert record.joint_velocity_linf_rad_s <= config.joint_velocity_half_width_rad_s
        assert record.root_tilt_angle_rad <= 2**0.5 * config.root_tilt_half_width_rad
        assert (
            record.root_linear_velocity_linf_mps
            <= config.root_linear_velocity_half_width_mps
        )
        assert (
            record.root_angular_velocity_linf_rad_s
            <= config.root_angular_velocity_half_width_rad_s
        )


def test_perturbation_holdout_rejects_duplicate_base_snapshots() -> None:
    snapshot = _snapshot()
    with pytest.raises(ValueError, match="unique snapshots"):
        build_recovery_perturbation_holdout((snapshot, snapshot))
