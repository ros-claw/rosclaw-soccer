from __future__ import annotations

import inspect
import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from rosclaw_soccer.training.opentrack_recovery_student_exam import (
    _run_student_trial,
)
from rosclaw_soccer.training.recovery_student import (
    RecoveryProprioceptionSpec,
    RecoveryTeacherEpisode,
    build_recovery_proprioception,
    denormalize_absolute_motor_targets,
    load_recovery_distillation_corpus,
    normalize_absolute_motor_targets,
    recovery_teacher_episodes_from_corpus,
    write_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_student_train import (
    RecoveryStudentTrainingConfig,
    build_recovery_student_samples,
    split_recovery_student_episodes,
)

_HASH = "sha256:" + "a" * 64


def _episode() -> RecoveryTeacherEpisode:
    proprio = np.arange(120 * 93, dtype=np.float32).reshape(120, 93) / 10_000.0
    targets = np.linspace(-0.5, 0.5, 120 * 29, dtype=np.float32).reshape(120, 29)
    return RecoveryTeacherEpisode(
        episode_id="teacher-000",
        base_snapshot_hash=_HASH,
        initial_snapshot_hash="sha256:" + "b" * 64,
        fixed_route_trial_hash="sha256:" + "c" * 64,
        perturbation_hash="sha256:" + "d" * 64,
        proprio=proprio,
        absolute_motor_targets_rad=targets,
        ready_handoff=np.arange(120) >= 100,
        time_dilation=1,
        teacher_succeeded=True,
    )


def test_proprioception_has_no_reference_or_teacher_features() -> None:
    spec = RecoveryProprioceptionSpec(history_steps=6)
    assert not set(spec.features) & set(spec.forbidden_features)
    frame = build_recovery_proprioception(
        projected_gravity_body=(0.0, 0.0, -1.0),
        pelvis_gyro_rad_s=(1.0, 2.0, 3.0),
        joint_position_rad=np.full(29, 0.4),
        joint_velocity_rad_s=np.full(29, 2.0),
        last_motor_target_rad=np.full(29, 0.3),
        default_joint_position_rad=np.full(29, 0.1),
        spec=spec,
    )
    assert frame.shape == (93,)
    assert frame[:3] == pytest.approx((0.0, 0.0, -1.0))
    assert frame[3:6] == pytest.approx((0.05, 0.10, 0.15))
    assert frame[6:35] == pytest.approx(np.full(29, 0.3))
    assert frame[35:64] == pytest.approx(np.full(29, 0.1))
    assert frame[64:93] == pytest.approx(np.full(29, 0.3))


def test_proprioception_rejects_non_unit_gravity() -> None:
    with pytest.raises(ValueError, match="normalized"):
        build_recovery_proprioception(
            projected_gravity_body=(0.0, 0.0, -2.0),
            pelvis_gyro_rad_s=np.zeros(3),
            joint_position_rad=np.zeros(29),
            joint_velocity_rad_s=np.zeros(29),
            last_motor_target_rad=np.zeros(29),
            default_joint_position_rad=np.zeros(29),
        )


def test_absolute_motor_target_normalization_is_bounded_and_round_trips() -> None:
    lower = np.linspace(-2.0, -1.0, 29, dtype=np.float32)
    upper = np.linspace(1.0, 2.0, 29, dtype=np.float32)
    targets = np.stack((lower, 0.25 * lower + 0.75 * upper, upper))
    normalized = normalize_absolute_motor_targets(
        targets, joint_lower_rad=lower, joint_upper_rad=upper
    )
    assert np.max(np.abs(normalized)) <= 1.0
    restored = denormalize_absolute_motor_targets(
        normalized, joint_lower_rad=lower, joint_upper_rad=upper
    )
    assert restored == pytest.approx(targets, abs=1e-6)


def test_distillation_corpus_round_trip_and_tamper_rejection(tmp_path: Path) -> None:
    spec = RecoveryProprioceptionSpec(history_steps=8)
    episode = _episode()
    output = tmp_path / "corpus"
    manifest = write_recovery_distillation_corpus(
        episodes=(episode,),
        output_dir=output,
        corpus_name="recovery-train-v1",
        proprioception_spec=spec,
        teacher_policy_hash=_HASH,
        body_hash=_HASH,
        physics_scene_hash=_HASH,
        development_report_hash=_HASH,
        default_joint_position_rad=np.zeros(29),
        joint_lower_rad=np.full(29, -2.0),
        joint_upper_rad=np.full(29, 2.0),
    )
    path = output / "recovery-train-v1.json"
    loaded = load_recovery_distillation_corpus(path)
    assert loaded.manifest_hash == manifest["manifest_hash"]
    assert loaded.proprio.shape == (120, 93)
    assert loaded.rows[0]["episode_hash"] == episode.episode_hash

    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["contains_reference_features"] = True
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="integrity"):
        load_recovery_distillation_corpus(path)


def test_teacher_episode_rejects_unsuccessful_or_short_rollout() -> None:
    episode = _episode()
    with pytest.raises(ValueError, match="invalid"):
        replace(episode, teacher_succeeded=False)
    with pytest.raises(ValueError, match="invalid"):
        replace(
            episode,
            proprio=episode.proprio[:50],
            absolute_motor_targets_rad=episode.absolute_motor_targets_rad[:50],
            ready_handoff=episode.ready_handoff[:50],
        )


def test_episode_split_keeps_same_initial_state_on_one_side(tmp_path: Path) -> None:
    original = _episode()
    repeated = replace(
        original,
        episode_id="dagger1-000",
        rollout_controller="MIXED_STUDENT_TEACHER",
        rollout_succeeded=False,
    )
    second_initial = replace(
        original,
        episode_id="teacher-001",
        initial_snapshot_hash="sha256:" + "e" * 64,
        perturbation_hash="sha256:" + "f" * 64,
        proprio=original.proprio + 0.1,
    )
    repeated_second = replace(
        second_initial,
        episode_id="dagger1-001",
        rollout_controller="MIXED_STUDENT_TEACHER",
    )
    output = tmp_path / "grouped"
    write_recovery_distillation_corpus(
        episodes=(original, repeated, second_initial, repeated_second),
        output_dir=output,
        corpus_name="grouped-recovery",
        proprioception_spec=RecoveryProprioceptionSpec(),
        teacher_policy_hash=_HASH,
        body_hash=_HASH,
        physics_scene_hash=_HASH,
        development_report_hash=_HASH,
        default_joint_position_rad=np.zeros(29),
        joint_lower_rad=np.full(29, -2.0),
        joint_upper_rad=np.full(29, 2.0),
    )
    corpus = load_recovery_distillation_corpus(output / "grouped-recovery.json")
    training, validation = split_recovery_student_episodes(corpus)
    assert set(training).isdisjoint(validation)
    for initial_hash in {
        original.initial_snapshot_hash,
        second_initial.initial_snapshot_hash,
    }:
        indexes = {
            int(row["episode_index"])
            for row in corpus.rows
            if row["initial_snapshot_hash"] == initial_hash
        }
        assert indexes <= set(training) or indexes <= set(validation)
    samples = build_recovery_student_samples(corpus, episode_indexes=training)
    assert samples.history.shape[1:] == (8, 93)
    assert np.count_nonzero(samples.control_step == 0) == len(training)
    restored = recovery_teacher_episodes_from_corpus(corpus)
    assert restored[1].rollout_controller == "MIXED_STUDENT_TEACHER"
    assert restored[1].rollout_succeeded is False


def test_training_config_rejects_authority_or_invalid_curriculum() -> None:
    with pytest.raises(ValueError, match="invalid"):
        RecoveryStudentTrainingConfig(initial_window_steps=0)
    with pytest.raises(ValueError, match="invalid"):
        RecoveryStudentTrainingConfig(hardware_authorized=True)


def test_direct_student_loop_has_no_reference_or_environment_step_reads() -> None:
    source = inspect.getsource(_run_student_trial)
    for forbidden in ("env.step(", ".th.", "ref_mj_data", "reference_phase"):
        assert forbidden not in source
