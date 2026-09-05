from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from rosclaw_soccer.growth.runtime_contact_target_actor import (
    G1RuntimeContactTargetActor,
    RuntimeContactTargetAction,
    RuntimeContactTargetMemory,
    load_runtime_contact_target_actor,
    save_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.skills.team.shared_world import _simulate_shared_world
from rosclaw_soccer.training.runtime_contact_target_exam import (
    fresh_runtime_contact_target_holdouts,
    validate_runtime_contact_target_exam,
)
from rosclaw_soccer.training.runtime_contact_target_growth import (
    train_runtime_contact_target_actor,
)
from rosclaw_soccer.training.runtime_contact_target_repair import (
    default_runtime_contact_target_repair_probes,
)

_HASHES = tuple("sha256:" + character * 64 for character in "abcdef")


def _memory(
    index: int,
    lateral_feature: float,
    target_y: float,
    quality: float,
) -> RuntimeContactTargetMemory:
    digit = format(index, "x")
    return RuntimeContactTargetMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "0" * 63 + digit,
        features=(3.7, lateral_feature, -1.8, -0.05, 1.3, -0.05, -0.3, 0.08, 176.0),
        action=RuntimeContactTargetAction((9.0, target_y, -1.0)),
        quality_score=quality,
    )


def _actor() -> G1RuntimeContactTargetActor:
    return G1RuntimeContactTargetActor(
        body_hash=_HASHES[0],
        kick_prior_hash=_HASHES[1],
        roster_hash=_HASHES[2],
        finisher_self_model_hash=_HASHES[3],
        neural_contact_actor_hash=_HASHES[4],
        source_evidence_hashes=(_HASHES[5],),
        training_snapshot_hash="sha256:" + "1" * 64,
        required_receive_action=RuntimeReceiveAction(
            maximum_arrival_advance_frames=18,
            arrival_alignment_tolerance_sec=0.02,
            stance_offset_x_m=-0.08,
            stance_offset_y_m=-0.06,
            contact_policy_frame=258,
            foot_yaw_offset_rad=-0.04,
            foot_pitch_offset_rad=0.01,
        ),
        feature_center=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        successful_memories=(
            _memory(1, -0.02, 2.0, 8.0),
            _memory(2, -0.02, 3.0, 10.0),
            _memory(3, -0.12, 5.0, 9.0),
            _memory(4, -0.12, 6.0, 11.0),
        ),
        failed_memories=(
            _memory(5, -0.12, 3.0, 2.0),
            _memory(6, -0.02, 5.0, 2.0),
        ),
    )


def test_runtime_contact_target_selects_best_local_verified_target() -> None:
    actor = _actor()

    near_center = actor.decide((3.7, -0.02, -1.8, -0.05, 1.3, -0.05, -0.3, 0.08, 176.0))
    near_wide = actor.decide((3.7, -0.12, -1.8, -0.05, 1.3, -0.05, -0.3, 0.08, 176.0))

    assert near_center.accepted and near_center.action is not None
    assert near_center.action.target_foot_velocity_xyz_mps == (9.0, 3.0, -1.0)
    assert near_wide.accepted and near_wide.action is not None
    assert near_wide.action.target_foot_velocity_xyz_mps == (9.0, 6.0, -1.0)
    assert actor.owned_skill == "contact_target_selection"
    assert not actor.direct_joint_torque_output


def test_runtime_contact_target_rejects_ood_and_failure_dominated_memory() -> None:
    actor = _actor()

    ood = actor.decide((30.0,) * 9)
    between_conflict = actor.decide((3.7, -0.069, -1.8, -0.05, 1.3, -0.05, -0.3, 0.08, 176.0))

    assert not ood.accepted and ood.action is None
    assert ood.route == "CONTACT_TARGET_OOD_FALLBACK"
    assert not between_conflict.accepted and between_conflict.action is None
    assert between_conflict.route == "CONTACT_TARGET_FAILURE_MEMORY_FALLBACK"


def test_runtime_contact_target_round_trip_is_content_bound(tmp_path: Path) -> None:
    actor = _actor()
    path = tmp_path / "target-actor.json"
    save_runtime_contact_target_actor(actor, path)

    assert load_runtime_contact_target_actor(path) == actor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"maximum_support_distance": 2.75',
            '"maximum_support_distance": 2.5',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_runtime_contact_target_actor(path)


def test_runtime_target_action_cannot_claim_direct_torque_authority() -> None:
    with pytest.raises(ValueError, match="task envelope"):
        RuntimeContactTargetAction((9.0, 3.0, -1.0), direct_joint_torque_output=True)
    with pytest.raises(ValueError, match="task envelope"):
        RuntimeContactTargetAction((13.0, 3.0, -1.0))


def test_shared_world_exposes_actor_path_but_no_runtime_target_override() -> None:
    parameters = inspect.signature(_simulate_shared_world).parameters

    assert "shooter_runtime_contact_target_actor_path" in parameters
    assert "shooter_runtime_contact_target_action" not in parameters


def test_s142_holdouts_are_unique_and_not_teacher_conditions() -> None:
    holdouts = fresh_runtime_contact_target_holdouts()

    assert len(holdouts) == 6
    assert len({context.context_hash for context, _ in holdouts}) == 6
    assert all(context.case_id.startswith("s142.holdout.") for context, _ in holdouts)
    teacher_conditions = {
        (1.209, -0.165, 0.1005),
        (1.214, -0.170, 0.1040),
    }
    assert teacher_conditions.isdisjoint(
        (
            context.passer_ball_local_xy_m[0],
            context.passer_ball_local_xy_m[1],
            context.ball_ground_friction,
        )
        for context, _ in holdouts
    )
    assert sum(action.stance_correction_x_m == -0.02 for _, action in holdouts) == 3


def test_s142_training_and_exam_surfaces_are_explicit() -> None:
    assert tuple(inspect.signature(train_runtime_contact_target_actor).parameters) == (
        "teacher_report_path",
        "neural_training_report_path",
        "neural_actor_path",
        "role_source_report_path",
        "output_dir",
        "additional_report_paths",
    )
    assert tuple(inspect.signature(validate_runtime_contact_target_exam).parameters) == ("path",)


def test_repair_curriculum_changes_only_supported_task_targets() -> None:
    cases = fresh_runtime_contact_target_holdouts()
    probes = default_runtime_contact_target_repair_probes(cases)

    assert len(probes) == 24
    assert len({probe.probe_hash for probe in probes}) == 24
    assert {probe.context.context_hash for probe in probes} == {
        context.context_hash for context, _ in cases
    }
    assert {probe.action.target_foot_velocity_xyz_mps for probe in probes} == {
        (10.0, 3.0, -1.0),
        (9.0, 4.0, -1.0),
        (9.0, 5.0, -1.0),
        (9.0, 6.0, -1.0),
    }
    assert all(not probe.action.direct_joint_torque_output for probe in probes)
