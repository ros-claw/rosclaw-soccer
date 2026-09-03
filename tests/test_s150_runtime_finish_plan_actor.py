from __future__ import annotations

import inspect
from pathlib import Path

import pytest

from rosclaw_soccer.growth.runtime_contact_target_actor import RuntimeContactTargetAction
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    G1RuntimeFinishPlanActor,
    RuntimeFinishPlanAction,
    RuntimeFinishPlanMemory,
    load_runtime_finish_plan_actor,
    save_runtime_finish_plan_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.media.runtime_finish_plan_video import (
    validate_runtime_finish_plan_video_manifest,
)
from rosclaw_soccer.skills.team.shared_world import _simulate_shared_world
from rosclaw_soccer.training.runtime_finish_plan_exam import (
    fresh_runtime_finish_plan_holdouts,
    fresh_runtime_finish_plan_holdouts_v2,
    validate_runtime_finish_plan_exam,
)
from rosclaw_soccer.training.runtime_finish_plan_growth import (
    _row_action,
    train_runtime_finish_plan_actor,
)

_CURRENT_EVIDENCE = Path(
    "/code/rosclaw/rosclaw_football/evidence/athlete-foundation-v1/"
    "s167-prepared-finish-plan-sealed-fresh-holdout-v2/exam-report.json"
)
_CURRENT_VIDEO = Path(
    "/code/rosclaw/rosclaw_football/videos/s167-prepared-finish-plan-growth-v2.json"
)


def _plan(*, stance_y: float, frame: int, target_y: float) -> RuntimeFinishPlanAction:
    return RuntimeFinishPlanAction(
        receive=RuntimeReceiveAction(
            maximum_arrival_advance_frames=18,
            stance_offset_x_m=-0.08,
            stance_offset_y_m=stance_y,
            contact_policy_frame=frame,
            foot_yaw_offset_rad=-0.04,
            foot_pitch_offset_rad=0.01,
        ),
        target=RuntimeContactTargetAction((9.0, target_y, -1.0)),
    )


def _memory(
    index: int, *, feature: float, action: RuntimeFinishPlanAction
) -> RuntimeFinishPlanMemory:
    digit = format(index, "x")
    return RuntimeFinishPlanMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "0" * 62 + f"{index:02x}",
        features=(3.7, feature, -1.8, -0.05, 1.3, -0.05, -0.3, 0.08, 176.0),
        action=action,
        quality_score=10.0 if index < 5 else 2.0,
    )


def _actor() -> G1RuntimeFinishPlanActor:
    early = _plan(stance_y=-0.03, frame=248, target_y=5.0)
    late = _plan(stance_y=-0.06, frame=258, target_y=6.0)
    return G1RuntimeFinishPlanActor(
        body_hash="sha256:" + "a" * 64,
        kick_prior_hash="sha256:" + "b" * 64,
        roster_hash="sha256:" + "c" * 64,
        finisher_self_model_hash="sha256:" + "d" * 64,
        neural_contact_actor_hash="sha256:" + "e" * 64,
        contact_handoff_actor_hash="sha256:" + "2" * 64,
        contact_handoff_offset_frames=7,
        source_evidence_hashes=("sha256:" + "f" * 64,),
        training_snapshot_hash="sha256:" + "1" * 64,
        feature_center=(0.0,) * 9,
        feature_scale=(1.0,) * 9,
        successful_memories=(
            _memory(1, feature=-0.02, action=early),
            _memory(2, feature=-0.03, action=early),
            _memory(3, feature=-0.12, action=late),
            _memory(4, feature=-0.13, action=late),
        ),
        failed_memories=tuple(
            _memory(index, feature=0.30 + index * 0.01, action=early if index < 9 else late)
            for index in range(5, 13)
        ),
    )


def test_runtime_finish_plan_selects_coupled_geometry_phase_and_target() -> None:
    actor = _actor()

    decision = actor.decide((3.7, -0.02, -1.8, -0.05, 1.3, -0.05, -0.3, 0.08, 176.0))

    assert decision.accepted and decision.action is not None
    assert decision.action.receive.contact_policy_frame == 248
    assert decision.action.receive.stance_offset_y_m == -0.03
    assert decision.action.target.target_foot_velocity_xyz_mps == (9.0, 5.0, -1.0)
    assert not decision.action.direct_joint_torque_output
    assert actor.owned_skill == "receive_and_strike_plan"


def test_runtime_finish_plan_rejects_ood_without_motor_authority() -> None:
    actor = _actor()

    decision = actor.decide((30.0,) * 9)

    assert not decision.accepted
    assert decision.action is None
    assert decision.route == "RUNTIME_FINISH_PLAN_OOD_FALLBACK"
    assert not actor.direct_joint_torque_output
    with pytest.raises(ValueError, match="high-level SIM-only"):
        RuntimeFinishPlanAction(
            receive=_plan(stance_y=-0.03, frame=248, target_y=5.0).receive,
            target=RuntimeContactTargetAction((9.0, 5.0, -1.0)),
            direct_joint_torque_output=True,
        )


def test_runtime_finish_plan_round_trip_is_content_bound(tmp_path: Path) -> None:
    actor = _actor()
    path = tmp_path / "finish-plan.json"
    save_runtime_finish_plan_actor(actor, path)

    assert load_runtime_finish_plan_actor(path) == actor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"maximum_support_distance": 2.75', '"maximum_support_distance": 2.5'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_runtime_finish_plan_actor(path)


def test_shared_world_exposes_only_actor_path_for_runtime_finish_plan() -> None:
    parameters = inspect.signature(_simulate_shared_world).parameters

    assert "shooter_runtime_finish_plan_actor_path" in parameters
    assert "shooter_runtime_finish_plan_action" not in parameters
    assert tuple(inspect.signature(train_runtime_finish_plan_actor).parameters) == (
        "base_target_actor_path",
        "neural_actor_path",
        "handoff_actor_path",
        "source_s95_dir",
        "source_report_paths",
        "output_dir",
    )


def test_runtime_finish_plan_fresh_holdouts_are_preregistered_and_unique() -> None:
    holdouts = fresh_runtime_finish_plan_holdouts()
    holdouts_v2 = fresh_runtime_finish_plan_holdouts_v2()

    assert len(holdouts) == 6
    assert len({context.context_hash for context, _ in holdouts}) == 6
    assert all(context.case_id.startswith("s152.finish.") for context, _ in holdouts)
    assert {action.body_yaw_correction_rad for _, action in holdouts} == {0.02, 0.04, 0.06}
    assert len(holdouts_v2) == 6
    assert len({context.context_hash for context, _ in holdouts_v2}) == 6
    assert all(context.case_id.startswith("s167.finish.") for context, _ in holdouts_v2)
    assert {context.context_hash for context, _ in holdouts}.isdisjoint(
        context.context_hash for context, _ in holdouts_v2
    )


def test_prepared_repair_row_restores_the_complete_joint_action() -> None:
    expected = _plan(stance_y=0.06, frame=252, target_y=0.0)

    restored = _row_action(
        schema="rosclaw_soccer.prepared_finish_plan_repair.v1",
        row={
            "action": {
                "receive": {
                    "maximum_arrival_advance_frames": 18,
                    "arrival_alignment_tolerance_sec": 0.02,
                    "stance_offset_x_m": -0.08,
                    "stance_offset_y_m": 0.06,
                    "contact_policy_frame": 252,
                    "foot_yaw_offset_rad": -0.04,
                    "foot_pitch_offset_rad": 0.01,
                    "activation_ceiling": "SIM_ONLY",
                    "direct_joint_torque_output": False,
                },
                "target": {
                    "target_foot_velocity_xyz_mps": [9.0, 0.0, -1.0],
                    "activation_ceiling": "SIM_ONLY",
                    "direct_joint_torque_output": False,
                },
                "activation_ceiling": "SIM_ONLY",
                "direct_joint_torque_output": False,
            }
        },
        base_receive=RuntimeReceiveAction(),
    )

    assert restored == expected


def test_current_sealed_finish_plan_evidence_is_fully_bound_when_available() -> None:
    if not _CURRENT_EVIDENCE.is_file():
        pytest.skip("current S167 prepared-finish evidence is unavailable")

    report = validate_runtime_finish_plan_exam(_CURRENT_EVIDENCE)

    assert report["status"] == "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT"
    assert report["metrics"]["candidate_strict_success_count"] >= 3
    assert report["metrics"]["strict_success_gain"] >= 2
    assert report["metrics"]["candidate_safe_count"] == 6
    assert report["metrics"]["exact_replay_count"] == 6


def test_current_finish_plan_video_is_evidence_downstream_when_available() -> None:
    if not _CURRENT_VIDEO.is_file():
        pytest.skip("current S167 prepared-finish video is unavailable")

    manifest = validate_runtime_finish_plan_video_manifest(_CURRENT_VIDEO)

    assert (
        manifest["source_exam_hash"]
        == validate_runtime_finish_plan_exam(_CURRENT_EVIDENCE)["report_hash"]
    )
    assert manifest["pixels_used_for_scoring"] is False
    assert manifest["commercial_use_allowed"] is False
