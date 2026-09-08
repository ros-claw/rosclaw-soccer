from __future__ import annotations

import math

import numpy as np
import pytest

from rosclaw_soccer.growth.contextual_strike_experts import (
    ContextualStrikeExpert,
    ContextualStrikeExpertMemory,
    ContextualStrikeNegative,
    StrikeTaskContext,
    build_strike_task_context,
)
from rosclaw_soccer.growth.dynamic_strike_coordination import (
    DynamicStrikeCoordinationActor,
    StrikeCoordinationObservation,
)
from rosclaw_soccer.training.contextual_strike_expert_exam import (
    _baseline_row,
    _exact_replay,
)
from rosclaw_soccer.training.contextual_strike_expert_probe import _route_summary


def _context(target_y: float) -> StrikeTaskContext:
    return StrikeTaskContext(
        goal_target_y_m=target_y,
        ball_lateral_y_m=-0.777,
        nearest_opponent_distance_m=4.033,
        shot_line_clearance_m=1.23 + 0.77 * (target_y - 0.8),
        shot_line_progress=0.803 - 0.10 * (target_y - 0.8),
    )


def _observation() -> StrikeCoordinationObservation:
    return StrikeCoordinationObservation(
        phase_progress=0.4,
        stance_depth_m=0.42,
        stance_lateral_error_m=0.31,
        stance_yaw_error_rad=0.72,
        approach_yaw_error_rad=0.24,
        ball_speed_mps=0.45,
    )


def _expert(index: int, target_y: float, stance: float) -> ContextualStrikeExpert:
    return ContextualStrikeExpert(
        expert_id=f"expert-{index}",
        context_center=tuple(float(value) for value in _context(target_y).vector()),
        actor=DynamicStrikeCoordinationActor.constant(
            stance_blend=stance,
            goal_yaw_blend=0.28,
        ),
        source_report_hash="sha256:" + f"{index + 1:x}" * 64,
        source_trajectory_digest="sha256:" + f"{index + 4:x}" * 64,
    )


def _memory() -> ContextualStrikeExpertMemory:
    return ContextualStrikeExpertMemory.build(
        experts=(
            _expert(0, 0.60, 0.17),
            _expert(1, 0.70, 0.15),
            _expert(2, 0.80, 0.18),
        ),
        negative_boundaries=(
            ContextualStrikeNegative(
                boundary_id="unresolved-high-corner",
                context_center=tuple(float(value) for value in _context(0.90).vector()),
                source_report_hash="sha256:" + "8" * 64,
                source_trajectory_digest="sha256:" + "9" * 64,
            ),
        ),
    )


def test_contextual_memory_routes_only_to_verified_local_expert() -> None:
    memory = _memory()

    selection = memory.select(context=_context(0.70), observation=_observation())

    assert selection.reason == "verified-expert"
    assert selection.expert_index == 1
    assert selection.normalized_distance == pytest.approx(0.0)
    assert selection.action is not None
    assert selection.action.stance_blend == pytest.approx(0.15)
    assert memory.activation_ceiling == "SIM_ONLY"
    assert memory.hardware_authorized is False


def test_contextual_memory_abstains_between_support_and_failure_regions() -> None:
    memory = _memory()

    selection = memory.select(context=_context(0.85), observation=_observation())

    assert selection.reason == "out-of-support"
    assert selection.action is None
    assert selection.expert_index == -1


def test_contextual_negative_memory_blocks_nearest_extrapolation() -> None:
    memory = _memory()

    selection = memory.select(context=_context(0.90), observation=_observation())

    assert selection.reason == "negative-memory"
    assert selection.action is None
    assert selection.normalized_distance == pytest.approx(0.0)


def test_contextual_memory_roundtrip_is_content_bound() -> None:
    memory = _memory()
    restored = ContextualStrikeExpertMemory.from_mapping(memory.to_dict())
    tampered = memory.to_dict()
    tampered["experts"][0]["actor"]["biases"] = [0.5, 0.0]

    assert restored == memory
    assert restored.memory_hash == memory.memory_hash
    with pytest.raises(ValueError, match="actor hash"):
        ContextualStrikeExpertMemory.from_mapping(tampered)


def test_task_context_uses_authoritative_opponent_shot_line_geometry() -> None:
    context = build_strike_task_context(
        goal_target_m=(10.0, 1.0),
        ball_position_m=np.asarray((0.0, 0.0), dtype=np.float64),
        opponent_positions_m=np.asarray(((5.0, 0.5), (2.0, 5.0)), dtype=np.float64),
    )

    assert context.goal_target_y_m == 1.0
    assert context.nearest_opponent_distance_m == pytest.approx(math.hypot(5.0, 0.5))
    assert context.shot_line_clearance_m == pytest.approx(0.0, abs=1.0e-12)
    assert context.shot_line_progress == pytest.approx(0.5)


def test_route_summary_separates_consultation_selection_and_abstention() -> None:
    trajectory = {
        "strike_context_memory_consulted": np.asarray((False, True, False, False)),
        "strike_coordination_actor_active": np.asarray((False, True, True, False)),
        "strike_context_abstained": np.asarray((False, False, False, True)),
        "strike_context_expert_index": np.asarray((-1, 2, 2, -1)),
        "strike_context_normalized_distance": np.asarray((0.0, 0.4, 0.4, 1.2)),
    }

    assert _route_summary(trajectory) == {
        "consulted_frames": 1,
        "selected_frames": 2,
        "abstained_frames": 1,
        "selected_expert_indices": [2],
        "maximum_selected_distance": 0.4,
    }


def test_exam_baseline_uses_same_candidate_contract_as_contextual_probe() -> None:
    report = {
        "goal_spec": {"target_y_m": 0.6},
        "world_safe": True,
        "non_replay_physics_gates_passed": False,
        "assessment": {
            "phase_sequence": [1, 2, 3, 4, 5, 6],
            "gates": {
                "physical_teammate_pass_received": True,
                "physical_foot_strike_in_strike_phase": True,
                "stable_recovery_completed": True,
                "strict_replay": False,
            },
            "metrics": {
                "receive_to_strike_sec": 4.3,
                "peak_shot_speed_mps": 8.2,
            },
        },
        "shot_projection": {
            "whole_ball_inside_goal": True,
            "target_error_m": 1.9,
        },
        "trajectory_digest": "sha256:" + "a" * 64,
    }

    row = _baseline_row("gy060", report)

    assert row["success"] is True


def test_exam_exact_replay_includes_route_and_world_outcome() -> None:
    report = {
        "trajectory_digest": "sha256:" + "a" * 64,
        "assessment": {"passed": True},
        "route_summary": {"selected_frames": 10},
        "shot_projection": {"whole_ball_inside_goal": True},
        "world_result": {"safe": True},
        "candidate_success": True,
    }

    assert _exact_replay(report, dict(report)) is True
    changed = dict(report)
    changed["route_summary"] = {"selected_frames": 9}
    assert _exact_replay(report, changed) is False
