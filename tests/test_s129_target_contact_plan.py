from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.target_contact_plan_actor import (
    G1TargetContactPlanActor,
    TargetContactPlanAction,
    TargetContactPlanMemory,
)
from rosclaw_soccer.training.dual_clock_contact_exam import (
    default_dual_clock_contact_holdouts,
)
from rosclaw_soccer.training.target_contact_retention_exam import (
    default_target_contact_holdouts,
)


def _sha(character: str) -> str:
    return "sha256:" + character * 64


def _action(x: float, target: tuple[float, float, float]) -> TargetContactPlanAction:
    return TargetContactPlanAction(0, x, -0.06, 248, 0.04, 0.01, target)


def _memory(index: int, action: TargetContactPlanAction, score: float) -> TargetContactPlanMemory:
    return TargetContactPlanMemory(
        context_hash=_sha(str(index % 10)),
        trajectory_hash=_sha(chr(ord("a") + index)),
        features=(-0.11, 1.34, 1.19, -0.15, 0.09),
        action=action,
        quality_score=score,
    )


def _actor() -> G1TargetContactPlanActor:
    fast = _action(0.04, (7.0, 1.0, 3.0))
    slow = _action(0.12, (5.0, 0.0, 0.0))
    return G1TargetContactPlanActor(
        body_hash=_sha("1"),
        kick_prior_hash=_sha("2"),
        target_contact_actor_hash=_sha("3"),
        source_replay_hash=_sha("4"),
        training_snapshot_hash=_sha("5"),
        feature_center=(-0.11, 1.34, 1.19, -0.15, 0.09),
        feature_scale=(0.03, 0.05, 0.03, 0.03, 0.02),
        successful_memories=(
            _memory(0, slow, 6.5),
            _memory(1, fast, 7.8),
            _memory(2, slow, 6.6),
            _memory(3, fast, 7.7),
        ),
        failed_memories=tuple(_memory(index + 4, slow, 0.0) for index in range(4)),
    )


def test_plan_uses_physics_quality_only_to_break_equal_context_ties() -> None:
    actor = _actor()
    decision = actor.decide((-0.11, 1.34, 1.19, -0.15, 0.09))

    assert decision.accepted
    assert decision.action is not None
    assert decision.action.target_foot_velocity_xyz_mps == (7.0, 1.0, 3.0)
    assert decision.action.stance_offset_x_m == 0.04
    with pytest.raises(ValueError, match="SIM-only contract"):
        replace(actor, hardware_authorized=True)


def test_s129_holdouts_are_fresh_and_inside_registered_curriculum() -> None:
    old = {context.context_hash for context in default_dual_clock_contact_holdouts()}
    fresh = default_target_contact_holdouts()

    assert len(fresh) == 6
    assert len({context.context_hash for context in fresh}) == 6
    assert not old & {context.context_hash for context in fresh}
    assert all(context.case_id.startswith("s129.holdout.v5.") for context in fresh)
