from __future__ import annotations

import pytest

from rosclaw_soccer.growth.planned_contact_mode_actor import (
    G1PlannedContactModeActor,
    PlannedContactModeMemory,
    load_planned_contact_mode_actor,
    planned_contact_mode_features,
    save_planned_contact_mode_actor,
)
from rosclaw_soccer.growth.runtime_contact_mode_actor import RuntimeContactModeAction
from rosclaw_soccer.training.dual_clock_contact_exam import (
    default_dual_clock_contact_holdouts,
)
from rosclaw_soccer.training.runtime_causal_strike_exam import (
    default_runtime_causal_strike_holdouts,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64


def _memory(index: int, lane: float, action: RuntimeContactModeAction) -> PlannedContactModeMemory:
    digit = format(index, "x")
    return PlannedContactModeMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "f" * 63 + digit,
        features=(lane, 1.30, 1.205, -0.16, 0.10),
        action=action,
    )


def _actor() -> G1PlannedContactModeActor:
    left = RuntimeContactModeAction(0, 0.12, -0.06)
    right = RuntimeContactModeAction(12, -0.12, 0.12)
    return G1PlannedContactModeActor(
        body_hash=_HASH_A,
        kick_prior_hash=_HASH_B,
        source_discovery_hashes=(_HASH_C,),
        training_snapshot_hash=_HASH_D,
        feature_center=(0.0,) * 5,
        feature_scale=(1.0,) * 5,
        successful_memories=(
            _memory(1, -0.10, left),
            _memory(2, -0.08, left),
            _memory(3, 0.08, right),
            _memory(4, 0.10, right),
        ),
        failed_memories=(
            _memory(5, -0.35, left),
            _memory(6, -0.30, left),
            _memory(7, 0.30, right),
            _memory(8, 0.35, right),
        ),
    )


def test_planned_actor_selects_precontact_mode_and_is_content_bound(tmp_path) -> None:
    actor = _actor()
    features = planned_contact_mode_features(
        receiver_lane_m=-0.09,
        reception_target_x_m=1.30,
        passer_ball_local_xy_m=(1.205, -0.16),
        ball_ground_friction=0.10,
    )
    decision = actor.decide(features)
    path = tmp_path / "planned.json"
    save_planned_contact_mode_actor(actor, path)

    assert decision.accepted
    assert decision.action == RuntimeContactModeAction(0, 0.12, -0.06)
    assert load_planned_contact_mode_actor(path) == actor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"maximum_support_distance": 1.75', '"maximum_support_distance": 1.5'
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash mismatch"):
        load_planned_contact_mode_actor(path)


def test_s128_holdouts_are_fresh_unique_and_cover_three_support_clusters() -> None:
    prior = {context.context_hash for context in default_runtime_causal_strike_holdouts()}
    holdouts = default_dual_clock_contact_holdouts()

    assert len(holdouts) == 6
    assert len({context.context_hash for context in holdouts}) == 6
    assert prior.isdisjoint(context.context_hash for context in holdouts)
    assert all(context.case_id.startswith("s128.holdout.v4.") for context in holdouts)
    assert all(context.receiver_lane_m < 0.0 for context in holdouts[:4])
    assert all(context.receiver_lane_m > 0.0 for context in holdouts[4:])
