from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.growth.causal_strike_router import CausalStrikeRouteAction
from rosclaw_soccer.growth.runtime_causal_strike_router import (
    G1RuntimeCausalStrikeRouter,
    RuntimeCausalStrikeMemory,
    load_runtime_causal_strike_router,
    runtime_causal_strike_features,
    save_runtime_causal_strike_router,
)
from rosclaw_soccer.training.causal_strike_option_exam import (
    default_causal_strike_option_holdouts,
)
from rosclaw_soccer.training.causal_strike_route_exam import (
    default_causal_strike_route_holdouts,
)
from rosclaw_soccer.training.runtime_causal_strike_exam import (
    default_runtime_causal_strike_holdouts,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64


def _memory(
    index: int, velocity_x: float, action: CausalStrikeRouteAction
) -> RuntimeCausalStrikeMemory:
    digit = format(index, "x")
    return RuntimeCausalStrikeMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "f" * 63 + digit,
        features=(3.65, 0.0, velocity_x, 0.0, 1.0, 0.0, 0.1),
        action=action,
    )


def _actor() -> G1RuntimeCausalStrikeRouter:
    still = CausalStrikeRouteAction(maximum_arrival_advance_frames=0)
    advance = CausalStrikeRouteAction(maximum_arrival_advance_frames=12)
    return G1RuntimeCausalStrikeRouter(
        body_hash=_HASH_A,
        kick_prior_hash=_HASH_B,
        source_discovery_hashes=(_HASH_C,),
        training_snapshot_hash=_HASH_D,
        feature_center=(0.0,) * 7,
        feature_scale=(1.0,) * 7,
        successful_memories=(
            _memory(1, -1.50, still),
            _memory(2, -1.60, still),
            _memory(3, -2.20, advance),
            _memory(4, -2.30, advance),
        ),
        failed_memories=(
            _memory(5, -0.50, still),
            _memory(6, -0.60, still),
            _memory(7, -3.20, advance),
            _memory(8, -3.30, advance),
        ),
    )


def test_runtime_causal_strike_features_include_measured_body_and_ball_state() -> None:
    features = runtime_causal_strike_features(
        ball_local_position_m=(3.5, 0.2, 0.115),
        ball_local_velocity_mps=(-2.0, 0.1, 0.0),
        ball_arrival_eta_sec=1.125,
        pelvis_local_position_m=(0.1, -0.3, 0.78),
        joint_velocity_rad_s=(0.2,) * 29,
    )

    assert features == pytest.approx((3.5, 0.2, -2.0, 0.1, 1.125, -0.3, 0.2))


def test_runtime_router_selects_velocity_conditioned_mode_and_rejects_failure() -> None:
    actor = _actor()
    selected = actor.decide((3.65, 0.0, -2.25, 0.0, 1.0, 0.0, 0.1))
    rejected = actor.decide((3.65, 0.0, -3.25, 0.0, 1.0, 0.0, 0.1))

    assert selected.accepted
    assert selected.action is not None
    assert selected.action.maximum_arrival_advance_frames == 12
    assert not rejected.accepted
    assert rejected.route == "RUNTIME_FAILURE_MEMORY_FALLBACK"


def test_runtime_router_round_trip_is_content_bound(tmp_path: Path) -> None:
    path = tmp_path / "runtime-router.json"
    actor = _actor()
    save_runtime_causal_strike_router(actor, path)

    assert load_runtime_causal_strike_router(path) == actor
    path.write_text(
        path.read_text(encoding="utf-8").replace(
            '"maximum_support_distance": 2.5',
            '"maximum_support_distance": 2.4',
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="hash does not match"):
        load_runtime_causal_strike_router(path)


def test_s127_runtime_holdouts_are_fresh_unique_and_mixed_support() -> None:
    prior = {
        item.context_hash
        for item in (
            *default_causal_strike_option_holdouts(),
            *default_causal_strike_route_holdouts(),
        )
    }
    holdouts = default_runtime_causal_strike_holdouts()

    assert len(holdouts) == 6
    assert len({item.context_hash for item in holdouts}) == 6
    assert prior.isdisjoint(item.context_hash for item in holdouts)
    assert all(item.case_id.startswith("s127.holdout.v3.") for item in holdouts)
