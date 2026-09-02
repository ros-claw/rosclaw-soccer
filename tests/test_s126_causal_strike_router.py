from __future__ import annotations

from pathlib import Path

import pytest

from rosclaw_soccer.growth.causal_strike_router import (
    CausalStrikeRouteAction,
    CausalStrikeRouteMemory,
    G1FailureAwareCausalStrikeRouter,
    load_causal_strike_router,
    save_causal_strike_router,
)
from rosclaw_soccer.training.causal_strike_option_exam import (
    default_causal_strike_option_holdouts,
)
from rosclaw_soccer.training.causal_strike_route_exam import (
    default_causal_strike_route_holdouts,
)
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionGrowthConfig,
    _context_kwargs,
)

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64


def _memory(index: int, lane: float, action: CausalStrikeRouteAction) -> CausalStrikeRouteMemory:
    digit = format(index, "x")
    return CausalStrikeRouteMemory(
        context_hash="sha256:" + digit * 64,
        trajectory_hash="sha256:" + "f" * 63 + digit,
        features=(lane, 0.0, 0.0, 0.0, 0.0),
        action=action,
    )


def _actor() -> G1FailureAwareCausalStrikeRouter:
    still = CausalStrikeRouteAction(maximum_arrival_advance_frames=0)
    advance = CausalStrikeRouteAction(maximum_arrival_advance_frames=12)
    return G1FailureAwareCausalStrikeRouter(
        body_hash=_HASH_A,
        kick_prior_hash=_HASH_B,
        source_discovery_hash=_HASH_C,
        training_snapshot_hash=_HASH_D,
        feature_center=(0.0,) * 5,
        feature_scale=(1.0,) * 5,
        successful_memories=(
            _memory(1, 1.00, still),
            _memory(2, 1.10, still),
            _memory(3, 2.00, advance),
            _memory(4, 2.10, advance),
        ),
        failed_memories=(
            _memory(5, -1.00, still),
            _memory(6, -1.10, still),
            _memory(7, -2.00, advance),
            _memory(8, -2.10, advance),
        ),
    )


def test_causal_strike_router_accepts_supported_success_mode() -> None:
    decision = _actor().decide((1.02, 0.0, 0.0, 0.0, 0.0))

    assert decision.accepted
    assert decision.route == "VERIFIED_CAUSAL_STRIKE_MODE"
    assert decision.action is not None
    assert decision.action.maximum_arrival_advance_frames == 0
    assert decision.confidence > 0.9


def test_causal_strike_router_rejects_failure_dominated_context() -> None:
    decision = _actor().decide((-1.0, 0.0, 0.0, 0.0, 0.0))

    assert not decision.accepted
    assert decision.route == "FAILURE_MEMORY_FALLBACK"
    assert decision.action is None


def test_causal_strike_router_round_trip_and_tamper_detection(tmp_path: Path) -> None:
    path = tmp_path / "actor.json"
    actor = _actor()
    save_causal_strike_router(actor, path)

    assert load_causal_strike_router(path) == actor
    payload = path.read_text(encoding="utf-8").replace(
        '"maximum_support_distance": 2.25',
        '"maximum_support_distance": 2.20',
    )
    path.write_text(payload, encoding="utf-8")
    with pytest.raises(ValueError, match="hash does not match"):
        load_causal_strike_router(path)


def test_causal_strike_route_action_rejects_extra_authority() -> None:
    with pytest.raises(ValueError, match="safe option envelope"):
        CausalStrikeRouteAction(
            maximum_arrival_advance_frames=12,
            shooter_precontact_joint_guard=False,
        )


def test_s126_holdouts_are_fresh_and_unique() -> None:
    development = default_causal_strike_option_holdouts()
    holdouts = default_causal_strike_route_holdouts()

    assert len(holdouts) == 6
    assert len({item.context_hash for item in holdouts}) == 6
    assert {item.context_hash for item in development}.isdisjoint(
        item.context_hash for item in holdouts
    )
    assert all(item.case_id.startswith("s126.holdout.v2.") for item in holdouts)


def test_context_kwargs_canonicalize_passer_contact_frame() -> None:
    class _LeadPolicy:
        @staticmethod
        def passer_world_yaw(*, target_lateral_m: float) -> float:
            return target_lateral_m

    context = default_causal_strike_route_holdouts()[2]
    kwargs = _context_kwargs(
        lead_policy=_LeadPolicy(),  # type: ignore[arg-type]
        config=CausalTransitionGrowthConfig(),
        context=context,
        receiver_start_sec=1.96,
    )

    parameters = kwargs["passer_parameter_overrides"]
    assert parameters["stance_offset_x"] == pytest.approx(context.passer_ball_local_xy_m[0] - 1.205)
    assert parameters["stance_offset_y"] == pytest.approx(context.passer_ball_local_xy_m[1] + 0.160)
    assert kwargs["passer_precontact_joint_guard_enabled"] is True
