from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.playmaker_pass_actor import (
    G1PlaymakerPassActor,
    PlaymakerPassMemory,
)
from rosclaw_soccer.training.playmaker_pass_discovery import PlaymakerPassProbeAction

_HASH_A = "sha256:" + "a" * 64
_HASH_B = "sha256:" + "b" * 64
_HASH_C = "sha256:" + "c" * 64
_HASH_D = "sha256:" + "d" * 64
_HASH_E = "sha256:" + "e" * 64


def _memory(index: int, *, success: bool, action: PlaymakerPassProbeAction) -> PlaymakerPassMemory:
    return PlaymakerPassMemory(
        context_hash="sha256:" + f"{index + 10:064x}",
        trajectory_hash="sha256:" + f"{index + 100:064x}",
        features=(float(index) * 0.01, 1.30, 1.21, -0.16, 0.10),
        action=action,
        delivery_error_m=0.20 if success else 0.70,
        safe=True,
        ordered_contacts=True,
    )


def _actor() -> G1PlaymakerPassActor:
    action = PlaymakerPassProbeAction(body_yaw_correction_rad=0.04)
    other = PlaymakerPassProbeAction(body_yaw_correction_rad=-0.02)
    successes = tuple(_memory(index, success=True, action=action) for index in range(6))
    failures = tuple(_memory(index + 20, success=False, action=other) for index in range(8))
    return G1PlaymakerPassActor(
        _HASH_A,
        _HASH_B,
        _HASH_C,
        _HASH_D,
        _HASH_E,
        (0.0, 1.30, 1.21, -0.16, 0.10),
        (0.10, 0.01, 0.01, 0.01, 0.01),
        successes,
        failures,
    )


def test_playmaker_actor_recalls_local_success_and_rejects_ood() -> None:
    actor = _actor()
    local = actor.decide(actor.successful_memories[2].features)
    assert local.accepted
    assert local.action == actor.successful_memories[2].action
    assert local.route == "VERIFIED_ROLE_LOCAL_PASS"
    ood = actor.decide((1.0, 1.30, 1.21, -0.16, 0.10))
    assert not ood.accepted
    assert ood.action is None


def test_playmaker_actor_rejects_same_action_failure_dominance() -> None:
    actor = _actor()
    failure = replace(
        actor.failed_memories[0],
        action=actor.successful_memories[0].action,
        features=(0.001, 1.30, 1.21, -0.16, 0.10),
    )
    guarded = replace(actor, failed_memories=(failure, *actor.failed_memories[1:]))
    decision = guarded.decide((0.001, 1.30, 1.21, -0.16, 0.10))
    assert not decision.accepted
    assert decision.route == "SAME_ACTION_FAILURE_FALLBACK"


def test_playmaker_actor_rejects_unqualified_success_memory() -> None:
    actor = _actor()
    bad = replace(actor.successful_memories[0], delivery_error_m=0.46)
    with pytest.raises(ValueError, match="unqualified pass"):
        replace(actor, successful_memories=(bad, *actor.successful_memories[1:]))
