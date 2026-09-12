"""Football teacher screening, not policy promotion or physical verification.

Keep every declared perturbation in a candidate's denominator. Callers must
independently verify physical evidence and bind the candidate/context hashes.
This module cannot authenticate caller-provided outcomes or authorize motion.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass


@dataclass(frozen=True)
class PassReplica:
    replica_id: int
    candidate_hash: str
    context_hash: str
    body_safe: bool
    ball_in_play: bool
    nonfoot_contact: bool
    late_contact: bool
    controlled_pass: bool
    plane_error_m: float | None

    def __post_init__(self) -> None:
        if type(self.replica_id) is not int or not 0 <= self.replica_id < 64:
            raise ValueError("bounded integer replica ID required")
        for digest in (self.candidate_hash, self.context_hash):
            if type(digest) is not str or re.fullmatch(r"sha256:[0-9a-f]{64}", digest) is None:
                raise ValueError("explicit SHA256 candidate and context bindings required")
        for flag in (
            self.body_safe,
            self.ball_in_play,
            self.nonfoot_contact,
            self.late_contact,
            self.controlled_pass,
        ):
            if type(flag) is not bool:
                raise ValueError("explicit boolean pass outcomes required")
        if self.plane_error_m is not None and (
            type(self.plane_error_m) not in (int, float)
            or not 0 <= self.plane_error_m <= 1000
            or not math.isfinite(self.plane_error_m)
        ):
            raise ValueError(
                "finite plane error in [0, 1000] m or explicit missing crossing required"
            )


@dataclass(frozen=True)
class PassReplicaScreen:
    attempted: int
    eligible: int
    precise: int
    screening_passed: bool
    failed_replica_ids: tuple[int, ...]


def screen_pass_replicas(
    replicas: tuple[PassReplica, ...], *, expected_replicas: int = 4, required_precise: int = 3
) -> PassReplicaScreen:
    """Require a complete bound group, all contact/body gates, and <=0.1 m precision.

    A passing search screen only identifies a teacher for further CPU testing;
    correlated replicas are not independent matches or statistical confidence.
    """
    if (
        type(expected_replicas) is not int
        or not 2 <= expected_replicas <= 64
        or type(required_precise) is not int
        or not 1 <= required_precise <= expected_replicas
        or type(replicas) is not tuple
        or len(replicas) != expected_replicas
        or any(type(row) is not PassReplica for row in replicas)
    ):
        raise ValueError("complete bounded replica group and explicit precision count required")
    if {row.replica_id for row in replicas} != set(range(expected_replicas)):
        raise ValueError("every declared replica ID must occur exactly once")
    if len({(row.candidate_hash, row.context_hash) for row in replicas}) != 1:
        raise ValueError("replicas must share the same candidate and entry context")
    eligible = 0
    precise = 0
    failed = []
    for row in replicas:
        allowed = (
            row.body_safe and row.ball_in_play and not row.nonfoot_contact and not row.late_contact
        )
        accurate = (
            allowed
            and row.controlled_pass
            and row.plane_error_m is not None
            and row.plane_error_m <= 0.1
        )
        eligible += int(allowed)
        precise += int(accurate)
        if not accurate:
            failed.append(row.replica_id)
    return PassReplicaScreen(
        expected_replicas,
        eligible,
        precise,
        eligible == expected_replicas and precise >= required_precise,
        tuple(sorted(failed)),
    )
