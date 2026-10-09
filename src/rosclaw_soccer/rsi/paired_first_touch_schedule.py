"""Fixed forty-course/three-policy scheduling, never Fresh admission.

No private coordinates, labels, model loading, physics or ledger access belong
here. A schedule only binds identities and ordering. The caller must separately
verify complete ancestry, native qualification, consumed exams and one-time
allocation before any execution. Hashes alone authenticate none of those.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from rosclaw.growth.canonical_json_snapshot import CanonicalJSONSnapshot

_HASH = re.compile(r"sha256:[0-9a-f]{64}\Z")
_ROLES = ("parent", "candidate973_primary0", "candidate974_primary0")


@dataclass(frozen=True)
class PairedFirstTouchJob:
    """Public case index, not private seed or physical initial state."""

    ordinal: int
    case_index: int
    policy_role: str
    policy_hash: str


class PairedFirstTouchSchedule:
    """Owned immutable declaration of exactly120 predetermined comparisons."""

    def __init__(
        self,
        *,
        protocol_hash: str,
        pool_hash: str,
        train_split_hash: str,
        physics_hash: str,
        parent_hash: str,
        candidate973_hash: str,
        candidate974_hash: str,
    ) -> None:
        binding = dict(
            protocol_hash=protocol_hash,
            pool_hash=pool_hash,
            train_split_hash=train_split_hash,
            physics_hash=physics_hash,
            parent_hash=parent_hash,
            candidate973_hash=candidate973_hash,
            candidate974_hash=candidate974_hash,
        )
        if any(
            type(value) is not str or _HASH.fullmatch(value) is None for value in binding.values()
        ):
            raise ValueError("complete ordinary sha256 scheduling identities required")
        if len({parent_hash, candidate973_hash, candidate974_hash}) != 3:
            raise ValueError("parent and both fixed primary candidate identities must differ")
        policy_hashes = (parent_hash, candidate973_hash, candidate974_hash)
        value = dict(
            schema="soccer.rsi.paired_first_touch_schedule.v1",
            binding=binding,
            distinct_case_count=40,
            independent_seed_cluster_count=10,
            lanes_per_cluster=4,
            native_execution_count_if_admitted=120,
            order="CASE_MAJOR_PARENT_973_PRIMARY0_974_PRIMARY0",
            jobs=[
                dict(
                    ordinal=3 * case_index + role_index,
                    case_index=case_index,
                    policy_role=role,
                    policy_hash=policy_hashes[role_index],
                )
                for case_index in range(40)
                for role_index, role in enumerate(_ROLES)
            ],
            ceiling="SCHEDULE_ONLY_NOT_EXECUTION_ADMISSION",
            private_case_coordinates_read=False,
            private_labels_read=False,
            ancestry_verified_here=False,
            consumed_exam_verified_here=False,
            native_transport_qualified_here=False,
            pool_allocated_here=False,
            fresh_execution_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        self._snapshot = CanonicalJSONSnapshot(value)

    def contract(self) -> dict[str, Any]:
        """Return owned public schedule, without implying prerequisites passed."""
        value = self._snapshot.restore()
        return dict(value, schedule_hash=self._snapshot.content_hash)

    def jobs(self) -> tuple[PairedFirstTouchJob, ...]:
        """Always regenerate from owned bytes; caller mutation cannot reorder."""
        value = self._snapshot.restore()
        return tuple(PairedFirstTouchJob(**job) for job in value["jobs"])

    def job(self, ordinal: int) -> PairedFirstTouchJob:
        if type(ordinal) is not int or not 0 <= ordinal < 120:
            raise ValueError("ordinary fixed job index0..119 required")
        return self.jobs()[ordinal]
