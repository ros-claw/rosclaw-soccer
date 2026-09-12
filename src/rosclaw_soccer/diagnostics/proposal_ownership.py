"""Read-only exclusive-proposal witnesses, independent of robot and simulator.

This validates caller-reported routing evidence, not execution or permission.
It cannot discover unreported proposals, authenticate hashes, stop an actuator,
or replace the caller's clock, lifecycle, physical guards and durable ledger.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass

_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.:-]{0,127}")
_HASH = re.compile(r"sha256:[0-9a-f]{64}")


def _identifier(value: object) -> bool:
    return isinstance(value, str) and _IDENTIFIER.fullmatch(value) is not None


def _hash(value: object) -> bool:
    return isinstance(value, str) and _HASH.fullmatch(value) is not None


@dataclass(frozen=True)
class ProposalActivity:
    owner_id: str
    proposal_count: int
    terminal: bool

    def __post_init__(self) -> None:
        if (
            not _identifier(self.owner_id)
            or type(self.proposal_count) is not int
            or self.proposal_count not in (0, 1)
            or type(self.terminal) is not bool
            or (self.terminal and self.proposal_count != 0)
        ):
            raise ValueError("one nonterminal proposal per declared owner required")


@dataclass(frozen=True)
class ProposalOwnershipSnapshot:
    """A bounded single-frame witness for an exclusive-routing profile.

    Context may be an observation or an offline report/frame commitment; callers
    must describe which. No clock continuity or inference-history proof is made.
    Zero proposals requires no returned target; one requires a target digest.
    Internal ensembles need their own qualified owner, not multiple routers
    silently overwriting each other. Fallback execution is a separate owner.
    """

    subject_id: str
    frame: int
    context_hash: str
    activities: tuple[ProposalActivity, ...]
    returned_target_hash: str | None

    def __post_init__(self) -> None:
        if (
            not _identifier(self.subject_id)
            or type(self.frame) is not int
            or not 0 <= self.frame < 2**31
            or not _hash(self.context_hash)
            or type(self.activities) is not tuple
            or not 1 <= len(self.activities) <= 16
            or any(type(activity) is not ProposalActivity for activity in self.activities)
            or (self.returned_target_hash is not None and not _hash(self.returned_target_hash))
        ):
            raise ValueError("bounded subject/frame/context and immutable activities required")
        for activity in self.activities:
            activity.__post_init__()
        if len({activity.owner_id for activity in self.activities}) != len(self.activities):
            raise ValueError("unique declared proposal owners required")
        count = sum(activity.proposal_count for activity in self.activities)
        if count > 1:
            raise ValueError("multiple proposal owners in an exclusive frame")
        if (self.returned_target_hash is not None) != (count == 1):
            raise ValueError("returned target does not match the declared proposal activity")

    @property
    def selected_owner_id(self) -> str | None:
        return next((a.owner_id for a in self.activities if a.proposal_count == 1), None)

    @property
    def binding_hash(self) -> str:
        payload = asdict(self)
        payload["activities"] = sorted(payload["activities"], key=lambda row: row["owner_id"])
        payload["schema"] = "rosclaw_soccer.exclusive_proposal_snapshot.v1"
        encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
        return "sha256:" + hashlib.sha256(encoded.encode()).hexdigest()
