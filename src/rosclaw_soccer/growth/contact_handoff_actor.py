"""Content-bound post-contact handoff muscle memory.

The actor owns no motor primitive.  It selects when an already qualified
standby/recovery controller may take ownership after measured ball contact.
Keeping that decision in a learned artifact makes failure-driven timing
reusable without granting the learner any new torque authority.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

_SCHEMA = "rosclaw.growth.g1_contact_handoff_actor.v1"


def _commitment(value: str, label: str) -> str:
    if not isinstance(value, str) or not value.startswith("sha256:") or len(value) != 71:
        raise ValueError(f"{label} must be a sha256 commitment")
    return value


@dataclass(frozen=True)
class ContactHandoffDecision:
    accepted: bool
    handoff_policy_frame: int | None
    route: str
    actor_hash: str


@dataclass(frozen=True)
class G1ContactHandoffActor:
    """Select a bounded post-contact recovery handoff relative to strike timing."""

    body_hash: str
    target_plan_actor_hash: str
    target_contact_actor_hash: str
    source_evidence_hashes: tuple[str, ...]
    selected_offset_frames: int
    evaluated_offset_frames: tuple[int, ...]
    safe_case_count: int
    recovered_failure_count: int
    training_case_count: int
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = _SCHEMA

    def __post_init__(self) -> None:
        for value, label in (
            (self.body_hash, "body_hash"),
            (self.target_plan_actor_hash, "target_plan_actor_hash"),
            (self.target_contact_actor_hash, "target_contact_actor_hash"),
            *[(item, "source_evidence_hash") for item in self.source_evidence_hashes],
        ):
            _commitment(value, label)
        if (
            self.schema_version != _SCHEMA
            or not self.source_evidence_hashes
            or not 0 <= self.selected_offset_frames <= 24
            or len(self.evaluated_offset_frames) < 2
            or len(set(self.evaluated_offset_frames)) != len(self.evaluated_offset_frames)
            or self.selected_offset_frames not in self.evaluated_offset_frames
            or any(not 0 <= value <= 32 for value in self.evaluated_offset_frames)
            or not 1 <= self.training_case_count <= 64
            or not 0 <= self.safe_case_count <= self.training_case_count
            or not 0 <= self.recovered_failure_count <= self.training_case_count
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("contact handoff actor violates its SIM-only contract")

    @property
    def actor_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value = asdict(self)
        value["source_evidence_hashes"] = list(self.source_evidence_hashes)
        value["evaluated_offset_frames"] = list(self.evaluated_offset_frames)
        value["authority"] = {
            "learned_output": "handoff timing only",
            "torque_authority": False,
            "contact_gate_required": True,
        }
        if include_hash:
            value["actor_hash"] = self.actor_hash
        return value

    def decide(self, *, contact_policy_frame: int) -> ContactHandoffDecision:
        if isinstance(contact_policy_frame, bool) or not 220 <= contact_policy_frame <= 290:
            return ContactHandoffDecision(False, None, "OUT_OF_SUPPORT", self.actor_hash)
        frame = contact_policy_frame + self.selected_offset_frames
        if frame > 300:
            return ContactHandoffDecision(False, None, "OUT_OF_SUPPORT", self.actor_hash)
        return ContactHandoffDecision(True, frame, "LEARNED_CONTACT_GATED_HANDOFF", self.actor_hash)


def save_contact_handoff_actor(actor: G1ContactHandoffActor, path: Path) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(actor.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, destination)


def load_contact_handoff_actor(path: Path) -> G1ContactHandoffActor:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("contact handoff actor must be an object")
    claimed = payload.pop("actor_hash", None)
    payload.pop("authority", None)
    payload["source_evidence_hashes"] = tuple(payload["source_evidence_hashes"])
    payload["evaluated_offset_frames"] = tuple(payload["evaluated_offset_frames"])
    actor = G1ContactHandoffActor(**payload)
    if claimed != actor.actor_hash:
        raise ValueError("contact handoff actor hash mismatch")
    return actor


__all__ = [
    "ContactHandoffDecision",
    "G1ContactHandoffActor",
    "load_contact_handoff_actor",
    "save_contact_handoff_actor",
]
