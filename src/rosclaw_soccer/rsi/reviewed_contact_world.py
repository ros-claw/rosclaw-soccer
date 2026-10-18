"""Pinned numerical world review for subsequent SIM-only neural experiments.

This authenticates an operator-selected review artifact, not its external
authority. The campaign also pins its producer source and physical witnesses.
Task failures in a numerically qualified world remain failures, not promotion.
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_COUNTS = {
    "soccer.rsi.mesh_contact_terminal_readback.v1": 3,
    "soccer.rsi.mesh_original_public_terminal_readback.v1": 4,
}


@dataclass(frozen=True)
class ReviewedContactWorld:
    path: Path
    expected_file_hash: str

    def verify(self, world_hash: str) -> dict[str, Any]:
        if (
            not isinstance(self.path, Path)
            or not self.path.is_absolute()
            or self.path.is_symlink()
            or any(p.is_symlink() for p in self.path.parents)
            or not self.path.is_file()
            or self.path.stat().st_size > 512 * 1024
            or type(self.expected_file_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.expected_file_hash) is None
            or type(world_hash) is not str
            or re.fullmatch(r"sha256:[0-9a-f]{64}", world_hash) is None
        ):
            raise ValueError("bounded pinned ordinary contact-world review file required")
        encoded = self.path.read_bytes()
        if hash_bytes(encoded) != self.expected_file_hash:
            raise ValueError("contact-world review bytes changed")
        value = json.loads(encoded)
        if (
            type(value) is not dict
            or value.get("report_hash")
            != hash_json({k: v for k, v in value.items() if k != "report_hash"})
            or type(value.get("schema")) is not str
            or value.get("schema") not in _COUNTS
            or any(
                value.get(k) is not True
                for k in (
                    "whole_input_and_witness_files_verified",
                    "shared_payload_logical_seals_rebuilt",
                    "prior_independent_replays_authenticated",
                )
            )
            or any(
                value.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")
            )
            or type(value.get("rows")) is not list
            or len(value["rows"]) != _COUNTS[value["schema"]]
            or any(
                type(row) is not dict or row.get("world_hash") != world_hash
                for row in value["rows"]
            )
        ):
            raise ValueError("whole sealed numerical review must bind the exact new contact world")
        return dict(
            schema="soccer.rsi.pinned_contact_world_review_binding.v1",
            path=str(self.path),
            file_hash=self.expected_file_hash,
            numerical_review_hash=value["report_hash"],
            world_hash=world_hash,
            numerical_review_rows=len(value["rows"]),
            physical_witnesses_reopened_here=False,
            campaign_must_pin_review_producer_and_original_witnesses=True,
            task_success_inherited=False,
            teacher_qualification_inherited=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )


def verify_contact_world_request(
    review: ReviewedContactWorld | None, *, world_hash: str | None, neural_requested: bool
) -> dict[str, Any] | None:
    if type(neural_requested) is not bool:
        raise ValueError("explicit neural request flag required")
    if world_hash is None:
        if review is not None:
            raise ValueError("contact-world review cannot silently accompany a legacy world")
        return None
    if review is None:
        if neural_requested:
            raise ValueError(
                "mesh contact world neural experiment requires pinned numerical review"
            )
        return None
    if type(review) is not ReviewedContactWorld:
        raise ValueError("typed pinned contact-world review required")
    return review.verify(world_hash)
