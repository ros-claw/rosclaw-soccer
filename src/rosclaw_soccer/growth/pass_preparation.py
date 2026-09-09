"""One-shot skill preparation bound to a fresh team agreement, SIM_ONLY.

This is a new preparation commitment, never a renewal of an in-flight pass.
The motor requests it; the world supplies the current peer agreement and gates.
No motion, possession, reception or hardware authority is granted here.
"""

from __future__ import annotations

import math
import re
from dataclasses import asdict, dataclass
from typing import Protocol, runtime_checkable

from rosclaw_soccer.growth.pass_handoff import PassHandoff
from rosclaw_soccer.sim.contracts import hash_json


@dataclass(frozen=True)
class PassPreparationRequest:
    source: str
    request_id: str
    frame: int
    time_sec: float
    target_xy: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            not isinstance(self.source, str)
            or re.fullmatch(r"[a-z][a-z0-9_]*\.[a-z][a-z0-9_]*", self.source) is None
            or not isinstance(self.request_id, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", self.request_id) is None
            or type(self.frame) is not int
            or self.frame < 0
            or type(self.time_sec) not in (int, float)
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0
            or type(self.target_xy) is not tuple
            or len(self.target_xy) != 2
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000
                for x in self.target_xy
            )
        ):
            raise ValueError("bounded immutable pass preparation request required")


@runtime_checkable
class PassPreparationProvider(Protocol):
    def preparation_request(
        self, *, frame: int, time_sec: float
    ) -> PassPreparationRequest | None: ...


class PassPreparationLedger:
    """Per-episode one-shot book; rejected bindings do not consume request IDs."""

    def __init__(self, scope_hash: str) -> None:
        if (
            not isinstance(scope_hash, str)
            or re.fullmatch(r"sha256:[0-9a-f]{64}", scope_hash) is None
        ):
            raise ValueError("explicit simulation scope hash required")
        self.scope_hash = scope_hash
        self._used: set[str] = set()
        self._prepared: set[tuple[str, str, float]] = set()

    def bind(
        self,
        previous: PassHandoff,
        request: PassPreparationRequest,
        *,
        source: str,
        receiver: str,
        accepted_target_xy: tuple[float, float],
        frame: int,
        time_sec: float,
        agreement_hash: str,
        motor_target_hash: str,
        source_ready: bool,
        receiver_ready: bool,
        motor_target_valid: bool,
    ) -> tuple[PassHandoff, dict[str, object]]:
        if (
            not isinstance(previous, PassHandoff)
            or not isinstance(request, PassPreparationRequest)
            or type(frame) is not int
            or type(time_sec) not in (int, float)
            or not math.isfinite(time_sec)
            or any(
                not isinstance(value, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
                for value in (agreement_hash, motor_target_hash)
            )
            or type(accepted_target_xy) is not tuple
            or len(accepted_target_xy) != 2
            or any(
                type(x) not in (int, float) or not math.isfinite(x) or abs(x) > 1000
                for x in accepted_target_xy
            )
            or any(
                type(v) is not bool or not v
                for v in (source_ready, receiver_ready, motor_target_valid)
            )
        ):
            raise ValueError("current world agreement and valid motor target required")
        if (
            source != request.source
            or source != previous.source
            or receiver != previous.receiver
            or request.frame != frame
            or abs(request.time_sec - time_sec) > 1e-9
            or request.target_xy != accepted_target_xy
            or request.request_id in self._used
            or len(self._used) >= 256
            or (previous.source, previous.receiver, previous.created_sec) in self._prepared
            or previous.source_foot_contact_sec is not None
            or previous.expired(time_sec)
        ):
            raise ValueError("stale, repeated, mismatched or already launched preparation")
        following = PassHandoff(
            source,
            receiver,
            time_sec,
            lifetime_sec=previous.lifetime_sec,
            flight_window_sec=previous.flight_window_sec,
            launch_target_xy=accepted_target_xy,
        )
        receipt: dict[str, object] = {
            "schema": "soccer.pass_preparation_binding.v1",
            "scope_hash": self.scope_hash,
            "agreement_hash": agreement_hash,
            "motor_target_hash": motor_target_hash,
            "request": asdict(request),
            "previous_handoff_hash": hash_json(asdict(previous)),
            "new_handoff_hash": hash_json(asdict(following)),
            "source": source,
            "receiver": receiver,
            "created_sec": time_sec,
            "preparation_deadline_sec": time_sec + following.lifetime_sec,
            "flight_window_sec": following.flight_window_sec,
            "activation_ceiling": "SIM_ONLY",
            "hardware_authorized": False,
        }
        receipt["receipt_hash"] = hash_json(receipt)
        self._used.add(request.request_id)
        self._prepared.add((previous.source, previous.receiver, previous.created_sec))
        self._prepared.add((source, receiver, time_sec))
        return following, receipt
