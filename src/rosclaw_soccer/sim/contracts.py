"""Immutable Soccer simulation contracts shared by worlds and G1 providers."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from typing import Any

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES

G1_HARD_TORQUE_LIMITS = (
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    139.0,
    88.0,
    139.0,
    50.0,
    50.0,
    88.0,
    50.0,
    50.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
    25.0,
    25.0,
    25.0,
    25.0,
    25.0,
    5.0,
    5.0,
)


@dataclass(frozen=True)
class ShotParameters:
    """Bounded, interpretable adapter around a fixed whole-body kick prior."""

    stance_offset_x: float = 0.0
    stance_offset_y: float = 0.0
    pelvis_yaw_offset: float = 0.0
    kick_foot: str = "right"
    com_shift_y: float = 0.0
    swing_amplitude: float = 1.0
    swing_speed_scale: float = 1.0
    foot_yaw_offset: float = 0.0
    foot_pitch_offset: float = 0.0
    loft_synergy: float = 0.0
    contact_phase_offset: float = 0.0
    kick_trigger_delay: float = 0.0
    recovery_step_length: float = 0.04
    recovery_step_yaw: float = 0.0
    policy_type: str = "fixed_prior"
    dataset_snapshot_hash: str | None = None

    def __post_init__(self) -> None:
        bounds = {
            "stance_offset_x": (self.stance_offset_x, -0.12, 0.12),
            "stance_offset_y": (self.stance_offset_y, -0.12, 0.12),
            "pelvis_yaw_offset": (self.pelvis_yaw_offset, -0.20, 0.20),
            "com_shift_y": (self.com_shift_y, -0.08, 0.08),
            "swing_amplitude": (self.swing_amplitude, 0.40, 1.15),
            "swing_speed_scale": (self.swing_speed_scale, 0.80, 1.50),
            "foot_yaw_offset": (self.foot_yaw_offset, -0.12, 0.12),
            "foot_pitch_offset": (self.foot_pitch_offset, -0.18, 0.18),
            "loft_synergy": (self.loft_synergy, 0.0, 0.30),
            "contact_phase_offset": (self.contact_phase_offset, -0.10, 0.10),
            "kick_trigger_delay": (self.kick_trigger_delay, 0.0, 0.20),
            "recovery_step_length": (self.recovery_step_length, 0.0, 0.15),
            "recovery_step_yaw": (self.recovery_step_yaw, -0.15, 0.15),
        }
        for name, (value, minimum, maximum) in bounds.items():
            if not math.isfinite(value) or not minimum <= value <= maximum:
                raise ValueError(f"{name} must be in [{minimum}, {maximum}]")
        if self.kick_foot not in {"left", "right"}:
            raise ValueError("kick_foot must be left or right")
        if self.policy_type not in {
            "fixed_prior",
            "parameter",
            "trajectory",
            "skill_graph",
            "learned_adapter",
        }:
            raise ValueError("unsupported Soccer policy_type")
        if self.policy_type == "learned_adapter" and (
            not self.dataset_snapshot_hash or not self.dataset_snapshot_hash.startswith("sha256:")
        ):
            raise ValueError("learned adapters require a dataset snapshot hash")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def policy_hash(self) -> str:
        return hash_json(self.to_dict())


def hash_json(value: Any) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode()
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def hash_bytes(payload: bytes) -> str:
    return "sha256:" + hashlib.sha256(payload).hexdigest()


def sanitize_nonfinite_evidence(
    value: Any,
    *,
    path: str = "",
) -> tuple[Any, list[str]]:
    """Replace non-finite diagnostics with null and retain their exact paths.

    Evidence hashes deliberately reject NaN and infinity.  Long-running
    simulation jobs must still be able to write a fail-closed report when a
    quarantined world leaves a non-finite diagnostic behind, so callers get a
    JSON-safe value plus an auditable list of every replacement.
    """

    if isinstance(value, float) and not math.isfinite(value):
        return None, [path or "$"]
    if isinstance(value, dict):
        sanitized: dict[str, Any] = {}
        paths: list[str] = []
        for key, item in value.items():
            child_path = f"{path}.{key}" if path else str(key)
            sanitized_item, child_paths = sanitize_nonfinite_evidence(
                item,
                path=child_path,
            )
            sanitized[str(key)] = sanitized_item
            paths.extend(child_paths)
        return sanitized, paths
    if isinstance(value, (list, tuple)):
        sanitized_sequence: list[Any] = []
        paths = []
        for index, item in enumerate(value):
            child_path = f"{path}[{index}]" if path else f"[{index}]"
            sanitized_item, child_paths = sanitize_nonfinite_evidence(
                item,
                path=child_path,
            )
            sanitized_sequence.append(sanitized_item)
            paths.extend(child_paths)
        return sanitized_sequence, paths
    return value, []


__all__ = [
    "G1_DDS_JOINT_NAMES",
    "G1_HARD_TORQUE_LIMITS",
    "ShotParameters",
    "hash_bytes",
    "hash_json",
    "sanitize_nonfinite_evidence",
]
