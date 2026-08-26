"""Fail-closed 23-DoF adapters for HumanUP and HoST G1 baselines.

Both upstream policies use a 23-DoF G1, but their joints and action semantics
are not the same.  This module makes the distinction explicit and expands an
accepted command into Soccer's canonical 29-DoF order while holding omitted
joints at their current position.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json


class BaselineActionSemantics(StrEnum):
    POSITION_OFFSET_FROM_DEFAULT = "POSITION_OFFSET_FROM_DEFAULT"
    POSITION_INCREMENT_FROM_CURRENT = "POSITION_INCREMENT_FROM_CURRENT"


@dataclass(frozen=True)
class RecoveryBaselineActionContract:
    baseline_name: str
    source_commit: str
    source_joint_names: tuple[str, ...]
    semantics: BaselineActionSemantics
    action_scale_rad: float
    maximum_source_action: float
    maximum_target_delta_rad: float
    control_period_s: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_baseline_action_contract.v1"

    def __post_init__(self) -> None:
        if (
            not self.baseline_name
            or len(self.source_commit) != 40
            or not all(character in "0123456789abcdef" for character in self.source_commit)
            or len(self.source_joint_names) != 23
            or len(set(self.source_joint_names)) != 23
            or not set(self.source_joint_names).issubset(G1_DDS_JOINT_NAMES)
        ):
            raise ValueError("recovery baseline identity or joint order is invalid")
        values = (
            self.action_scale_rad,
            self.maximum_source_action,
            self.maximum_target_delta_rad,
            self.control_period_s,
        )
        if (
            any(not math.isfinite(value) or value <= 0.0 for value in values)
            or self.maximum_target_delta_rad > 1.0
            or not 0.005 <= self.control_period_s <= 0.1
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery baseline action bounds are invalid")

    @property
    def omitted_joint_names(self) -> tuple[str, ...]:
        source = set(self.source_joint_names)
        return tuple(name for name in G1_DDS_JOINT_NAMES if name not in source)

    @property
    def contract_hash(self) -> str:
        payload = asdict(self)
        payload["semantics"] = self.semantics.value
        return str(hash_json(payload))


@dataclass(frozen=True)
class RecoveryBaselineAdaptation:
    contract_hash: str
    accepted: bool
    target_joint_position_rad: NDArray[np.float64]
    maximum_requested_delta_rad: float
    reasons: tuple[str, ...]
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.recovery_baseline_adaptation.v1"

    def __post_init__(self) -> None:
        target = np.asarray(self.target_joint_position_rad, dtype=np.float64)
        if (
            not self.contract_hash.startswith("sha256:")
            or len(self.contract_hash) != 71
            or target.shape != (29,)
            or not np.all(np.isfinite(target))
            or not math.isfinite(self.maximum_requested_delta_rad)
            or self.maximum_requested_delta_rad < 0.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("recovery baseline adaptation is invalid")
        if self.accepted == bool(self.reasons):
            raise ValueError("accepted baseline adaptation cannot carry rejection reasons")
        object.__setattr__(self, "target_joint_position_rad", target)


_HUMANUP_JOINTS = G1_DDS_JOINT_NAMES[:19] + G1_DDS_JOINT_NAMES[22:26]
_HOST_JOINTS = (
    *G1_DDS_JOINT_NAMES[:13],
    *G1_DDS_JOINT_NAMES[15:20],
    *G1_DDS_JOINT_NAMES[22:27],
)

HUMANUP_G1_ACTION_CONTRACT = RecoveryBaselineActionContract(
    baseline_name="HumanUP-G1",
    source_commit="7516e0f27e6f4d1e7365cf64ea577a78247bd8cb",
    source_joint_names=_HUMANUP_JOINTS,
    semantics=BaselineActionSemantics.POSITION_OFFSET_FROM_DEFAULT,
    action_scale_rad=0.5,
    maximum_source_action=10.0,
    maximum_target_delta_rad=0.60,
    control_period_s=0.02,
)

HOST_G1_PRONE_ACTION_CONTRACT = RecoveryBaselineActionContract(
    baseline_name="HoST-G1-prone",
    source_commit="70bb580949a336a920833700e4b5dc3bf7fe87ce",
    source_joint_names=_HOST_JOINTS,
    semantics=BaselineActionSemantics.POSITION_INCREMENT_FROM_CURRENT,
    action_scale_rad=1.0,
    maximum_source_action=100.0,
    maximum_target_delta_rad=0.60,
    control_period_s=0.02,
)


def adapt_recovery_baseline_action(
    *,
    contract: RecoveryBaselineActionContract,
    source_action: NDArray[np.floating[Any]],
    current_joint_position_rad: NDArray[np.floating[Any]],
    default_joint_position_rad: NDArray[np.floating[Any]],
) -> RecoveryBaselineAdaptation:
    """Expand one source action, returning a no-op target on rejection."""

    action = np.asarray(source_action, dtype=np.float64)
    current = np.asarray(current_joint_position_rad, dtype=np.float64)
    default = np.asarray(default_joint_position_rad, dtype=np.float64)
    if (
        action.shape != (23,)
        or current.shape != (29,)
        or default.shape != (29,)
        or not np.all(np.isfinite(action))
        or not np.all(np.isfinite(current))
        or not np.all(np.isfinite(default))
    ):
        raise ValueError("recovery baseline adapter inputs have invalid shape or values")
    reasons = []
    maximum_source = float(np.max(np.abs(action)))
    if maximum_source > contract.maximum_source_action:
        reasons.append("SOURCE_ACTION_CLIP_EXCEEDED")
    indices = np.asarray(
        [G1_DDS_JOINT_NAMES.index(name) for name in contract.source_joint_names],
        dtype=np.int64,
    )
    requested = current.copy()
    if contract.semantics is BaselineActionSemantics.POSITION_OFFSET_FROM_DEFAULT:
        requested[indices] = default[indices] + contract.action_scale_rad * action
    else:
        requested[indices] = current[indices] + contract.action_scale_rad * action
    maximum_delta = float(np.max(np.abs(requested - current)))
    if maximum_delta > contract.maximum_target_delta_rad:
        reasons.append("TARGET_DELTA_EXCEEDED")
    accepted = not reasons
    return RecoveryBaselineAdaptation(
        contract_hash=contract.contract_hash,
        accepted=accepted,
        target_joint_position_rad=requested if accepted else current.copy(),
        maximum_requested_delta_rad=maximum_delta,
        reasons=tuple(reasons),
    )


__all__ = [
    "BaselineActionSemantics",
    "HOST_G1_PRONE_ACTION_CONTRACT",
    "HUMANUP_G1_ACTION_CONTRACT",
    "RecoveryBaselineActionContract",
    "RecoveryBaselineAdaptation",
    "adapt_recovery_baseline_action",
]
