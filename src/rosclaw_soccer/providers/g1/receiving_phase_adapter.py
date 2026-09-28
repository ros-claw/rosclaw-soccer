"""Bounded SIM_ONLY phase-conditioned motor memory around a frozen receiving actor.

The eight amplitudes are learned from physical episodes.  The anatomical
synergies are a low-rank search space; this adapter is not a robot executor or
an approved skill.  It reads only local body and ball state plus measured
contact history and cannot touch the simulator, ball, or other agents.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.sim.contracts import hash_bytes

ACTION_COUNT = 8
MAX_TARGET_RESIDUAL_RAD = 0.10

_BASIS: NDArray[np.float64] = np.zeros((ACTION_COUNT, 29), dtype=np.float64)
# Anticipatory lateral catch: align the left foot while the right foot stays
# planted.  The next three rows respectively shape the catching leg, support
# leg, and torso; rows four through seven unload those chains after contact.
_BASIS[0, (1, 5, 13)] = (1.0, -0.55, -0.20)
_BASIS[1, (0, 3, 4)] = (0.80, -0.60, -0.60)
_BASIS[2, (6, 9, 10, 14)] = (0.65, -0.45, -0.70, 0.25)
_BASIS[3, (7, 11, 13)] = (0.65, -0.45, -0.20)
_BASIS[4, (1, 5, 13)] = (0.75, -0.60, -0.15)
_BASIS[5, (6, 10, 14)] = (1.0, -0.80, 0.30)
_BASIS[6, (0, 3, 4)] = (0.70, -0.55, -0.60)
_BASIS[7, (14, 6, 15, 22)] = (0.80, -0.45, 0.20, 0.20)
_BASIS.setflags(write=False)


@dataclass(frozen=True)
class ReceivingPhaseAdapter:
    """Immutable eight-dimensional local feedback adapter; simulation only."""

    parameters: NDArray[np.float64]
    activation_ceiling: str = "SIM_ONLY"

    def __post_init__(self) -> None:
        value = self.parameters
        if (
            not isinstance(value, np.ndarray)
            or value.shape != (ACTION_COUNT,)
            or value.dtype.kind not in "f"
            or not np.isfinite(value).all()
            or np.max(np.abs(value)) > 1.0
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("bounded finite eight-action SIM_ONLY receiving adapter required")
        immutable = np.frombuffer(np.asarray(value, dtype="<f8").tobytes(), dtype="<f8")
        object.__setattr__(self, "parameters", immutable)

    @property
    def artifact_hash(self) -> str:
        return str(hash_bytes(self.parameters.tobytes()))

    def motor_target(
        self,
        *,
        qpos: NDArray[np.float64],
        qvel: NDArray[np.float64],
        foundation_target: NDArray[np.float64],
        joint_ranges: NDArray[np.float64],
        has_foot_contact: bool,
        elapsed_sec: float,
    ) -> NDArray[np.float64]:
        if (
            qpos.shape != (43,)
            or qvel.shape != (41,)
            or foundation_target.shape != (29,)
            or joint_ranges.shape != (29, 2)
            or type(has_foot_contact) is not bool
            or not np.isfinite(elapsed_sec)
            or not all(
                np.isfinite(value).all() for value in (qpos, qvel, foundation_target, joint_ranges)
            )
        ):
            raise ValueError("finite local measured state and bounded target required")
        dx = float(qpos[36] - qpos[0])
        pre_gate = 0.0
        post_gate = 0.0
        if not has_foot_contact:
            # Smooth visual pre-contact gate, not a fixed global frame index.
            pre_gate = float(
                np.clip((0.90 - dx) / 0.25, 0.0, 1.0) * np.clip((dx - 0.12) / 0.10, 0.0, 1.0)
            )
        elif 0.0 <= elapsed_sec <= 0.50:
            post_gate = float(np.exp(-elapsed_sec / 0.25))
        weights = np.r_[pre_gate * self.parameters[:4], post_gate * self.parameters[4:]]
        residual = np.clip(
            0.08 * (weights @ _BASIS), -MAX_TARGET_RESIDUAL_RAD, MAX_TARGET_RESIDUAL_RAD
        )
        return np.clip(foundation_target + residual, joint_ranges[:, 0], joint_ranges[:, 1])
