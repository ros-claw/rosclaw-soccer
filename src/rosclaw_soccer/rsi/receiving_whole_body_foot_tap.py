"""Read-only A1 contact/body/foot tape with no motor proposal authority."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.receiving_whole_body_contact_tap import ReceivingWholeBodyContactTap
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingWholeBodyFootTap(ReceivingWholeBodyContactTap):
    requires_foot_kinematics: bool = field(init=False, default=True)
    _foot_rows: list[tuple[float, ...]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_whole_body_foot_tap.v1",
                "parent_contract_hash": self.contract_hash,
                "requires_foot_kinematics": True,
                "proposal": "ZERO_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if observation.foot_kinematics is None:
            raise ValueError("current same-player measured foot geometry required")
        result = super().propose(observation)
        feet = observation.foot_kinematics
        assert feet.foot_linear_velocity_world_mps is not None
        self._foot_rows.append(
            (
                *np.asarray(feet.foot_position_world_m).ravel(),
                *np.asarray(feet.foot_linear_velocity_world_mps).ravel(),
                *np.asarray(feet.foot_linear_jacobian_world).ravel(),
            )
        )
        return result

    def arrays(self) -> dict[str, NDArray[np.float64]]:
        arrays = super().arrays()
        feet = np.asarray(self._foot_rows, dtype=np.float64)
        if feet.shape != (len(arrays["frame"]), 48) or not np.isfinite(feet).all():
            raise ValueError("complete aligned foot/body/contact tape required")
        return {
            **arrays,
            "foot_position_world_m": feet[:, :6].reshape(-1, 2, 3).copy(),
            "foot_velocity_world_mps": feet[:, 6:12].reshape(-1, 2, 3).copy(),
            "foot_jacobian_world": feet[:, 12:48].reshape(-1, 2, 3, 6).copy(),
        }
