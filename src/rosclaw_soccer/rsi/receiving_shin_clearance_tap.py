"""Zero-action read-only shin-clearance tape for A1 receiving research."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.receiving_whole_body_foot_tap import ReceivingWholeBodyFootTap
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingShinClearanceTap(ReceivingWholeBodyFootTap):
    requires_shin_clearance: bool = field(init=False, default=True)
    _shin_rows: list[tuple[float, ...]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        super().__post_init__()
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_shin_clearance_tap.v1",
                "parent_contract_hash": self.contract_hash,
                "requires_shin_clearance": True,
                "proposal": "ZERO_ONLY",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        shin = observation.shin_clearance
        if shin is None:
            raise ValueError("same-player measured shin differential required")
        shin.__post_init__()
        if shin.agent_id != self.agent_id or shin.frame != observation.frame:
            raise ValueError("shin differential must share player and frame")
        result = super().propose(observation)
        self._shin_rows.append(
            (*shin.clearance_m, *shin.gradient_m_per_rad[0], *shin.gradient_m_per_rad[1])
        )
        return result

    def arrays(self) -> dict[str, NDArray[np.float64]]:
        arrays = super().arrays()
        shin = np.asarray(self._shin_rows, dtype=np.float64)
        if shin.shape != (len(arrays["frame"]), 14) or not np.isfinite(shin).all():
            raise ValueError("complete aligned shin clearance tape required")
        return {
            **arrays,
            "shin_clearance_m": shin[:, :2].copy(),
            "shin_gradient_m_per_rad": shin[:, 2:].reshape(-1, 2, 6).copy(),
        }
