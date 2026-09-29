"""SIM_ONLY measured-far receiving whole-body synergy policy."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from rosclaw_soccer.rsi.receiving_lateral_piecewise_expert import ReceivingLateralPiecewiseExpert
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingFarBodySynergyExpert(ReceivingLateralPiecewiseExpert):
    far_coordination: tuple[float, ...] = (0.0,) * 8

    def __post_init__(self) -> None:
        super().__post_init__()
        if (
            type(self.far_coordination) is not tuple
            or len(self.far_coordination) != 8
            or any(
                type(value) is not float or not math.isfinite(value) or abs(value) > 1
                for value in self.far_coordination
            )
        ):
            raise ValueError("bounded eight-coefficient far-body synergy required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_far_body_synergy_expert.v1",
                "parent_contract_hash": self.contract_hash,
                "far_coordination": self.far_coordination,
                "activation_ceiling": "SIM_ONLY",
                "far_activation": "measured_initial_ball_pelvis_lateral",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        base = np.asarray(super().propose(observation), dtype=np.float64)
        if self._selected_side or self._far_feature is None or self._far_feature <= 0:
            return tuple(float(value) for value in base)
        snapshot = self.mailbox.snapshot
        first_time = snapshot.first_own_foot_time_sec
        touched = first_time is not None
        if not touched and abs(observation.qpos[36] - observation.qpos[0]) > 1.2:
            return tuple(float(value) for value in base)
        if first_time is not None and observation.time_sec - first_time > 0.20:
            return tuple(float(value) for value in base)
        weights = np.asarray(
            self.far_coordination[4:] if touched else self.far_coordination[:4],
            dtype=np.float64,
        )
        # Anatomical left swing with right-leg support, trunk and counter-arm.
        # The scalar far feature is measured once, never inferred from a course label.
        basis = np.zeros((4, 29), dtype=np.float64)
        basis[0, (3, 4)] = (1.0, -0.5)
        basis[1, (0, 4)] = (0.8, -0.8)
        basis[2, (6, 9, 10)] = (-0.6, 0.8, -0.4)
        basis[3, (12, 15, 22)] = (0.5, 0.5, -0.5)
        base += 0.12 * self._far_feature * (weights @ basis)
        np.clip(base, -0.35, 0.35, out=base)
        return tuple(float(value) for value in base)
