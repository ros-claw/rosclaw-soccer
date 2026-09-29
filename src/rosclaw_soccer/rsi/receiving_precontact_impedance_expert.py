"""SIM_ONLY side-conditioned pre-contact velocity/impedance on one A2 authority."""

from __future__ import annotations

import math
from dataclasses import dataclass

from rosclaw_soccer.rsi.receiving_precontact_expert import ReceivingPrecontactExpert
from rosclaw_soccer.rsi.receiving_side_conditioned_expert import ReceivingSideConditionedExpert
from rosclaw_soccer.sim.contracts import hash_json


@dataclass
class ReceivingPrecontactImpedanceExpert(ReceivingSideConditionedExpert):
    left_pre_gain: float = 0.35
    right_pre_gain: float = 0.35

    def __post_init__(self) -> None:
        super().__post_init__()
        if any(
            type(value) is not float or not math.isfinite(value) or not 0 <= value <= 1
            for value in (self.left_pre_gain, self.right_pre_gain)
        ):
            raise ValueError("finite bounded pre-contact impedance gains required")
        old_contract_hash = self.contract_hash
        children = []
        for pre_gain, post_gain, horizon, depth, lateral in (
            (
                self.left_pre_gain,
                self.left_post_gain,
                self.left_velocity_horizon_sec,
                self.left_target_depth_m,
                self.left_target_lateral_m,
            ),
            (
                self.right_pre_gain,
                self.right_post_gain,
                self.right_velocity_horizon_sec,
                self.right_target_depth_m,
                self.right_target_lateral_m,
            ),
        ):
            children.append(
                ReceivingPrecontactExpert(
                    self.agent_id,
                    self.schedule_hash,
                    self.mailbox,
                    pre_gain,
                    post_gain,
                    horizon,
                    target_depth_m=depth,
                    target_lateral_m=lateral,
                    coordination=self.coordination,
                    left_weights=self.left_weights,
                    right_weights=self.right_weights,
                )
            )
        self._children = (children[0], children[1])
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_precontact_impedance_expert.v1",
                "parent_contract_hash": old_contract_hash,
                "children": [child.contract_hash for child in self._children],
                "left_pre_gain": self.left_pre_gain,
                "right_pre_gain": self.right_pre_gain,
                "activation_ceiling": "SIM_ONLY",
            }
        )
