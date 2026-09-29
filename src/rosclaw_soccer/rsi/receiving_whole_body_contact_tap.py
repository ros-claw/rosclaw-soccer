"""Read-only 8-G1 A1 body/control/contact alignment for future SIM_ONLY learning."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation


@dataclass
class ReceivingWholeBodyContactTap:
    agent_id: str
    schedule_hash: str
    mailbox: ReceiveContactMailbox
    action_substrate: str = field(init=False, default="A1_body29")
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    requires_contact_history: bool = field(init=False, default=True)
    requires_locomotion_memory: bool = field(init=False, default=False)
    requires_navigation_context: bool = field(init=False, default=False)
    requires_navigation_target: bool = field(init=False, default=False)
    contract_hash: str = field(init=False)
    _next_frame: int | None = field(init=False, default=None)
    _rows: list[tuple[float, ...]] = field(init=False, default_factory=list)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mailbox, ReceiveContactMailbox)
            or self.mailbox.agent_id != self.agent_id
            or not self.schedule_hash.startswith("sha256:")
            or len(self.schedule_hash) != 71
        ):
            raise ValueError("same-player scheduled contact/body tap required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_whole_body_contact_tap.v1",
                "agent_id": self.agent_id,
                "schedule_hash": self.schedule_hash,
                "mailbox_hash": self.mailbox.contract_hash,
                "action_substrate": "A1_body29",
                "proposal": "ZERO_ONLY",
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not isinstance(observation, ReceivingFeedbackObservation):
            raise ValueError("typed whole-body feedback observation required")
        observation.__post_init__()
        snapshot = self.mailbox.snapshot
        if (
            observation.agent_id != self.agent_id
            or observation.action_substrate != "A1_body29"
            or observation.contact_history is None
            or observation.previous_body_residual_rad is None
            or (self._next_frame is not None and observation.frame != self._next_frame)
            or abs(snapshot.physics_time_sec - observation.time_sec) > 0.002001
        ):
            raise ValueError("consecutive completed same-player whole-body contact required")
        self._next_frame = observation.frame + 1
        self._rows.append(
            (
                float(observation.frame),
                observation.time_sec,
                snapshot.physics_time_sec,
                -1.0
                if snapshot.first_own_foot_time_sec is None
                else snapshot.first_own_foot_time_sec,
                -1.0
                if snapshot.first_contact_relative_y_mps is None
                else snapshot.first_contact_relative_y_mps,
                float(snapshot.prefoot_nonfoot_count),
                float(observation.committed_receive),
                float(observation.residual_admitted),
                *observation.qpos,
                *observation.qvel,
                *observation.foundation_target.target_rad,
                *observation.previous_body_residual_rad,
            )
        )
        return (0.0,) * 29

    def arrays(self) -> dict[str, NDArray[np.float64]]:
        values = np.asarray(self._rows, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != 150 or not np.isfinite(values).all():
            raise ValueError("complete finite A1 contact/body tape required")
        return {
            "frame": values[:, 0].copy(),
            "time_sec": values[:, 1].copy(),
            "physics_time_sec": values[:, 2].copy(),
            "first_own_foot_time_sec": values[:, 3].copy(),
            "first_contact_relative_y_mps": values[:, 4].copy(),
            "prefoot_nonfoot_count": values[:, 5].copy(),
            "committed_receive": values[:, 6].copy(),
            "residual_admitted": values[:, 7].copy(),
            "qpos": values[:, 8:51].copy(),
            "qvel": values[:, 51:92].copy(),
            "foundation_target_rad": values[:, 92:121].copy(),
            "previous_body_residual_rad": values[:, 121:150].copy(),
        }
