"""Bounded, measured-state A1 residual for SIM_ONLY receiving experiments.

This actor only proposes joint-target offsets. The receiving feedback slot and
oracle cursor retain action admission, rate limiting, and motor ownership.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactMailbox
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.receiving_feedback import ReceivingFeedbackObservation

_SYNERGIES = np.zeros((6, 29), dtype=np.float64)
# Left-foot capture, contralateral support, torso counterbalance, and arm swing.
_SYNERGIES[0, [0, 3, 4]] = [0.6, -0.8, 1.0]
_SYNERGIES[1, [6, 9, 10]] = [-0.6, 0.8, -1.0]
_SYNERGIES[2, [1, 5, 7, 11]] = [0.7, -1.0, -0.7, 1.0]
_SYNERGIES[3, [12, 13, 14]] = [0.4, 0.7, 1.0]
_SYNERGIES[4, [15, 16, 22, 23]] = [0.8, 0.5, -0.8, -0.5]
_SYNERGIES[5, [18, 25]] = [-1.0, 1.0]


@dataclass
class ReceivingWholeBodyResidual:
    """Twelve learned coefficients: six pre-contact and six post-contact.

    The transition uses completed 500 Hz contact evidence; no future ball
    trajectory, privileged simulator handle, or scripted target is available.
    """

    agent_id: str
    schedule_hash: str
    mailbox: ReceiveContactMailbox
    coefficients: tuple[float, ...]
    action_substrate: str = field(init=False, default="A1_body29")
    activation_ceiling: str = field(init=False, default="SIM_ONLY")
    requires_contact_history: bool = field(init=False, default=True)
    contract_hash: str = field(init=False)
    _next_frame: int | None = field(init=False, default=None)
    _nonzero_frames: int = field(init=False, default=0)

    def __post_init__(self) -> None:
        if (
            not isinstance(self.mailbox, ReceiveContactMailbox)
            or self.mailbox.agent_id != self.agent_id
            or not self.schedule_hash.startswith("sha256:")
            or len(self.schedule_hash) != 71
            or type(self.coefficients) is not tuple
            or len(self.coefficients) != 12
            or any(
                type(x) not in (float, int) or not math.isfinite(x) or abs(x) > 1
                for x in self.coefficients
            )
        ):
            raise ValueError("bounded same-player whole-body residual required")
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receiving_whole_body_residual.v1",
                "agent_id": self.agent_id,
                "schedule_hash": self.schedule_hash,
                "mailbox_hash": self.mailbox.contract_hash,
                "coefficients": self.coefficients,
                "activation_ceiling": self.activation_ceiling,
            }
        )

    @property
    def nonzero_frames(self) -> int:
        return self._nonzero_frames

    def propose(self, observation: ReceivingFeedbackObservation) -> tuple[float, ...]:
        if not isinstance(observation, ReceivingFeedbackObservation):
            raise ValueError("typed receiving feedback required")
        observation.__post_init__()
        snapshot = self.mailbox.snapshot
        if (
            observation.agent_id != self.agent_id
            or observation.action_substrate != self.action_substrate
            or observation.contact_history is None
            or observation.previous_body_residual_rad is None
            or (self._next_frame is not None and observation.frame != self._next_frame)
            or abs(snapshot.physics_time_sec - observation.time_sec) > 0.002001
        ):
            raise ValueError("consecutive completed same-player contact/body observation required")
        self._next_frame = observation.frame + 1
        touched = snapshot.first_own_foot_time_sec is not None
        weights = np.asarray(self.coefficients[6:] if touched else self.coefficients[:6])
        # Smooth bounded feedback; the motor cursor subsequently enforces its
        # independent 0.1 rad amplitude and 0.02 rad/frame rate limits.
        desired = np.clip(weights @ _SYNERGIES, -1.0, 1.0) * 0.08
        if np.any(desired):
            self._nonzero_frames += 1
        return tuple(float(x) for x in desired)
