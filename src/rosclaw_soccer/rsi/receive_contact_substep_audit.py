"""Bounded, read-only substep evidence for SIM_ONLY receive experiments."""

from __future__ import annotations

from dataclasses import asdict

from rosclaw_soccer.rsi.team_receive_contact_evidence import ReceiveContactEvidence
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorPhysicsObservation


class ReceiveContactSubstepAudit:
    """Wrap an existing evidence consumer; no simulator or control handle is retained.

    The saved per-contact force is the normal component, not total force; its
    scalar time integral is not a vector impulse. Every actual contact is kept
    up to a hard ceiling, where the consumer fails closed instead of truncating.
    """

    needs_contact_velocity = True
    activation_ceiling = "SIM_ONLY"

    def __init__(self, delegate: ReceiveContactEvidence, *, max_samples: int = 10000) -> None:
        if not isinstance(delegate, ReceiveContactEvidence) or type(max_samples) is not int:
            raise ValueError("typed contact observer and finite capacity required")
        if not 1 <= max_samples <= 100000:
            raise ValueError("bounded evidence capacity required")
        self.delegate = delegate
        self.agent_id = delegate.agent_id
        self.max_samples = max_samples
        self.samples: list[dict] = []
        self.contract_hash = hash_json(
            {
                "schema": "rosclaw_soccer.rsi.receive_contact_substep_audit.v1",
                "delegate_hash": delegate.contract_hash,
                "agent_id": self.agent_id,
                "max_samples": max_samples,
                "motor_authority": False,
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        self.delegate.observe_physics(observation)
        if not observation.ball_contacts:
            return
        if len(self.samples) >= self.max_samples:
            raise ValueError("substep evidence capacity exhausted; never silently truncate")
        if any(
            c.counterpart_minus_ball_velocity_world_mps is None for c in observation.ball_contacts
        ):
            raise ValueError("contact-point velocity evidence required")
        self.samples.append(
            {
                "time_sec": observation.time_sec,
                "world_bodies_safe": observation.world_bodies_safe,
                "ball_position_m": observation.qpos[-7:-4],
                "ball_velocity_mps": observation.qvel[-6:-3],
                "robot_qpos": observation.qpos[:-7],
                "robot_qvel": observation.qvel[:-6],
                "contacts": [asdict(c) for c in observation.ball_contacts],
            }
        )
