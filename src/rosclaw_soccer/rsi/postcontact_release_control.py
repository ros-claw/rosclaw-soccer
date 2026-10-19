"""Bounded current-body post-touch action authority probe, SIM research only.

No predictions, source event frames, future targets or learning. A fixed
declared action name is mapped to the *measured* contacting foot's current
Jacobian once, after a completed foot-contact observation. Its full H300
outcome must be measured independently; this is not a qualified teacher.
"""

from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates

ACTION_NAMES = (
    "baseline",
    "forward_plus",
    "forward_minus",
    "lateral_plus",
    "lateral_minus",
    "lift_plus",
    "lift_minus",
    "yaw_plus",
    "yaw_minus",
    "lateral-1_yaw-1",
    "lateral-1_yaw+1",
    "lateral+1_yaw-1",
    "lateral+1_yaw+1",
)


@dataclass(frozen=True)
class PostContactReleaseCommand:
    offset_nm: np.ndarray
    report: dict[str, Any]


class PostContactReleaseControl:
    def __init__(self, model: Any, *, world_hash: str, action_name: str) -> None:
        if type(action_name) is not str or action_name not in ACTION_NAMES:
            raise ValueError("one declared post-contact research action required")
        self.model = model
        self.sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self.action_name = action_name
        self.next_frame = 0
        self.trigger_frame: int | None = None
        self.contact_side: int | None = None
        self.action_available: bool | None = None
        self.held = np.zeros(29)
        self.cancelled = self.failed = False

    def plan(
        self, data: Any, *, frame: int, previous_contact_forces: np.ndarray, pelvis_z_m: float
    ) -> PostContactReleaseCommand:
        if self.failed:
            raise ValueError("failed release probe cannot resume or retry")
        self.failed = True
        if (
            type(frame) is not int
            or frame != self.next_frame
            or not 0 <= frame < 300
            or not isinstance(data, mujoco.MjData)
            or data.model is not self.model
            or not np.isfinite(data.qpos).all()
            or not np.isfinite(data.qvel).all()
            or not np.isclose(data.time, frame * 0.02, rtol=0, atol=1e-8)
            or type(previous_contact_forces) is not np.ndarray
            or previous_contact_forces.shape != (6,)
            or previous_contact_forces.dtype != np.float64
            or not np.isfinite(previous_contact_forces).all()
            or np.any(previous_contact_forces < 0)
            or type(pelvis_z_m) not in (float, int)
            or not np.isfinite(pelvis_z_m)
            or (frame == 0 and np.any(previous_contact_forces))
        ):
            raise ValueError("next current-body frame and completed typed contact forces required")
        offset = np.zeros(29)
        if pelvis_z_m < 0.65 or np.any(previous_contact_forces[2:] > 1):
            self.cancelled = True
        if self.cancelled:
            status = "CANCELLED_OBSERVED_NONFOOT_OR_LOW_PELVIS"
        elif frame < 30:
            status = "OUTSIDE_RELEASE_RESEARCH_CLOCK"
        elif self.trigger_frame is None:
            if not np.any(previous_contact_forces[:2] > 1):
                status = "WAITING_FOR_COMPLETED_FOOT_CONTACT"
            else:
                self.trigger_frame = frame
                self.contact_side = int(np.argmax(previous_contact_forces[:2]))
                candidates = dict(
                    impact_torque_candidates(self.sampler.sample(data), self.contact_side)
                )
                self.action_available = self.action_name in candidates
                if self.action_available:
                    self.held = candidates[self.action_name].copy()
                    offset = self.held.copy()
                    status = "TRIGGERED_MEASURED_POSTCONTACT60MS_ACTION"
                else:
                    status = "DECLARED_ACTION_ALIASED_AT_CURRENT_JACOBIAN"
        elif not self.action_available:
            status = "DECLARED_ACTION_ALIASED_AT_CURRENT_JACOBIAN"
        elif frame < self.trigger_frame + 3:
            offset = self.held.copy()
            status = "HELD_MEASURED_POSTCONTACT60MS_ACTION"
        else:
            status = "RELEASE_ACTION_FINISHED_NO_SECOND_ATTEMPT"
        report = dict(
            schema="soccer.rsi.postcontact_release_command.v1",
            frame=frame,
            status=status,
            action_name=self.action_name,
            trigger_frame=self.trigger_frame,
            measured_contact_side=self.contact_side,
            action_available=self.action_available,
            requested_offset_nm=offset.tolist(),
            action_duration_frames=3,
            forecast_substeps=0,
            source_future_frame_or_target_used=False,
            neural_policy_trained=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        self.next_frame += 1
        self.failed = False
        return PostContactReleaseCommand(offset, report)

    def review(self) -> dict[str, Any]:
        if self.failed or self.next_frame != 300:
            raise ValueError("complete release-probe episode required")
        return dict(
            action_name=self.action_name,
            trigger_frame=self.trigger_frame,
            measured_contact_side=self.contact_side,
            action_available=self.action_available,
            cancelled=self.cancelled,
            maximum_action_duration_frames=3,
            selection_from_completed_contact_only=True,
            future_source_trace_consumed=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
