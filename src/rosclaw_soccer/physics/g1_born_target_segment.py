"""Born precontact SIM target segment from a genuinely completed own actor.

Copies the complete native integration state without mj_forward. No time-zero
constructor is invoked and its clock overwritten. This proves no new policy.
"""

from typing import Any

import mujoco
import numpy as np

from rosclaw_soccer.physics.g1_pd_torque_forecast import G1PDTorqueForecaster
from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
from rosclaw_soccer.rsi.causal_impact_replay import CausalImpactReplay
from rosclaw_soccer.rsi.feedback_branch_control import BODY_SHAPES
from rosclaw_soccer.rsi.full_task_branch import CAUSAL_FIELDS, integration_hash


class G1BornTargetSegment(G1TargetEpisode):
    """Distinct origin and new-step accounting, exact continuous native body."""

    def __init__(self, parent: CausalImpactReplay, *, world_hash: str) -> None:
        if (
            type(parent) is not CausalImpactReplay
            or set(vars(parent)) != CAUSAL_FIELDS
            or parent.failed is not False
            or type(parent.next_frame) is not int
            or not 31 <= parent.next_frame < 300
            or parent.attempted_steps != parent.next_frame * 10
            or parent.integrated_steps != parent.attempted_steps
            or parent.teacher is not None
            or parent.release_control is not None
            or parent.navigation_recovery is not None
            or parent.born_release_capture
            or parent.ik.pending is not None
            or parent.ik.contact is not None
            or parent.ik.next_frame != parent.next_frame
            or parent.decoder._active_frame is not None
            or parent.decoder._memory.last_frame != parent.next_frame - 1
            or parent.decoder._recurrent.next_index != parent.next_frame - 30
            or parent.tracker is None
            or parent.tracker._next_frame != parent.next_frame
            or parent.navigation._next_frame != parent.next_frame
            or set(parent.body) != set(BODY_SHAPES)
            or any(len(v) != parent.next_frame for v in parent.body.values())
            or not np.isclose(parent.data.time, parent.next_frame * 0.02, atol=1e-8, rtol=0)
            or tuple(int(w.number) for w in parent.data.warning) != parent.warnings
        ):
            raise ValueError("completed own plain precontact causal actor boundary required")
        checked = G1PDTorqueForecaster(
            parent.model,
            expected_world_hash=world_hash,
            kp=parent.kp,
            kd=parent.kd,
            max_forecast_steps=(300 - parent.next_frame) * 10,
        )
        donor_hash = integration_hash(parent.model, parent.data)
        self.model = parent.model
        self.data = mujoco.MjData(self.model)
        mujoco.mj_copyData(self.data, self.model, parent.data)
        if integration_hash(self.model, self.data) != donor_hash:
            raise ValueError("complete owned native precontact handoff state differs")
        self.qi, self.vi, self.ai = checked._qi, checked._vi, checked._ai
        self.ball, self.bg, self.pelvis = checked._ball, checked._bg, checked._pelvis
        self.bq, self.bv = checked._bq, checked._bv
        self.kp, self.kd, self.caps = parent.kp.copy(), parent.kd.copy(), checked._limits.copy()
        self.contacts = list(parent.contacts)
        self.record_all_ball_contacts = False
        self.warnings = parent.warnings
        self.origin_frame = self.next_frame = parent.next_frame
        self.donor_integration_state_hash = donor_hash
        self.attempted_steps = self.integrated_steps = 0
        self.failed = False

    def review(self) -> dict[str, Any]:
        expected = (300 - self.origin_frame) * 10
        if (
            self.failed
            or self.next_frame != 300
            or self.attempted_steps != expected
            or self.integrated_steps != expected
        ):
            raise ValueError("whole remaining native born-target segment required")
        return dict(
            schema="soccer.physics.g1_born_target_segment.v1",
            origin_frame=self.origin_frame,
            frames=300 - self.origin_frame,
            donor_integration_state_hash=self.donor_integration_state_hash,
            attempted_steps=self.attempted_steps,
            integrated_steps=self.integrated_steps,
            borrowed_prefix_steps_counted_as_new=False,
            policy_causality_verified=False,
            teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
