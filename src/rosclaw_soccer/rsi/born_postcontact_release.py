"""New current-boundary60ms torque lease; no fabricated episode prefix.

Shares the legacy measured-Jacobian action algorithm, not its time-zero
constructor/history. A surrounding SIM actor must authenticate the completed
body/contact boundary. This control performs zero physics or NN inference.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.impact_torque_screen import CAPS
from rosclaw_soccer.rsi.postcontact_release_control import (
    ACTION_NAMES,
    PostContactReleaseCommand,
    PostContactReleaseControl,
)

BORN_RELEASE_ACTIONS = {"born_release:" + name: name for name in ACTION_NAMES}


@dataclass(frozen=True)
class BornPostContactReleaseCommand(PostContactReleaseCommand):
    """Distinct capture namespace and provenance from a time-zero controller."""


class BornPostContactReleaseControl(PostContactReleaseControl):
    def __init__(self, model: Any, *, world_hash: str, action_name: str, start_frame: int) -> None:
        if (
            type(start_frame) is not int
            or not 31 <= start_frame < 300
            or type(action_name) is not str
            or action_name not in ACTION_NAMES
        ):
            raise ValueError("declared born-at-current post-foot release required")
        # Deliberately do not call the time-zero constructor and overwrite its
        # clock. This object starts with ZERO observations at its declared
        # origin; no preceding control command is claimed to have run.
        self.model = model
        self.sampler = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash)
        self.action_name = action_name
        self.origin_frame = start_frame
        self.next_frame = start_frame
        self.observed_command_frames = 0
        self.trigger_frame: int | None = None
        self.contact_side: int | None = None
        self.action_available: bool | None = None
        self.held = np.zeros(29)
        self.cancelled = self.failed = False

    def plan(
        self, data: Any, *, frame: int, previous_contact_forces: np.ndarray, pelvis_z_m: float
    ) -> BornPostContactReleaseCommand:
        if (
            self.failed
            or self.next_frame != self.origin_frame + self.observed_command_frames
            or (
                self.observed_command_frames == 0
                and (
                    type(previous_contact_forces) is not np.ndarray
                    or previous_contact_forces.shape != (6,)
                    or previous_contact_forces.dtype != np.float64
                    or not np.isfinite(previous_contact_forces).all()
                    or not np.any(previous_contact_forces[:2] > 1)
                )
            )
        ):
            self.failed = True
            raise ValueError("new current foot-force lease cannot borrow missing observations")
        command = super().plan(
            data,
            frame=frame,
            previous_contact_forces=previous_contact_forces,
            pelvis_z_m=pelvis_z_m,
        )
        if (
            command.offset_nm.shape != (29,)
            or not np.isfinite(command.offset_nm).all()
            or np.any(np.abs(command.offset_nm) > CAPS)
        ):
            self.failed = True
            raise ValueError("unchanged bounded measured-Jacobian action required")
        self.observed_command_frames += 1
        report = dict(
            command.report,
            schema="soccer.rsi.born_postcontact_release_command.v1",
            origin_frame=self.origin_frame,
            observed_command_frames=self.observed_command_frames,
            commands_before_origin_observed=0,
            control_born_at_current_boundary=True,
        )
        return BornPostContactReleaseCommand(command.offset_nm.copy(), report)

    def review(self) -> dict[str, Any]:
        if self.observed_command_frames != 300 - self.origin_frame:
            raise ValueError("every remaining born-lease command required")
        return dict(
            super().review(),
            schema="soccer.rsi.born_postcontact_release_review.v1",
            origin_frame=self.origin_frame,
            observed_command_frames=self.observed_command_frames,
            commands_before_origin_observed=0,
            new_physical_steps=0,
            new_nn_inferences=0,
        )
