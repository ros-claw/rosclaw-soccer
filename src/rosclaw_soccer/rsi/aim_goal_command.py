"""Explicit desired-command input for SIM research; never alter measured ball/body state."""

from dataclasses import dataclass
from typing import Any

import numpy as np


@dataclass(frozen=True)
class AimGoalCommand:
    world_point: tuple[float, float, float]

    def target(self) -> np.ndarray:
        if (
            type(self.world_point) is not tuple
            or len(self.world_point) != 3
            or any(type(v) is not float for v in self.world_point)
        ):
            raise ValueError("immutable finite world-frame desired aim command required")
        point = np.array(self.world_point, dtype=np.float64)
        if (
            not np.isfinite(point).all()
            or point[0] != 8.0
            or abs(point[1]) > 2.0
            or not 0.11 <= point[2] <= 1.11
        ):
            raise ValueError("declared SIM aim-command domain x8/y+-2/z.11..1.11 required")
        return point.astype(np.float32)


def condition_aim_goal(policy: Any, command: AimGoalCommand) -> dict[str, Any]:
    if type(command) is not AimGoalCommand:
        raise ValueError("explicit desired aim-command object required")
    target = command.target()
    if (
        policy.runtime_mode != "sim"
        or policy.use_body_frame_ball
        or policy.control_dt != 0.02
        or int(policy.time_step) < int(policy.WARMUP_STEPS)
        or type(policy.target_pos_w) is not np.ndarray
        or policy.target_pos_w.shape != (3,)
        or not np.isfinite(policy.target_pos_w).all()
    ):
        raise ValueError(
            "qualified active SIM policy with an explicit existing desired goal required"
        )
    old = policy.target_pos_w.copy()
    changed = not np.array_equal(target, old)
    if changed:
        # Only future entries change when the original policy appends them.
        # Never rewrite any past target history buffer.
        policy.target_pos_w = target.copy()
    return dict(
        schema="soccer.sim_desired_aim_command.v1",
        applied=changed,
        previous_command_world_point=old.tolist(),
        command_world_point=target.tolist(),
        past_history_modified=False,
        measured_ball_modified=False,
        physical_state_modified=False,
        reference_motion_modified=False,
        neural_weights_modified=False,
        task_scoring_goal_modified=False,
        hardware_authorized=False,
    )
