"""Training-only task-space receive teacher on one live shared SIM body."""

from dataclasses import replace

import mujoco
import numpy as np

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    locomotion_contact_teacher_effect,
)
from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


class SharedNativeReceiveTeacher:
    """Apply measured-geometry torque proposals; never reposition bodies or ball.

    The shared world calls this once before each native substep and records the
    actual resulting actuator torque. This is explicitly an assisted training
    teacher, not a learned policy, a hardware command, or a success gate.
    """

    def __init__(self, world: SharedNativeMotorWorld, prefix: str, *, horizon_controls: int):
        world._require_current_boundary()
        if (
            type(world) is not SharedNativeMotorWorld
            or prefix not in world.layouts
            or prefix in world._torque_layer_owners
            or type(horizon_controls) is not int
            or not 1 <= horizon_controls <= 600
            or len(world.layouts) != 2
        ):
            raise ValueError("own unmodified two-player SIM boundary required")
        self.world, self.prefix = world, prefix
        self.policy = self
        self.layout = world.layouts[prefix]
        self.teammate = next(p for p in world.layouts if p != prefix)
        self.feet = tuple(
            int(mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, prefix + side))
            for side in ("left_ankle_roll_link", "right_ankle_roll_link")
        )
        if min(self.feet) < 0:
            raise ValueError("own named left and right ankles required")
        base = G1LocomotionContactTeacherConfig()
        self.config = replace(
            base,
            receive_follow_through_speed_mps=base.committed_receive_follow_through_speed_mps,
            receive_ankle_lateral_offset_m=base.committed_receive_ankle_lateral_offset_m,
            velocity_damping_n_per_mps=base.committed_receive_velocity_damping_n_per_mps,
            maximum_task_force_n=base.committed_receive_maximum_task_force_n,
            maximum_joint_residual_nm=base.committed_receive_maximum_joint_residual_nm,
            aim_yaw_bias_rad=base.committed_receive_aim_yaw_bias_rad,
        )
        self.next_native_step = world.native_steps
        self.horizon_native_steps = self.next_native_step + 10 * horizon_controls
        self.pending = self.failed = False
        self.commands = self.applied = self.active_commands = 0
        self._last_foot_contact_step: int | None = None
        world._torque_layer_owners[prefix] = self

    def command(self, parent_torque: np.ndarray) -> np.ndarray:
        w, layout = self.world, self.layout
        parent = np.asarray(parent_torque)
        if (
            self.failed
            or self.pending
            or w.failed
            or w.native_steps != self.next_native_step
            or w._torque_layer_owners.get(self.prefix) is not self
            or self.next_native_step >= self.horizon_native_steps
            or parent.shape != (29,)
            or parent.dtype != np.float64
            or not np.isfinite(parent).all()
            or np.any(np.abs(parent) > w.parameters[self.prefix].effort)
        ):
            self.failed = True
            raise ValueError("consecutive own native parent torque required")
        try:
            m, d = w.model, w.data
            mujoco.mj_forward(m, d)
            ball = d.qpos[layout.bq : layout.bq + 3].copy()
            ball_velocity = d.qvel[layout.bv : layout.bv + 3].copy()
            direction = (
                d.qpos[w.layouts[self.teammate].rq : w.layouts[self.teammate].rq + 2] - ball[:2]
            )
            lengths = [np.linalg.norm(d.xpos[foot] - ball) for foot in self.feet]
            use_left = bool(lengths[0] < lengths[1])
            foot = self.feet[0 if use_left else 1]
            quat = d.qpos[layout.rq + 3 : layout.rq + 7]
            yaw = np.arctan2(
                2 * (quat[0] * quat[3] + quat[1] * quat[2]), 1 - 2 * (quat[2] ** 2 + quat[3] ** 2)
            )
            lateral = np.array([-direction[1], direction[0]])
            anatomical_left = np.array([-np.sin(yaw), np.cos(yaw)])
            left_sign = 1.0 if np.dot(anatomical_left, lateral) >= 0 else -1.0
            recent = self._last_foot_contact_step is not None and (
                self.next_native_step - self._last_foot_contact_step <= 150
            )
            effect = locomotion_contact_teacher_effect(
                model=m,
                data=d,
                ankle_body_id=foot,
                actuated_dof_indices=layout.vi,
                ball_position_m=ball,
                ball_velocity_mps=ball_velocity,
                desired_ball_direction_xy=direction.copy(),
                contact_mode="receive",
                local_lateral_sign=left_sign if use_left else -left_sign,
                contact_recent=recent,
                config=self.config,
            )
            proposal = np.clip(
                parent + effect.torque_nm,
                -w.parameters[self.prefix].effort,
                w.parameters[self.prefix].effort,
            )
            if not np.isfinite(proposal).all():
                raise ValueError("finite teacher proposal required")
            self.active_commands += int(effect.active)
        except BaseException:
            self.failed = True
            raise
        self.pending = True
        self.commands += 1
        return proposal

    def note_applied(self, actual: np.ndarray) -> None:
        w, layout = self.world, self.layout
        if (
            self.failed
            or not self.pending
            or w.native_steps != self.next_native_step + 1
            or actual.shape != (29,)
            or actual.dtype != np.float64
            or not np.isfinite(actual).all()
            or not np.array_equal(actual, w.data.actuator_force[layout.ai])
        ):
            self.failed = True
            raise ValueError("actual applied native torque required")
        for i, contact in enumerate(w.data.contact):
            bodies = {int(w.model.geom_bodyid[g]) for g in contact.geom}
            if w.ball in bodies and any(foot in bodies for foot in self.feet):
                force = np.zeros(6)
                mujoco.mj_contactForce(w.model, w.data, i, force)
                if np.linalg.norm(force[:3]) > 1.0:
                    self._last_foot_contact_step = self.next_native_step
        self.pending = False
        self.next_native_step += 1
        self.applied += 1
