"""Core neural-torque/teacher contract over one body in a shared SIM clock.

Only the two known pure Core policy implementations are admitted. No Runtime,
driver, hardware executor, transport, checkpoint loader or permit is created.
The parent joint target remains a PD setpoint; actual total actuator torque is
separately observed and fed back at every native step.
"""

from typing import Any

import numpy as np

from rosclaw_soccer.rsi.shared_native_motor_world import SharedNativeMotorWorld


class SharedNativeTorqueLayer:
    def __init__(
        self,
        world: SharedNativeMotorWorld,
        prefix: str,
        policy: Any,
        *,
        target_y_m: float,
        target_z_m: float,
        horizon_controls: int,
        expected_parent_policy_hash: str | None = None,
        intervention_fraction: float = 1.0,
    ) -> None:
        import mujoco
        from rosclaw.simforge.g1_neural_torque import G1NeuralTorquePolicy, G1TeacherTorqueCollector

        world._require_current_boundary()
        if (
            world.native_steps != 0
            or prefix not in world.layouts
            or type(policy) not in (G1NeuralTorquePolicy, G1TeacherTorqueCollector)
            or type(horizon_controls) is not int
            or not 1 <= horizon_controls <= 6000
            or not np.isfinite([target_y_m, target_z_m]).all()
            or not -1.2 <= target_y_m <= 1.2
            or not 0.11 <= target_z_m <= 1.2
            or prefix in world._torque_layer_owners
            or getattr(policy, "_soccer_native_layer_owner", None) is not None
            or not np.isfinite(intervention_fraction)
            or not 0.0 <= intervention_fraction <= 1.0
            or any(other.policy is policy for other in world._torque_layer_owners.values())
        ):
            raise ValueError(
                "new explicit bounded SIM task and private Core torque policy required"
            )
        self.world, self.prefix, self.policy = world, prefix, policy
        self.layout = world.layouts[prefix]
        from rosclaw_soccer.sim.contracts import hash_json

        self.body_hash = hash_json(self.layout.profile())
        if type(policy) is G1NeuralTorquePolicy and (
            policy.artifact.body_hash != self.body_hash
            or expected_parent_policy_hash is None
            or policy.artifact.parent_policy_hash != expected_parent_policy_hash
        ):
            raise ValueError(
                "neural artifact must bind this actual body and explicit parent controller"
            )
        self.target_y_m, self.target_z_m = float(target_y_m), float(target_z_m)
        self.horizon_native_steps = horizon_controls * 10
        self.intervention_fraction = float(intervention_fraction)
        self.torso = int(
            mujoco.mj_name2id(world.model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link")
        )
        self.feet = [
            int(
                mujoco.mj_name2id(
                    world.model, mujoco.mjtObj.mjOBJ_BODY, prefix + side + "_ankle_roll_link"
                )
            )
            for side in ("left", "right")
        ]
        if self.torso < 0 or min(self.feet) < 0:
            raise ValueError("actual named G1 torso and both feet required")
        for body in (self.torso, *self.feet):
            ancestor = body
            while ancestor not in (0, self.layout.pelvis):
                ancestor = int(world.model.body_parentid[ancestor])
            if ancestor != self.layout.pelvis:
                raise ValueError("torso and feet must belong to the actual own player")
        self.next_native_step = self.commands = self.applied = 0
        self.pending = False
        self.failed = False
        policy.reset()
        policy._soccer_native_layer_owner = self
        world._torque_layer_owners[prefix] = self

    def command(self, parent_torque: np.ndarray) -> np.ndarray:
        from rosclaw.simforge.g1_neural_torque import G1TorqueControlFrame

        w, layout = self.world, self.layout
        if (
            self.failed
            or self.pending
            or w.failed
            or w.native_steps != self.next_native_step
            or w._torque_layer_owners.get(self.prefix) is not self
            or not self.next_native_step < self.horizon_native_steps
        ):
            raise ValueError("own consecutive before-integration native torque boundary required")
        parent = np.asarray(parent_torque)
        if (
            parent.shape != (29,)
            or parent.dtype != np.float64
            or not np.isfinite(parent).all()
            or np.any(np.abs(parent) > w.parameters[self.prefix].effort)
        ):
            self.failed = True
            raise ValueError("actual finite float64 HG29 parent torque required")
        m, d = w.model, w.data
        contact = [False, False]
        for point in d.contact:
            bodies = {int(m.geom_bodyid[g]) for g in point.geom}
            for i, foot in enumerate(self.feet):
                contact[i] |= bodies == {0, foot}
        limited = m.jnt_limited[layout.joints].astype(bool)
        bounds = m.jnt_range[layout.joints]
        frame = G1TorqueControlFrame(
            joint_position=d.qpos[layout.qi].copy(),
            joint_velocity=d.qvel[layout.vi].copy(),
            joint_lower_limits=np.where(limited, bounds[:, 0], -1e6),
            joint_upper_limits=np.where(limited, bounds[:, 1], 1e6),
            torso_quaternion_wxyz=d.xquat[self.torso].copy(),
            pelvis_position=d.qpos[layout.rq : layout.rq + 3].copy(),
            ball_position=d.qpos[layout.bq : layout.bq + 3].copy(),
            ball_velocity=d.qvel[layout.bv : layout.bv + 3].copy(),
            target_y_m=self.target_y_m,
            target_z_m=self.target_z_m,
            policy_phase=self.next_native_step / self.horizon_native_steps,
            left_contact=bool(contact[0]),
            right_contact=bool(contact[1]),
        )
        try:
            proposed = np.asarray(self.policy.command(frame, parent.copy()))
            if (
                proposed.shape != (29,)
                or proposed.dtype != np.float64
                or not np.isfinite(proposed).all()
                or np.any(np.abs(proposed) > w.parameters[self.prefix].effort)
            ):
                raise ValueError("finite torque within unchanged actual effort caps required")
            # Note ACTUAL mixed torque back to Core, not its discarded full
            # proposal. Core receipts describe proposal/fallback paths; actual
            # intervention is measured separately in the native world trace.
            if self.intervention_fraction == 0.0:
                proposed = parent.copy()
            elif self.intervention_fraction != 1.0:
                proposed = parent + self.intervention_fraction * (proposed - parent)
        except BaseException:
            self.failed = True
            raise
        self.pending = True
        self.commands += 1
        return proposed.copy()

    def note_applied(self, actual: np.ndarray) -> None:
        if (
            self.failed
            or not self.pending
            or self.world.native_steps != self.next_native_step + 1
            or actual.shape != (29,)
            or actual.dtype != np.float64
            or not np.isfinite(actual).all()
            or not np.array_equal(actual, self.world.data.actuator_force[self.layout.ai])
        ):
            self.failed = True
            raise ValueError("one actual native actuator measurement after command required")
        try:
            self.policy.note_applied(actual.copy())
        except BaseException:
            self.failed = True
            raise
        self.pending = False
        self.next_native_step += 1
        self.applied += 1
