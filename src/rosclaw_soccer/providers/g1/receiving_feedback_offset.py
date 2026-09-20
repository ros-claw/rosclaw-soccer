"""SIM-only measured-state contact proposals, not a qualified teacher.

Reuse the existing contact primitive on private FK data, convert its torque
proposal into a bounded target offset, and retain the live frozen foundation.
No world/data handles, direct torque execution, future trajectory or weights
cross the motor-option boundary. Downstream world guards remain authoritative.
"""

import re
from dataclasses import replace
from pathlib import Path

import numpy as np

from rosclaw_soccer.growth.locomotion_contact_teacher import (
    G1LocomotionContactTeacherConfig,
    locomotion_contact_teacher_effect,
)
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
)


def bounded_contact_offset(torque: np.ndarray, kp: np.ndarray, previous: np.ndarray) -> np.ndarray:
    """Same ±0.1 rad / .25 filter / .02 rad step envelope, never divide by zero."""
    arrays = tuple(np.asarray(v) for v in (torque, kp, previous))
    if any(
        v.shape != (29,) or v.dtype.kind not in "fiu" or not np.isfinite(v).all() for v in arrays
    ):
        raise ValueError("finite 29-joint proposal, gains and predecessor required")
    torque, kp, previous = arrays
    if np.any(kp < 0) or np.any(kp > 300) or np.any(abs(previous) > 0.100000001):
        raise ValueError("bounded gains and actual offset predecessor required")
    if np.any(abs(torque) > 20.000000001):
        raise ValueError("contact primitive torque envelope exceeded")
    desired = np.divide(torque, kp, out=np.zeros(29), where=kp > 1e-6)
    desired = np.clip(desired, -0.1, 0.1)
    return np.asarray(previous + np.clip(0.25 * (desired - previous), -0.02, 0.02))


class MeasuredContactOffset:
    """One immutable-observation stream and one private FK model per instance."""

    def __init__(
        self, asset_root: Path, agent_id: str, *, start_frame: int, enabled: bool = True
    ) -> None:
        import mujoco

        from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
        from rosclaw_soccer.world.field import build_g1_stadium_model

        if (
            type(agent_id) is not str
            or re.fullmatch(r"(?:red|blue)\.(?:defender|finisher|goalkeeper|playmaker)", agent_id)
            is None
            or type(start_frame) is not int
            or not 0 <= start_frame <= 1000
            or type(enabled) is not bool
        ):
            raise ValueError("explicit receiving player and entry required")
        qualified = qualify_g1_assets(asset_root)
        qualified.require_eligible()
        self.agent_id, self.start_frame = agent_id, start_frame
        self._enabled = enabled
        self._model = build_g1_stadium_model(asset_root)
        self._data = mujoco.MjData(self._model)
        if (self._model.nq, self._model.nv) != (43, 41):
            raise ValueError("qualified single-body G1/ball FK layout required")
        for i, name in enumerate(G1_DDS_JOINT_NAMES):
            joint = mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if (
                joint < 0
                or self._model.jnt_qposadr[joint] != 7 + i
                or self._model.jnt_dofadr[joint] != 6 + i
            ):
                raise ValueError("G1 DDS state-to-FK joint order differs")
        self._feet = tuple(
            mujoco.mj_name2id(self._model, mujoco.mjtObj.mjOBJ_BODY, side + "_ankle_roll_link")
            for side in ("left", "right")
        )
        if min(self._feet) < 0:
            raise ValueError("qualified ankle bodies required")
        self._config = replace(
            G1LocomotionContactTeacherConfig(),
            committed_receive_ankle_lateral_offset_m=0.12,
            one_touch_finish_enabled=False,
        )
        self.contract_hash = hash_json(
            dict(
                schema="soccer.measured_contact_offset.v1",
                agent_id=agent_id,
                start_frame=start_frame,
                enabled=enabled,
                body_hash=qualified.body_hash,
                teacher_config_hash=self._config.config_hash,
                observation="current immutable state and completed attributed contact",
                offset_rad=0.1,
                filter=0.25,
                step_rad=0.02,
                qualified_teacher=False,
                activation_ceiling="SIM_ONLY",
            )
        )
        self._next_frame = 0
        self._last_physics_time = -1.0
        self._first_contact: float | None = None
        self._contact_left = False
        self._offset = np.zeros(29)
        self._faulted = False
        self._foundation_identity: tuple[str, str] | None = None
        self.records: list[dict[str, object]] = []

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if self._faulted:
            raise ValueError("contact option fault is latched")
        try:
            if not isinstance(observation, TeamMotorPhysicsObservation):
                raise ValueError("typed completed physics observation required")
            observation.__post_init__()
            if (
                not observation.contacts_complete
                or observation.observer_agent_id != self.agent_id
                or observation.time_sec <= self._last_physics_time
            ):
                raise ValueError("ordered attributed completed contact required")
            self._last_physics_time = observation.time_sec
            for contact in observation.ball_contacts:
                if (
                    contact.agent_id == self.agent_id
                    and contact.is_foot
                    and contact.normal_force_n > 0
                    and self._first_contact is None
                ):
                    self._first_contact = observation.time_sec
                    self._contact_left = contact.effector == "left_foot"
        except (ValueError, TypeError, FloatingPointError):
            self._faulted = True
            raise

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        import mujoco

        if self._faulted:
            raise ValueError("contact option fault is latched")
        try:
            if not isinstance(observation, TeamMotorObservation):
                raise ValueError("typed motor observation required")
            observation.__post_init__()
            if (
                observation.agent_id != self.agent_id
                or observation.frame != self._next_frame
                or observation.frame > 1000
                or abs(observation.time_sec - observation.frame * 0.02) > 1e-6
                or any(
                    abs(np.linalg.norm(observation.qpos[a:b]) - 1) > 1e-5
                    for a, b in ((3, 7), (39, 43))
                )
            ):
                raise ValueError("consecutive same-player observations required")
            self._next_frame += 1
            if observation.frame < self.start_frame:
                return None
            foundation = observation.foundation
            if foundation is None:
                raise ValueError("actual same-tick frozen foundation required")
            foundation.__post_init__()
            foundation.target.__post_init__()
            identity = (foundation.policy_hash, foundation.configuration_hash)
            if self._foundation_identity is not None and identity != self._foundation_identity:
                raise ValueError("frozen foundation identity changed")
            self._foundation_identity = identity
            if (
                observation.frame > self.start_frame
                and abs(self._last_physics_time - observation.time_sec) > 1e-6
            ):
                raise ValueError("completed contact observation missing or stale")
            if self._last_physics_time > observation.time_sec + 1e-6:
                raise ValueError("future contact cannot influence current proposal")
            self._data.qpos[:] = observation.qpos
            self._data.qvel[:] = observation.qvel
            self._data.time = observation.time_sec
            mujoco.mj_forward(self._model, self._data)  # private FK model only
            ball = np.asarray(observation.qpos[36:39])
            direction = np.asarray([1.0 if self.agent_id.startswith("red.") else -1.0, 0.0])
            recent = (
                self._first_contact is not None
                and 0 <= observation.time_sec - self._first_contact <= 0.6
            )
            left = (
                self._contact_left
                if recent
                else bool(
                    np.linalg.norm(self._data.xpos[self._feet[0]] - ball)
                    < np.linalg.norm(self._data.xpos[self._feet[1]] - ball)
                )
            )
            config = self._config
            if recent:
                config = replace(
                    config,
                    receive_ankle_lateral_offset_m=config.committed_receive_ankle_lateral_offset_m,
                    velocity_damping_n_per_mps=config.committed_receive_velocity_damping_n_per_mps,
                    maximum_task_force_n=config.committed_receive_maximum_task_force_n,
                    maximum_joint_residual_nm=config.committed_receive_maximum_joint_residual_nm,
                    aim_yaw_bias_rad=config.committed_receive_aim_yaw_bias_rad,
                )
            # Anatomical side projected into the chosen target's lateral axis.
            lateral = np.asarray([-direction[1], direction[0]])
            w, x, y, z = observation.qpos[3:7]
            yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            anatomical = np.asarray([-np.sin(yaw), np.cos(yaw)])
            sign = (1.0 if anatomical @ lateral >= 0 else -1.0) * (1.0 if left else -1.0)
            effect = locomotion_contact_teacher_effect(
                model=self._model,
                data=self._data,
                ankle_body_id=self._feet[0 if left else 1],
                actuated_dof_indices=np.arange(6, 35),
                ball_position_m=ball,
                ball_velocity_mps=np.asarray(observation.qvel[35:38]),
                desired_ball_direction_xy=direction,
                contact_mode="receive",
                local_lateral_sign=sign,
                contact_recent=recent,
                config=config,
                receive_capture_progress=(
                    min(1.0, (observation.time_sec - self._first_contact) / 0.6)
                    if recent and self._first_contact is not None
                    else None
                ),
            )
            kp = np.asarray(foundation.target.kp)
            self._offset = bounded_contact_offset(
                effect.torque_nm if self._enabled else np.zeros(29), kp, self._offset
            )
            target = np.asarray(foundation.target.target_rad) + self._offset
            self.records.append(
                dict(
                    frame=observation.frame,
                    offset_rad=self._offset.tolist(),
                    foundation_target_rad=list(foundation.target.target_rad),
                    task_torque_proposal_nm=effect.torque_nm.tolist(),
                    active=effect.active,
                    foot="left" if left else "right",
                    measured_contact_recent=recent,
                )
            )
            return TeamMotorTarget(
                tuple(float(x) for x in target), foundation.target.kp, foundation.target.kd
            )
        except (ValueError, TypeError, RuntimeError, FloatingPointError):
            self._faulted = True
            raise
