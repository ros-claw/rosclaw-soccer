"""SIM_ONLY bilateral adapter for the existing causal two-glove decoder.

This is a motor teacher, not a newly trained actor.  A detached canonical
MuJoCo model supplies Jacobians; only bounded arm targets leave this adapter.
No ball state, body pose, collision geometry, or shared control is written.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.providers.g1.keeper_frame import KeeperFrame
from rosclaw_soccer.skills.goalkeeper_v2.observations import GoalkeeperActorObserver


@dataclass(frozen=True)
class SharedKeeperReachConfig:
    gain_scale: float = 1.5
    maximum_step_rad: float = 0.15
    maximum_memory_rad: float = 0.80
    memory_decay: float = 0.96
    maximum_horizon_sec: float = 1.0
    exit_step_rad: float = 0.04
    gmt_model_path: str | None = None
    gmt_skill_path: str | None = None
    gmt_leg_blend: float = 0.0
    gmt_waist_blend: float = 0.0
    gmt_arm_blend: float = 1.0
    reference_tracking: bool = False
    neutralize_foundation_arm_observation: bool = False
    ready_reference_time_sec: float | None = None
    timing_lead_sec: float = 0.0
    punch_force_n: float = 0.0
    entry_step_rad: float = 0.10
    ballistic_airborne_velocity: bool = False
    minimum_intercept_confidence: float = 0.5
    impedance_scale: float = 1.0
    reach_forward_m: float = 0.24
    reach_height_offset_m: float = 0.0
    hand_half_span_m: float = 0.08
    support_arm_blend: float = 1.0
    support_overhead_bias_rad: float = 0.0
    muscle_actor_path: str | None = None

    def __post_init__(self) -> None:
        bounds = (
            (self.gain_scale, 0.5, 4.0),
            (self.maximum_step_rad, 0.04, 0.25),
            (self.maximum_memory_rad, 0.20, 0.80),
            (self.memory_decay, 0.75, 0.98),
            (self.maximum_horizon_sec, 0.10, 1.20),
            (self.exit_step_rad, 0.01, 0.10),
            (self.gmt_leg_blend, 0, 1),
            (self.gmt_waist_blend, 0, 1),
            (self.gmt_arm_blend, 0, 1),
            (self.timing_lead_sec, 0, 0.30),
            (self.punch_force_n, 0, 40),
            (self.entry_step_rad, 0.04, 2.50),
            (self.minimum_intercept_confidence, 0.25, 1),
            (self.impedance_scale, 1, 3),
            (self.reach_forward_m, -0.35, 0.35),
            (self.reach_height_offset_m, -0.12, 0.12),
            (self.hand_half_span_m, 0.04, 0.20),
            (self.support_arm_blend, 0, 1),
            (self.support_overhead_bias_rad, 0, 0.35),
        )
        if any(not np.isfinite(v) or not lo <= v <= hi for v, lo, hi in bounds):
            raise ValueError("shared keeper reach configuration outside SIM_ONLY envelope")
        if (self.gmt_model_path is None) != (self.gmt_skill_path is None):
            raise ValueError("GMT model and bound imitation skill must be supplied together")
        if self.ready_reference_time_sec is not None and (
            not np.isfinite(self.ready_reference_time_sec)
            or not -0.50 <= self.ready_reference_time_sec <= -0.30
            or self.gmt_model_path is None
        ):
            raise ValueError(
                "ready pose requires the bound imitation model and bounded reference phase"
            )
        if (
            type(self.reference_tracking) is not bool
            or type(self.neutralize_foundation_arm_observation) is not bool
            or type(self.ballistic_airborne_velocity) is not bool
        ):
            raise ValueError("reference tracking selector must be boolean")


class SharedKeeperReach:
    """One observer and recovery memory per keeper, independent of team yaw."""

    def __init__(
        self,
        *,
        asset_root: Path,
        goal: Any,
        frame: KeeperFrame,
        prefix: str,
        config: SharedKeeperReachConfig | None = None,
    ) -> None:
        import mujoco

        from rosclaw_soccer.world.field import build_g1_stadium_model

        self.config = config or SharedKeeperReachConfig()
        self.frame, self.prefix = frame, prefix
        self.model = build_g1_stadium_model(asset_root, goal)
        self.data = mujoco.MjData(self.model)
        ids = np.array([self.model.joint(n).id for n in G1_DDS_JOINT_NAMES])
        self.robot = SimpleNamespace(
            origin=np.array((4.52, 0, 0)),
            world_from_local_quat=np.array((0, 0, 0, 1)),
            qpos_base=0,
            joint_ids=ids,
            joint_qvel=self.model.jnt_dofadr[ids].copy(),
            left_hand_body=self.model.body("left_wrist_yaw_link").id,
            right_hand_body=self.model.body("right_wrist_yaw_link").id,
            last_target=np.zeros(29),
            goalkeeper_reach_memory=np.zeros(29),
            goalkeeper_reach_memory_peak_rad=0.0,
        )
        self.artifact = SimpleNamespace(
            operational_space_reach_damping=0.08,
            operational_space_reach_gain=0.25,
            operational_space_reach_maximum_step_rad=self.config.maximum_step_rad,
            operational_space_reach_ramp_sec=0.18,
            operational_space_memory_decay=self.config.memory_decay,
        )
        self.observer = GoalkeeperActorObserver(
            ballistic_airborne_velocity=self.config.ballistic_airborne_velocity
        )
        self.previous = np.zeros(29)
        self.last_time: float | None = None
        self.started: float | None = None
        self.active = False
        self.peak_residual_rad = 0.0
        self.torque_nm = np.zeros(29)
        self.contact_time: float | None = None
        self.contact_target: np.ndarray | None = None
        self.last_output: np.ndarray | None = None
        self.last_intercept = (0.0, 0.0, 1.3)
        self.last_muscle_observation: np.ndarray | None = None

        self.gmt: Any = None
        self.mirror_latch: bool | None = None
        self.gmt_contract: Any = None
        self.muscle: Any = None
        if self.config.muscle_actor_path is not None:
            from rosclaw_soccer.providers.g1.keeper_muscle_actor import KeeperMuscleActor

            self.muscle = KeeperMuscleActor(Path(self.config.muscle_actor_path))
        if self.config.gmt_model_path is not None and self.config.gmt_skill_path is not None:
            import torch

            from rosclaw_soccer.growth.mosaic_gmt import (
                MosaicGMTTorchController,
                load_g1_mosaic_gmt_overhead_skill,
                load_mosaic_gmt_torch,
            )

            policy, self.gmt_contract = load_mosaic_gmt_torch(
                Path(self.config.gmt_model_path), device=torch.device("cpu")
            )
            skill = load_g1_mosaic_gmt_overhead_skill(Path(self.config.gmt_skill_path))
            self.gmt = MosaicGMTTorchController(
                policy=policy,
                contract=self.gmt_contract,
                skill=skill,
                environment_count=1,
                device=torch.device("cpu"),
            )

    @property
    def policy_hash(self) -> str:
        from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

        return str(
            hash_json(
                {
                    "config": asdict(self.config),
                    "adapter": hash_bytes(Path(__file__).read_bytes()),
                    "model": None
                    if self.gmt_contract is None
                    else self.gmt_contract.checkpoint_hash,
                    "imitation": None if self.gmt is None else self.gmt.skill.skill_hash,
                    "muscle": None if self.muscle is None else self.muscle.policy_hash,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )

    def step(self, model: Any, data: Any, foundation_target: np.ndarray) -> np.ndarray:
        import mujoco

        from rosclaw_soccer.providers.g1.mujoco_primitives import mirror_g1_joint_positions
        from rosclaw_soccer.skills.team.independent_team_world import _gravity_orientation
        from rosclaw_soccer.skills.team.shared_world import (
            _apply_goalkeeper_bimanual_operational_space_reach,
            _apply_goalkeeper_bimanual_support_arm,
            _goalkeeper_bimanual_punch_torque,
            _mirror_gmt_proprioception,
        )

        target = np.asarray(foundation_target, dtype=np.float64)
        self.last_muscle_observation = None
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("keeper foundation target must contain 29 finite values")
        snapshot = self.frame.project(model, data, prefix=self.prefix)
        if self.last_time is not None and not np.isclose(
            snapshot.time - self.last_time, 0.02, atol=1e-7, rtol=0
        ):
            raise ValueError("keeper decoder requires monotonic 50 Hz observations")
        self.last_time = snapshot.time
        self.data.qpos[:] = snapshot.qpos
        self.data.qvel[:] = snapshot.qvel
        mujoco.mj_forward(self.model, self.data)
        # Stable birth frame: canonical keeper faces -x.  Position history,
        # not privileged ball velocity or a future launch cue, drives reach.
        local_ball = snapshot.qpos[36:39] - np.array((4.52, 0, 0))
        local_ball[:2] *= -1
        observation = self.observer.observe(
            timestamp_sec=snapshot.time,
            ball_relative_position_m=local_ball,
            gravity_orientation=_gravity_orientation(snapshot.qpos[3:7]),
            root_linear_velocity_mps=snapshot.qvel[:3],
            angular_velocity_rad_s=snapshot.qvel[3:6],
            joint_position_rad=snapshot.qpos[7:36],
            joint_velocity_rad_s=snapshot.qvel[6:35],
            previous_action_rad=self.previous,
        )
        horizon, lateral, height = observation.estimated_intercept
        gravity = _gravity_orientation(snapshot.qpos[3:7])
        active = bool(
            observation.intercept_confidence >= self.config.minimum_intercept_confidence
            and 0 < horizon <= self.config.maximum_horizon_sec
            and 0.65 <= height <= 1.70
            and abs(lateral + snapshot.qpos[1]) <= 0.65
            and snapshot.qpos[2] >= 0.60
            and gravity[2] < -0.8
            and local_ball[0] > -0.12
        )
        if self.active and observation.intercept_confidence >= 0.5:
            # Preserve a single incoming motor epoch across a noisy height
            # boundary. Never retain authority across a fall or receding ball.
            active = active or bool(
                0 < horizon <= 1.2
                and 0.55 <= height <= 1.85
                and local_ball[0] > 0
                and snapshot.qpos[2] >= 0.60
                and gravity[2] < -0.8
            )
        following_contact = bool(
            self.contact_time is not None and snapshot.time - self.contact_time <= 0.18
        )
        if following_contact:
            active = bool(snapshot.qpos[2] >= 0.60 and gravity[2] < -0.8)
            observation = replace(observation, estimated_intercept=(0.0, *self.last_intercept[1:]))
        elif self.contact_time is not None and snapshot.time - self.contact_time > 0.5:
            self.contact_time = None
        self.robot.last_target = target.copy()
        ready_active = False
        if self.gmt is not None:
            import torch

            def tensor(value: Any) -> Any:
                return torch.as_tensor(np.asarray(value, dtype=np.float32)[None, ...])

            if not active:
                self.mirror_latch = None
            elif self.mirror_latch is None:
                self.mirror_latch = bool(lateral > 0.10)
            position, velocity, torso, angular = (
                snapshot.qpos[7:36],
                snapshot.qvel[6:35],
                self.data.xquat[self.model.body("torso_link").id],
                snapshot.qvel[3:6],
            )
            if self.mirror_latch:
                position, velocity, torso, angular = _mirror_gmt_proprioception(
                    position, velocity, torso, angular
                )

            gmt_target, _ = self.gmt.target(
                canonical_joint_position=tensor(position),
                canonical_joint_velocity=tensor(velocity),
                torso_quaternion_wxyz=tensor(torso),
                base_angular_velocity_body_rad_s=tensor(angular),
                heading_quaternion_wxyz=tensor((0, 0, 0, 1)),
                relative_time_sec=torch.tensor([min(0.0, -horizon + self.config.timing_lead_sec)]),
                active=torch.tensor([active]),
            )
            if active:
                if self.config.reference_tracking:
                    gmt_target = self.gmt.reference_target(
                        torch.tensor([min(0.0, -horizon + self.config.timing_lead_sec)])
                    )
                scales = np.array(
                    (self.config.gmt_leg_blend,) * 12
                    + (self.config.gmt_waist_blend,) * 3
                    + (self.config.gmt_arm_blend,) * 14
                )
                decoded = gmt_target[0].numpy()
                if self.mirror_latch:
                    decoded = mirror_g1_joint_positions(decoded)
                self.robot.last_target += scales * (decoded - target)
                self.robot.hold_target = target.copy()
                _apply_goalkeeper_bimanual_support_arm(
                    cast(Any, self.robot),
                    local_intercept_y_m=1.0 if self.mirror_latch else -1.0,
                    blend=self.config.support_arm_blend,
                    overhead_bias_rad=self.config.support_overhead_bias_rad,
                )
            else:
                self.gmt.reset()
                if self.config.ready_reference_time_sec is not None and local_ball[0] > 0:
                    ready_target = self.gmt.reference_target(
                        torch.tensor([self.config.ready_reference_time_sec])
                    )
                    self.robot.last_target[15:] = ready_target[0, 15:].numpy()
                    ready_active = True
        if self.muscle is not None and active and not following_contact:
            from rosclaw_soccer.providers.g1.keeper_muscle_actor import muscle_observation

            self.last_muscle_observation = muscle_observation(
                np.asarray(observation.estimated_intercept),
                float(snapshot.qpos[2]),
                gravity,
                snapshot.qpos[7:36],
                snapshot.qvel[6:35],
            )
            self.robot.last_target[15:] = self.muscle.target(self.last_muscle_observation)
        if following_contact and active and self.contact_target is not None:
            self.robot.last_target = self.contact_target.copy()
        posture_delta = self.robot.last_target - target
        if active and not following_contact and self.muscle is None:
            if self.started is None:
                self.started = snapshot.time
            _apply_goalkeeper_bimanual_operational_space_reach(
                cast(Any, self.robot),
                model=self.model,
                data=self.data,
                observation=observation,
                artifact=self.artifact,
                target_local_x_m=self.config.reach_forward_m,
                half_span_m=self.config.hand_half_span_m,
                height_offset_m=self.config.reach_height_offset_m,
                reach_fraction=1.0,
                gain_scale=self.config.gain_scale,
                memory_decay=self.config.memory_decay,
                memory_maximum_rad=self.config.maximum_memory_rad,
                elapsed_sec=snapshot.time - self.started,
            )
        else:
            self.started = None
            self.robot.goalkeeper_reach_memory *= self.config.memory_decay
        # Include the decoder's early-exit decay, and rate-limit activation as
        # well as release. Legs and waist are exclusively foundation-owned.
        desired = (
            posture_delta
            if following_contact and active
            else (
                posture_delta + (self.robot.goalkeeper_reach_memory if self.muscle is None else 0)
                if active
                else (posture_delta if ready_active else np.zeros(29))
            )
        )
        step = self.config.entry_step_rad if active else self.config.exit_step_rad
        residual = self.previous + np.clip(desired - self.previous, -step, step)
        writable_start = (
            0
            if self.gmt is not None and self.config.gmt_leg_blend > 0
            else (12 if self.gmt is not None and self.config.gmt_waist_blend > 0 else 15)
        )
        residual[:writable_start] = 0
        residual[:12] = self.previous[:12] + np.clip(
            residual[:12] - self.previous[:12], -0.04, 0.04
        )
        ids = self.robot.joint_ids[writable_start:]
        ranges = self.model.jnt_range[ids]
        residual[writable_start:] = (
            np.clip(
                target[writable_start:] + residual[writable_start:],
                ranges[:, 0] + 0.08,
                ranges[:, 1] - 0.08,
            )
            - target[writable_start:]
        )
        self.previous = residual.copy()
        self.active = active
        self.torque_nm = np.zeros(29)
        if active and self.config.punch_force_n > 0:
            torque, _ = _goalkeeper_bimanual_punch_torque(
                cast(Any, self.robot),
                model=self.model,
                data=self.data,
                observation=observation,
                force_n=self.config.punch_force_n,
                vertical_force_scale=0.0,
                outward_force_scale=0.0,
                window_sec=0.30,
            )
            self.torque_nm = np.clip(torque, -12.0, 12.0)
        self.peak_residual_rad = max(self.peak_residual_rad, float(np.max(np.abs(residual))))
        self.last_output = np.asarray(target + residual, dtype=np.float64)
        if not following_contact:
            self.last_intercept = observation.estimated_intercept
        return self.last_output.copy()

    def notify_glove_contact(self, timestamp_sec: float) -> None:
        """Latch an actual physics contact, never a predicted or scheduled save."""
        if (
            not np.isfinite(timestamp_sec)
            or self.last_time is None
            or not self.last_time <= timestamp_sec <= self.last_time + 0.020001
        ):
            raise ValueError("keeper contact must belong to the current physics control frame")
        if self.active and self.contact_time is None and self.last_output is not None:
            self.contact_time = timestamp_sec
            self.contact_target = self.last_output.copy()

    def impedance(self, kp: np.ndarray, kd: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Track the motion with its own impedance; smoothly return to Loco."""
        result_kp, result_kd = np.asarray(kp).copy(), np.asarray(kd).copy()
        if self.gmt_contract is not None:
            from rosclaw_soccer.growth.mosaic_g1_contract import MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF

            order = np.asarray(MOSAIC_G1_ISAACLAB_TO_CANONICAL_DOF)
            blend = float(np.clip(np.max(np.abs(self.previous)) / 0.20, 0, 1))
            scales = np.array(
                (self.config.gmt_leg_blend,) * 12
                + (self.config.gmt_waist_blend,) * 3
                + (self.config.gmt_arm_blend,) * 14
            )
            for result, values, scale in (
                (result_kp, self.gmt_contract.joint_stiffness, self.config.impedance_scale),
                (result_kd, self.gmt_contract.joint_damping, np.sqrt(self.config.impedance_scale)),
            ):
                result[:] = (1 - blend * scales) * result + blend * scales * np.asarray(values)[
                    order
                ] * scale
        return result_kp, result_kd
