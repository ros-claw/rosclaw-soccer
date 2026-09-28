"""Delayed frozen SONIC entry for paired receiving experiments, SIM_ONLY.

Before entry the world must explicitly retain its existing controller through
motor_idle_residual_fallback. At entry SONIC cold-starts from measured state;
this does not claim transfer of the old policy's hidden state.
"""

import math
from dataclasses import replace
from pathlib import Path

from rosclaw_soccer.providers.g1.feedback_leg_actor import FrozenFeedbackLegActor
from rosclaw_soccer.providers.g1.sonic_command_scale import SonicCommandScaleSchedule
from rosclaw_soccer.providers.g1.sonic_latent import SonicLatentSchedule
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
)


class ReceivingSonicOption:
    def __init__(
        self,
        model_root: Path,
        agent_id: str,
        *,
        start_frame: int,
        velocity_scale: float = 1.0,
        planner_seed: int = 920101,
        latent_schedule: SonicLatentSchedule | None = None,
        command_scale_schedule: SonicCommandScaleSchedule | None = None,
        pose_reference: SonicPoseReference | None = None,
        experimental_command_replanning: bool = False,
    ) -> None:
        if (
            type(start_frame) is not int
            or not 0 <= start_frame <= 1000
            or type(velocity_scale) not in (int, float)
            or not math.isfinite(velocity_scale)
            or not 0 <= velocity_scale <= 1
        ):
            raise ValueError("bounded declared SONIC entry and velocity scale required")
        if command_scale_schedule is not None:
            if not isinstance(command_scale_schedule, SonicCommandScaleSchedule):
                raise ValueError("typed command attenuation schedule required")
            command_scale_schedule.__post_init__()
        if pose_reference is not None and (
            velocity_scale != 0 or command_scale_schedule is not None
        ):
            raise ValueError("fixed reference probe must explicitly disable navigation commands")
        self.agent_id = agent_id
        self.start_frame = start_frame
        self.velocity_scale = velocity_scale
        self.command_scale_schedule = command_scale_schedule
        self.command_scale_records: list[tuple[int, float]] = []
        self.navigation = G1SonicNavigation(
            model_root,
            agent_id,
            SonicNavigationConfig(
                maximum_frames=1000,
                planner_seed=planner_seed,
                model_variant="low_latency",
                latent_schedule=latent_schedule,
                pose_reference=pose_reference,
                experimental_command_replanning=experimental_command_replanning,
            ),
        )
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.receiving_sonic_option.v1",
                    "navigation": self.navigation.contract_hash,
                    "start_frame": start_frame,
                    "velocity_scale": velocity_scale,
                    "history": "cold_start_at_measured_entry_not_hidden_state_transfer",
                    "activation_ceiling": "SIM_ONLY",
                    **(
                        {"command_scale_schedule_hash": command_scale_schedule.contract_hash}
                        if command_scale_schedule is not None
                        else {}
                    ),
                }
            )
        )
        self.next_frame = 0
        self.faulted = False

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        if self.faulted:
            raise ValueError("receiving SONIC fault is latched")
        try:
            if (
                observation.agent_id != self.agent_id
                or observation.frame != self.next_frame
                or abs(observation.time_sec - observation.frame * 0.02) > 1e-6
            ):
                raise ValueError("consecutive same-player measured observations required")
            self.next_frame += 1
            if observation.frame < self.start_frame:
                return None
            if observation.navigation_command is None:
                raise ValueError("receiving navigation command unavailable")
            # No larger command is introduced. Physical collision guards and
            # evidence remain necessary; scaling is not a new clearance proof.
            vx, vy, yaw = self._navigation_command(observation)
            scale = self.velocity_scale
            if self.command_scale_schedule is not None:
                scale *= self.command_scale_schedule.at(observation.frame - self.start_frame)
            measured = replace(
                observation,
                navigation_command=(
                    float(vx * scale),
                    float(vy * scale),
                    float(yaw * scale),
                ),
            )
            if observation.frame == self.start_frame:
                self.navigation.start_from_observation(measured)
            result = self.navigation.propose(measured)
            if self.command_scale_schedule is not None:
                self.command_scale_records.append((observation.frame - self.start_frame, scale))
            return result
        except (ValueError, TypeError, RuntimeError, FloatingPointError):
            self.faulted = True
            raise

    def _navigation_command(self, observation: TeamMotorObservation) -> tuple[float, float, float]:
        if observation.navigation_command is None:
            raise ValueError("receiving navigation command unavailable")
        return tuple(float(value) for value in observation.navigation_command)  # type: ignore[return-value]


class ReceivingSonicBallFollowOption(ReceivingSonicOption):
    """SIM-only current-ball velocity following for a measured receiving probe."""

    def __init__(
        self,
        model_root: Path,
        agent_id: str,
        *,
        start_frame: int,
        response_gain: float,
        fast_replan: bool = False,
        post_touch_chase: bool = False,
        brake_distance_m: float | None = None,
        brake_axis: str = "xy",
        post_touch_target_distance_m: float = 0.30,
        post_touch_speed_limit_mps: float | None = None,
    ) -> None:
        if (
            type(response_gain) not in (int, float)
            or not math.isfinite(response_gain)
            or not 0 < response_gain <= 1
        ):
            raise ValueError("bounded ball-follow response gain required")
        if (
            type(fast_replan) is not bool
            or type(post_touch_chase) is not bool
            or (post_touch_chase and not fast_replan)
        ):
            raise ValueError("explicit ball-follow replan and chase modes required")
        if brake_distance_m is not None and (
            type(brake_distance_m) not in (int, float)
            or not math.isfinite(brake_distance_m)
            or not 0.45 <= brake_distance_m <= 0.90
            or not fast_replan
        ):
            raise ValueError("bounded early receiving brake requires fast replanning")
        if brake_axis not in ("xy", "x") or (brake_axis != "xy" and brake_distance_m is None):
            raise ValueError("explicit bounded receiving brake axis required")
        if (
            type(post_touch_target_distance_m) not in (int, float)
            or not math.isfinite(post_touch_target_distance_m)
            or not 0.30 <= post_touch_target_distance_m <= 0.50
            or post_touch_target_distance_m != 0.30
            and not post_touch_chase
            or post_touch_speed_limit_mps is not None
            and (
                type(post_touch_speed_limit_mps) not in (int, float)
                or not math.isfinite(post_touch_speed_limit_mps)
                or not 0.25 <= post_touch_speed_limit_mps <= 0.70
                or not post_touch_chase
            )
        ):
            raise ValueError("bounded post-touch follow radius and speed required")
        super().__init__(
            model_root,
            agent_id,
            start_frame=start_frame,
            experimental_command_replanning=fast_replan,
        )
        self.response_gain = float(response_gain)
        self.fast_replan = fast_replan
        self.post_touch_chase = post_touch_chase
        self.brake_distance_m = float(brake_distance_m) if brake_distance_m is not None else None
        self.brake_axis = brake_axis
        self.post_touch_target_distance_m = float(post_touch_target_distance_m)
        self.post_touch_speed_limit_mps = (
            float(post_touch_speed_limit_mps) if post_touch_speed_limit_mps is not None else None
        )
        self.contact_foot_seen = False
        self.last_physics_time_sec = -1.0
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.receiving_sonic_ball_follow_option.v1",
                    "base": self.contract_hash,
                    "response_gain": self.response_gain,
                    "fast_replan": fast_replan,
                    "post_touch_chase": post_touch_chase,
                    "brake_distance_m": self.brake_distance_m,
                    **({"brake_axis": brake_axis} if brake_axis != "xy" else {}),
                    **(
                        {"post_touch_target_distance_m": self.post_touch_target_distance_m}
                        if self.post_touch_target_distance_m != 0.30
                        else {}
                    ),
                    **(
                        {"post_touch_speed_limit_mps": self.post_touch_speed_limit_mps}
                        if self.post_touch_speed_limit_mps is not None
                        else {}
                    ),
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )

    def _navigation_command(self, observation: TeamMotorObservation) -> tuple[float, float, float]:
        original = super()._navigation_command(observation)
        dx = observation.qpos[36] - observation.qpos[0]
        dy = observation.qpos[37] - observation.qpos[1]
        distance = math.hypot(dx, dy)
        if self.post_touch_chase and self.contact_foot_seen and distance > 0.05:
            approach = 2.0 * (distance - self.post_touch_target_distance_m) / distance
            vx = observation.qvel[35] + approach * dx
            vy = observation.qvel[36] + approach * dy
            limit = (
                observation.navigation_envelope.maximum_speed_mps
                if observation.navigation_envelope is not None
                else 0.7
            )
            if self.post_touch_speed_limit_mps is not None:
                limit = min(limit, self.post_touch_speed_limit_mps)
            speed = math.hypot(vx, vy)
            if speed > limit:
                vx *= limit / speed
                vy *= limit / speed
            return float(vx), float(vy), float(original[2])
        if distance >= 1.2 or distance < 0.15:
            return original
        weight = self.response_gain * min(1.0, (1.2 - distance) / 0.4)
        lateral_weight = weight
        if self.brake_distance_m is not None and not self.contact_foot_seen:
            brake = max(
                0.0,
                min(1.0, (distance - 0.25) / (self.brake_distance_m - 0.25)),
            )
            weight *= brake
            if self.brake_axis == "xy":
                lateral_weight *= brake
        vx = (1.0 - weight) * original[0] + weight * observation.qvel[35]
        vy = (1.0 - lateral_weight) * original[1] + lateral_weight * observation.qvel[36]
        limit = (
            observation.navigation_envelope.maximum_speed_mps
            if observation.navigation_envelope is not None
            else 0.7
        )
        speed = math.hypot(vx, vy)
        if speed > limit:
            vx *= limit / speed
            vy *= limit / speed
        return float(vx), float(vy), float(original[2])

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if (
            self.faulted
            or not isinstance(observation, TeamMotorPhysicsObservation)
            or not observation.contacts_complete
            or observation.observer_agent_id != self.agent_id
            or observation.time_sec <= self.last_physics_time_sec
        ):
            self.faulted = True
            raise ValueError("complete same-player ball-follow contact evidence required")
        self.last_physics_time_sec = observation.time_sec
        self.contact_foot_seen |= any(
            contact.agent_id == self.agent_id and contact.is_foot
            for contact in observation.ball_contacts
        )


class ReceivingSonicFeedbackOption(ReceivingSonicOption):
    """Explicit opt-in per-player feedback motor; legacy SONIC remains unchanged."""

    def __init__(
        self,
        model_root: Path,
        agent_id: str,
        *,
        start_frame: int,
        feedback_actor_path: Path,
        velocity_scale: float = 1.0,
        planner_seed: int = 920101,
        latent_schedule: SonicLatentSchedule | None = None,
        command_scale_schedule: SonicCommandScaleSchedule | None = None,
        pose_reference: SonicPoseReference | None = None,
        experimental_command_replanning: bool = False,
    ) -> None:
        super().__init__(
            model_root,
            agent_id,
            start_frame=start_frame,
            velocity_scale=velocity_scale,
            planner_seed=planner_seed,
            latent_schedule=latent_schedule,
            command_scale_schedule=command_scale_schedule,
            pose_reference=pose_reference,
            experimental_command_replanning=experimental_command_replanning,
        )
        self.feedback_actor = FrozenFeedbackLegActor.load(feedback_actor_path)
        self.feedback_foot_seen = False
        self.feedback_nonfoot_seen = False
        self.feedback_last_physics_time_sec = -1.0
        self.contract_hash = str(
            hash_json(
                {
                    "schema": "soccer.receiving_sonic_feedback_option.v1",
                    "base": self.contract_hash,
                    "feedback_actor_hash": self.feedback_actor.artifact_hash,
                    "activation_ceiling": "SIM_ONLY",
                }
            )
        )

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        result = super().propose(observation)
        if result is None:
            return None
        try:
            return self.feedback_actor.propose(
                observation,
                result,
                foot_seen=self.feedback_foot_seen,
                nonfoot_seen=self.feedback_nonfoot_seen,
            )
        except (ValueError, TypeError, FloatingPointError):
            self.faulted = True
            raise

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if self.faulted:
            raise ValueError("latched receiving feedback motor cannot observe")
        if (
            not isinstance(observation, TeamMotorPhysicsObservation)
            or not observation.contacts_complete
            or observation.observer_agent_id != self.agent_id
            or observation.time_sec <= self.feedback_last_physics_time_sec
        ):
            self.faulted = True
            raise ValueError("complete consecutive same-player contact evidence required")
        self.feedback_last_physics_time_sec = observation.time_sec
        for contact in observation.ball_contacts:
            if contact.agent_id != self.agent_id:
                continue
            if contact.is_foot:
                self.feedback_foot_seen = True
            else:
                self.feedback_nonfoot_seen = True


class RecordingReceivingSonicBallFollowOption(ReceivingSonicBallFollowOption):
    """Read-only training capture of the measured-ball SONIC teacher."""

    def __init__(
        self,
        model_root: Path,
        agent_id: str,
        *,
        start_frame: int,
        response_gain: float,
        fast_replan: bool = False,
        post_touch_chase: bool = False,
        brake_distance_m: float | None = None,
        brake_axis: str = "xy",
        post_touch_target_distance_m: float = 0.30,
        post_touch_speed_limit_mps: float | None = None,
    ) -> None:
        super().__init__(
            model_root,
            agent_id,
            start_frame=start_frame,
            response_gain=response_gain,
            fast_replan=fast_replan,
            post_touch_chase=post_touch_chase,
            brake_distance_m=brake_distance_m,
            brake_axis=brake_axis,
            post_touch_target_distance_m=post_touch_target_distance_m,
            post_touch_speed_limit_mps=post_touch_speed_limit_mps,
        )
        self.recorded: list[tuple[TeamMotorObservation, TeamMotorTarget | None]] = []

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        target = super().propose(observation)
        self.recorded.append((observation, target))
        return target


class RecordingReceivingSonicOption(ReceivingSonicOption):
    """Read-only exact motor-observation capture for offline receiving training."""

    def __init__(
        self,
        model_root: Path,
        agent_id: str,
        *,
        start_frame: int,
        velocity_scale: float = 1.0,
        planner_seed: int = 920101,
        latent_schedule: SonicLatentSchedule | None = None,
        command_scale_schedule: SonicCommandScaleSchedule | None = None,
        pose_reference: SonicPoseReference | None = None,
        experimental_command_replanning: bool = False,
    ) -> None:
        super().__init__(
            model_root,
            agent_id,
            start_frame=start_frame,
            velocity_scale=velocity_scale,
            planner_seed=planner_seed,
            latent_schedule=latent_schedule,
            command_scale_schedule=command_scale_schedule,
            pose_reference=pose_reference,
            experimental_command_replanning=experimental_command_replanning,
        )
        self.recorded: list[tuple[TeamMotorObservation, TeamMotorTarget | None]] = []

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget | None:
        target = super().propose(observation)
        self.recorded.append((observation, target))
        return target
