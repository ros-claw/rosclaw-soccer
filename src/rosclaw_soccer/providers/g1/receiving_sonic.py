"""Delayed frozen SONIC entry for paired receiving experiments, SIM_ONLY.

Before entry the world must explicitly retain its existing controller through
motor_idle_residual_fallback. At entry SONIC cold-starts from measured state;
this does not claim transfer of the old policy's hidden state.
"""

import math
from dataclasses import replace
from pathlib import Path

from rosclaw_soccer.providers.g1.sonic_command_scale import SonicCommandScaleSchedule
from rosclaw_soccer.providers.g1.sonic_latent import SonicLatentSchedule
from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation, SonicNavigationConfig
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


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
            vx, vy, yaw = observation.navigation_command
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
