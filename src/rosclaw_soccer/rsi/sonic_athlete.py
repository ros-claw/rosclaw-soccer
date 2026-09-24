"""Proprioceptive, SIM_ONLY adapter for the existing SONIC navigation policy.

The legacy team observation contains seven ball qpos and six ball qvel values.
SONIC reads only the leading robot state; this adapter pads the ignored tail
solely to reuse its vetted navigation implementation. It is not ball contact
control, a new learned policy, or an actuator executor.
"""

from __future__ import annotations

import math

from rosclaw_soccer.providers.g1.sonic_navigation import G1SonicNavigation
from rosclaw_soccer.rsi.contracts import (
    AthleteObservation,
    AthleticIntent,
    MotorAction,
    PolicyArtifact,
    validate_athlete_proposal,
)
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.skills.team.motor_option import TeamMotorObservation, TeamMotorTarget


def _gain_hash(target: TeamMotorTarget) -> str:
    return str(hash_json({"kp": list(target.kp), "kd": list(target.kd)}))


class SonicAthleteAdapter:
    """One SONIC instance per athlete/episode, fail-closed after any bad tick."""

    def __init__(self, navigation: G1SonicNavigation, artifact: PolicyArtifact) -> None:
        if not isinstance(navigation, G1SonicNavigation) or not isinstance(
            artifact, PolicyArtifact
        ):
            raise ValueError("qualified SONIC navigation and typed artifact required")
        qualification = navigation.backend.qualification
        qualification.require_eligible()
        expected_weights = hash_json(
            {
                "planner": qualification.planner_hash,
                "encoder": qualification.encoder_hash,
                "decoder": qualification.decoder_hash,
            }
        )
        kp = tuple(float(value) for value in navigation.backend.kp)
        kd = tuple(float(value) for value in navigation.backend.kd)
        expected_gains = hash_json({"kp": list(kp), "kd": list(kd)})
        if (
            artifact.backend_id != "sonic_navigation"
            or artifact.backend_contract_hash != navigation.contract_hash
            or artifact.weights_hash != expected_weights
            or artifact.gain_hash != expected_gains
            or artifact.action_kind != "JOINT_TARGET"
            or artifact.action_size != 29
            or not math.isclose(artifact.control_dt_s, 0.02, abs_tol=1.0e-9)
        ):
            raise ValueError("SONIC model, gain, action and clock binding mismatch")
        self.navigation = navigation
        self.artifact = artifact
        self._started = False
        self._faulted = False

    def step(self, observation: AthleteObservation, intent: AthleticIntent) -> MotorAction:
        if self._faulted:
            raise ValueError("SONIC athlete fault latched for this episode")
        try:
            if (
                not isinstance(observation, AthleteObservation)
                or not isinstance(intent, AthleticIntent)
                or observation.body_id != self.navigation.agent_id
                or observation.body_hash != self.artifact.body_hash
                or observation.joint_map_hash != self.artifact.joint_map_hash
                or observation.physics_hash != self.artifact.physics_hash
                or abs(observation.time_sec - observation.frame * 0.02) > 1.0e-6
                or len(observation.joint_position) != 29
                or intent.contact_intent not in ("none", "locomotion")
            ):
                raise ValueError("foreign or contact-seeking SONIC athlete state")
            target_xy = intent.future_target_xy_m or (0.0, 0.0)
            # The existing SONIC team input is robot qpos[0:36]/qvel[0:35]
            # followed by a ball free joint. The backend never consumes the tail.
            # It is intentionally non-physical padding, not a ball observation.
            qpos = (
                *observation.root_position_m,
                *observation.root_quaternion_wxyz,
                *observation.joint_position,
                0.0,
                0.0,
                0.0,
                1.0,
                0.0,
                0.0,
                0.0,
            )
            qvel = (
                *observation.root_velocity_mps,
                *observation.root_angular_velocity_rad_s,
                *observation.joint_velocity,
                *(0.0,) * 6,
            )
            motor_observation = TeamMotorObservation(
                agent_id=observation.body_id,
                frame=observation.frame,
                time_sec=observation.time_sec,
                intent="other",
                prospective_owner=False,
                qpos=qpos,
                qvel=qvel,
                target_position_m=(target_xy[0], target_xy[1], 0.0),
                navigation_command=(
                    *intent.velocity_xy_mps,
                    intent.yaw_rate_rad_s,
                ),
                navigation_envelope=self.navigation.navigation_envelope,
            )
            if not self._started:
                self.navigation.start_from_observation(motor_observation)
                self._started = True
            proposal = self.navigation.propose(motor_observation)
            if _gain_hash(proposal) != self.artifact.gain_hash:
                raise ValueError("SONIC gains changed after artifact binding")
            action = MotorAction(
                self.artifact.body_hash,
                self.artifact.contract_hash,
                observation.frame,
                joint_target=proposal.target_rad,
            )
            validate_athlete_proposal(self.artifact, observation, action)
            return action
        except Exception as error:
            self._faulted = True
            raise ValueError("SONIC athlete proposal failed; episode latched off") from error
