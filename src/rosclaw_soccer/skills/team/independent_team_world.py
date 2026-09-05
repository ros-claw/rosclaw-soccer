"""Six-G1 role-autonomy arena backed by independent ROSClaw agent cells.

The arena is a locomotion and coordination bridge, not a complete learned
match.  Every G1 receives its own egocentric observation, private policy
identity, memory namespaces, and role-authorized decision.  Decisions become
bounded velocity commands for the frozen RoboNaldo locomotion policy.  The
module never writes a root pose after initialization, never writes football
state after initialization, and exposes no hardware path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.independent_agent_cell import (
    AgentCellDecision,
    AgentCellObservation,
    AgentPhysicalState,
    RosclawSoccerAgentCell,
    build_team_coordination_frame,
)
from rosclaw_soccer.growth.role_self_model import (
    MatchRole,
    TacticalIntent,
    TeamRoleRoster,
)
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    load_robonaldo,
    mirror_g1_joint_gains,
    mirror_g1_joint_positions,
)
from rosclaw_soccer.sim.contracts import G1_DDS_JOINT_NAMES, G1_HARD_TORQUE_LIMITS, hash_json
from rosclaw_soccer.world.field import (
    G1CompliantGoalNetState,
    G1TrainingGoalSpec,
    apply_g1_compliant_goal_net_force,
)
from rosclaw_soccer.world.multi_player import (
    G1PitchPlayerSpec,
    build_g1_multi_player_stadium_model,
)

_CONTROL_DT = 0.02
_PHYSICS_DT = 0.002
_SUBSTEPS = 10
_MOTION_REL = Path("policy/robonaldo/model/freekick_motion.npz")
_INTENT_CODES = {intent: index for index, intent in enumerate(TacticalIntent)}
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


@dataclass(frozen=True)
class IndependentTeamWorldConfig:
    simulation_duration_sec: float = 10.0
    decision_period_sec: float = 0.10
    maximum_speed_mps: float = 0.38
    goalkeeper_maximum_speed_mps: float = 0.30
    maximum_acceleration_mps2: float = 0.60
    position_gain: float = 0.85
    arrival_radius_m: float = 0.14
    minimum_player_separation_m: float = 0.95
    collision_avoidance_gain: float = 1.20
    maximum_collision_correction_mps: float = 0.28
    possession_radius_m: float = 1.25
    left_goal_plane_x_m: float = -1.50
    minimum_pelvis_height_m: float = 0.55
    maximum_tilt_rad: float = 0.80
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.independent_team_world_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.simulation_duration_sec,
            self.decision_period_sec,
            self.maximum_speed_mps,
            self.goalkeeper_maximum_speed_mps,
            self.maximum_acceleration_mps2,
            self.position_gain,
            self.arrival_radius_m,
            self.minimum_player_separation_m,
            self.collision_avoidance_gain,
            self.maximum_collision_correction_mps,
            self.possession_radius_m,
            self.left_goal_plane_x_m,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 5.0 <= self.simulation_duration_sec <= 25.0
            or not 0.08 <= self.decision_period_sec <= 0.20
            or not 0.20 <= self.maximum_speed_mps <= 0.70
            or not 0.20 <= self.goalkeeper_maximum_speed_mps <= 0.60
            or not 0.30 <= self.maximum_acceleration_mps2 <= 3.0
            or not 0.50 <= self.position_gain <= 3.0
            or not 0.05 <= self.arrival_radius_m <= 0.25
            or not 0.55 <= self.minimum_player_separation_m <= 1.20
            or not 0.20 <= self.collision_avoidance_gain <= 3.0
            or not 0.05 <= self.maximum_collision_correction_mps <= 0.30
            or not 0.50 <= self.possession_radius_m <= 1.50
            or not -5.0 <= self.left_goal_plane_x_m <= 0.0
            or not 0.45 <= self.minimum_pelvis_height_m <= 0.70
            or not 0.45 <= self.maximum_tilt_rad <= 1.00
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("independent team world violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class IndependentTeamWorldScenario:
    scenario_id: str
    ball_initial_position_m: tuple[float, float, float]
    ball_initial_velocity_mps: tuple[float, float, float]
    seed: int
    schema_version: str = "rosclaw_soccer.independent_team_world_scenario.v1"

    def __post_init__(self) -> None:
        values = (*self.ball_initial_position_m, *self.ball_initial_velocity_mps)
        if (
            not self.scenario_id.startswith("s199.")
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= 2**32 - 1
            or len(self.ball_initial_position_m) != 3
            or len(self.ball_initial_velocity_mps) != 3
            or any(not math.isfinite(value) for value in values)
            or self.ball_initial_position_m[2] <= 0.0
        ):
            raise ValueError("independent team scenario is invalid")

    @property
    def scenario_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class AgentWorldQuality:
    agent_id: str
    role: MatchRole
    active_fraction: float
    displacement_m: float
    distinct_intent_count: int
    intent_switch_count: int
    decision_count: int
    pass_intent_count: int
    shot_intent_count: int
    save_intent_count: int
    distribution_intent_count: int
    minimum_pelvis_height_m: float
    maximum_tilt_rad: float
    joint_limit_violation: bool
    torque_limit_violation: bool
    required_minimum_pelvis_height_m: float
    allowed_maximum_tilt_rad: float
    schema_version: str = "rosclaw_soccer.agent_world_quality.v1"

    def __post_init__(self) -> None:
        counts = (
            self.distinct_intent_count,
            self.intent_switch_count,
            self.decision_count,
            self.pass_intent_count,
            self.shot_intent_count,
            self.save_intent_count,
            self.distribution_intent_count,
        )
        values = (
            self.active_fraction,
            self.displacement_m,
            self.minimum_pelvis_height_m,
            self.maximum_tilt_rad,
            self.required_minimum_pelvis_height_m,
            self.allowed_maximum_tilt_rad,
        )
        if (
            not _IDENTIFIER.fullmatch(self.agent_id)
            or not isinstance(self.role, MatchRole)
            or any(isinstance(value, bool) or value < 0 for value in counts)
            or any(not math.isfinite(value) for value in values)
            or not 0.0 <= self.active_fraction <= 1.0
            or self.displacement_m < 0.0
            or self.minimum_pelvis_height_m < 0.0
            or self.maximum_tilt_rad < 0.0
            or not 0.45 <= self.required_minimum_pelvis_height_m <= 0.70
            or not 0.45 <= self.allowed_maximum_tilt_rad <= 1.00
            or not isinstance(self.joint_limit_violation, bool)
            or not isinstance(self.torque_limit_violation, bool)
        ):
            raise ValueError("agent world quality contract is invalid")

    @property
    def safe(self) -> bool:
        return bool(
            self.minimum_pelvis_height_m >= self.required_minimum_pelvis_height_m
            and self.maximum_tilt_rad <= self.allowed_maximum_tilt_rad
            and not self.joint_limit_violation
            and not self.torque_limit_violation
        )

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["role"] = self.role.value
        value["safe"] = self.safe
        return value


@dataclass(frozen=True)
class IndependentTeamWorldResult:
    scenario_hash: str
    roster_hash: str
    config_hash: str
    trajectory_hash: str
    player_count: int
    red_player_count: int
    blue_player_count: int
    decision_frame_count: int
    coordination_frame_hashes: tuple[str, ...]
    qualities: tuple[AgentWorldQuality, ...]
    pass_handshake_count: int
    pass_intent_count: int
    shot_intent_count: int
    save_intent_count: int
    distribution_intent_count: int
    finite_state: bool
    robot_robot_contact_count: int
    rolling_distance_m: float
    peak_ball_speed_mps: float
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.independent_team_world_result.v1"

    def __post_init__(self) -> None:
        counts = (
            self.player_count,
            self.red_player_count,
            self.blue_player_count,
            self.decision_frame_count,
            self.pass_handshake_count,
            self.pass_intent_count,
            self.shot_intent_count,
            self.save_intent_count,
            self.distribution_intent_count,
            self.robot_robot_contact_count,
        )
        quality_ids = tuple(value.agent_id for value in self.qualities)
        if (
            any(
                not _HASH.fullmatch(value)
                for value in (
                    self.scenario_hash,
                    self.roster_hash,
                    self.config_hash,
                    self.trajectory_hash,
                )
            )
            or any(isinstance(value, bool) or value < 0 for value in counts)
            or self.player_count != self.red_player_count + self.blue_player_count
            or self.player_count != len(self.qualities)
            or len(quality_ids) != len(set(quality_ids))
            or self.red_player_count != sum(agent_id.startswith("red.") for agent_id in quality_ids)
            or self.blue_player_count
            != sum(agent_id.startswith("blue.") for agent_id in quality_ids)
            or len(self.coordination_frame_hashes) != self.decision_frame_count
            or any(not _HASH.fullmatch(value) for value in self.coordination_frame_hashes)
            or self.pass_handshake_count != self.pass_intent_count
            or self.pass_intent_count != sum(value.pass_intent_count for value in self.qualities)
            or self.shot_intent_count != sum(value.shot_intent_count for value in self.qualities)
            or self.save_intent_count != sum(value.save_intent_count for value in self.qualities)
            or self.distribution_intent_count
            != sum(value.distribution_intent_count for value in self.qualities)
            or not math.isfinite(self.rolling_distance_m)
            or not math.isfinite(self.peak_ball_speed_mps)
            or self.rolling_distance_m < 0.0
            or self.peak_ball_speed_mps < 0.0
            or not isinstance(self.finite_state, bool)
            or not isinstance(self.hardware_command_sent, bool)
            or not isinstance(self.pixels_used_for_scoring, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.physics_authority != "CPU_MUJOCO"
            or self.hardware_command_sent
            or self.pixels_used_for_scoring
        ):
            raise ValueError("independent team result contract is invalid")

    @property
    def all_agents_active(self) -> bool:
        return all(value.active_fraction >= 0.20 for value in self.qualities)

    @property
    def all_roles_autonomous(self) -> bool:
        return all(
            value.distinct_intent_count >= 1 and value.decision_count == self.decision_frame_count
            for value in self.qualities
        )

    @property
    def role_complete_both_teams(self) -> bool:
        required = {MatchRole.GOALKEEPER, MatchRole.PLAYMAKER, MatchRole.FINISHER}
        return all(
            required.issubset(
                {value.role for value in self.qualities if value.agent_id.startswith(f"{team_id}.")}
            )
            for team_id in ("red", "blue")
        )

    @property
    def safe(self) -> bool:
        return bool(
            self.finite_state
            and all(value.safe for value in self.qualities)
            and self.robot_robot_contact_count == 0
            and not self.hardware_command_sent
            and not self.pixels_used_for_scoring
        )

    @property
    def passed(self) -> bool:
        return bool(
            self.player_count >= 6
            and self.red_player_count >= 3
            and self.blue_player_count >= 3
            and self.decision_frame_count >= 10
            and len(self.coordination_frame_hashes) == self.decision_frame_count
            and len(set(self.coordination_frame_hashes)) == self.decision_frame_count
            and self.all_agents_active
            and self.all_roles_autonomous
            and self.role_complete_both_teams
            and self.safe
        )

    @property
    def result_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "scenario_hash": self.scenario_hash,
            "roster_hash": self.roster_hash,
            "config_hash": self.config_hash,
            "trajectory_hash": self.trajectory_hash,
            "player_count": self.player_count,
            "red_player_count": self.red_player_count,
            "blue_player_count": self.blue_player_count,
            "decision_frame_count": self.decision_frame_count,
            "coordination_frame_hashes": list(self.coordination_frame_hashes),
            "qualities": [value.to_dict() for value in self.qualities],
            "pass_handshake_count": self.pass_handshake_count,
            "pass_intent_count": self.pass_intent_count,
            "shot_intent_count": self.shot_intent_count,
            "save_intent_count": self.save_intent_count,
            "distribution_intent_count": self.distribution_intent_count,
            "finite_state": self.finite_state,
            "robot_robot_contact_count": self.robot_robot_contact_count,
            "rolling_distance_m": self.rolling_distance_m,
            "peak_ball_speed_mps": self.peak_ball_speed_mps,
            "all_agents_active": self.all_agents_active,
            "all_roles_autonomous": self.all_roles_autonomous,
            "safe": self.safe,
            "passed": self.passed,
            "activation_ceiling": self.activation_ceiling,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
            "pixels_used_for_scoring": self.pixels_used_for_scoring,
        }


@dataclass
class _PlayerController:
    spec: G1PitchPlayerSpec
    cell: RosclawSoccerAgentCell
    qpos_base: int
    qvel_base: int
    joint_ids: NDArray[np.int64]
    joint_qpos: NDArray[np.int64]
    joint_qvel: NDArray[np.int64]
    actuators: NDArray[np.int64]
    pelvis_body: int
    torso_body: int
    left_ankle_body: int
    right_ankle_body: int
    robot_geoms: frozenset[int]
    state: Any
    output: Any
    policy: Any
    decision: AgentCellDecision | None = None
    last_world_command: NDArray[np.float64] | None = None
    current_intent: TacticalIntent | None = None
    intent_switch_count: int = 0
    seen_intents: set[TacticalIntent] | None = None
    active_frames: int = 0
    decision_count: int = 0
    pass_intent_count: int = 0
    shot_intent_count: int = 0
    save_intent_count: int = 0
    distribution_intent_count: int = 0
    minimum_pelvis_height_m: float = math.inf
    maximum_tilt_rad: float = 0.0
    joint_limit_violation: bool = False
    torque_limit_violation: bool = False

    def __post_init__(self) -> None:
        if self.seen_intents is None:
            self.seen_intents = set()


def simulate_independent_team_world(
    *,
    asset_root: Path,
    roster: TeamRoleRoster,
    cells: tuple[RosclawSoccerAgentCell, ...],
    players: tuple[G1PitchPlayerSpec, ...],
    scenario: IndependentTeamWorldScenario,
    goal: G1TrainingGoalSpec,
    config: IndependentTeamWorldConfig | None = None,
) -> tuple[IndependentTeamWorldResult, dict[str, NDArray[Any]]]:
    """Run all agent cells and all six neural locomotion bodies in one clock."""

    active = config or IndependentTeamWorldConfig()
    cell_by_id = {cell.agent_id: cell for cell in cells}
    player_by_id = {player.agent_id: player for player in players}
    roster_ids = {agent.agent_id for agent in roster.agents}
    if (
        len(roster.agents) < 6
        or set(cell_by_id) != roster_ids
        or set(player_by_id) != roster_ids
        or len(cell_by_id) != len(cells)
        or len(player_by_id) != len(players)
    ):
        raise ValueError("independent team world roster/cell/body identities differ")
    for cell in cells:
        if cell.self_model.self_model_hash != roster.agent(cell.agent_id).self_model_hash:
            raise ValueError("independent team world cell changed after roster commitment")
        if abs(cell.tactical_profile.decision_period_sec - active.decision_period_sec) > 1.0e-12:
            raise ValueError("agent and world decision clocks differ")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()

    import mujoco

    model = build_g1_multi_player_stadium_model(asset_root, players=players, spec=goal)
    model.opt.timestep = _PHYSICS_DT
    data = mujoco.MjData(model)
    state_type, output_type, _, _ = load_robonaldo(qualification.asset_root)
    loco_type = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
    with np.load(qualification.asset_root / _MOTION_REL) as motion:
        pelvis_height = float(np.asarray(motion["body_pos_w"])[0, 0, 2])
    controllers = tuple(
        _make_player_controller(
            model=model,
            data=data,
            spec=player_by_id[agent.agent_id],
            cell=cell_by_id[agent.agent_id],
            pelvis_height=pelvis_height,
            state_type=state_type,
            output_type=output_type,
            loco_type=loco_type,
        )
        for agent in sorted(roster.agents, key=lambda item: item.agent_id)
    )
    ball_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    data.qpos[ball_qpos : ball_qpos + 3] = scenario.ball_initial_position_m
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[ball_qvel : ball_qvel + 3] = scenario.ball_initial_velocity_mps
    data.qvel[ball_qvel + 3 : ball_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)

    total_frames = int(round(active.simulation_duration_sec / _CONTROL_DT))
    decision_stride = max(1, int(round(active.decision_period_sec / _CONTROL_DT)))
    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    guarded_limits = 0.85 * hard_limits
    trace: dict[str, list[Any]] = {
        "time": [],
        "ball_pose": [],
        "ball_velocity": [],
        "coordination_frame_index": [],
    }
    for controller in controllers:
        key = _agent_key(controller.cell.agent_id)
        trace.update(
            {
                f"{key}_pelvis_pose": [],
                f"{key}_joint_position": [],
                f"{key}_joint_velocity": [],
                f"{key}_joint_torque": [],
                f"{key}_left_foot_position": [],
                f"{key}_right_foot_position": [],
                f"{key}_intent_code": [],
                f"{key}_target_position": [],
                f"{key}_world_command": [],
                f"{key}_movement_active": [],
            }
        )
    initial_positions = {
        controller.cell.agent_id: np.asarray(
            data.qpos[controller.qpos_base : controller.qpos_base + 3], dtype=np.float64
        ).copy()
        for controller in controllers
    }
    coordination_hashes: list[str] = []
    intent_counts = {intent: 0 for intent in TacticalIntent}
    pass_handshake_count = 0
    current_coordination_index = -1
    finite = True
    robot_contact_count = 0
    peak_ball_speed = float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3]))
    initial_ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64).copy()
    net_state = G1CompliantGoalNetState()

    for frame in range(total_frames):
        for controller in controllers:
            _fill_locomotion_state(controller, data, ball_body, ball_qvel)
        if frame % decision_stride == 0:
            physical_states = tuple(_physical_state(controller, data) for controller in controllers)
            state_by_id = {state.agent_id: state for state in physical_states}
            possession = _infer_possession(
                controllers=controllers,
                data=data,
                ball_position=np.asarray(data.qpos[ball_qpos : ball_qpos + 3]),
                maximum_distance_m=active.possession_radius_m,
            )
            observations = tuple(
                _agent_observation(
                    controller=controller,
                    roster=roster,
                    state_by_id=state_by_id,
                    data=data,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    goal=goal,
                    left_goal_plane_x_m=active.left_goal_plane_x_m,
                    possession_agent_id=possession,
                )
                for controller in controllers
            )
            decisions = tuple(
                cell_by_id[observation.observer_agent_id].decide(observation)
                for observation in observations
            )
            coordination = build_team_coordination_frame(
                roster=roster,
                cells=cells,
                observations=observations,
                decisions=decisions,
                frame_index=len(coordination_hashes),
            )
            coordination_hashes.append(coordination.frame_hash)
            pass_handshake_count += len(coordination.pass_receive_handshakes)
            for decision in decisions:
                intent_counts[decision.intent] += 1
            current_coordination_index += 1
            for decision in decisions:
                controller = next(
                    item for item in controllers if item.cell.agent_id == decision.agent_id
                )
                if controller.current_intent is not None and (
                    controller.current_intent is not decision.intent
                ):
                    controller.intent_switch_count += 1
                controller.current_intent = decision.intent
                assert controller.seen_intents is not None
                controller.seen_intents.add(decision.intent)
                controller.decision = decision
                controller.decision_count += 1
                controller.pass_intent_count += int(decision.intent is TacticalIntent.PASS)
                controller.shot_intent_count += int(decision.intent is TacticalIntent.SHOOT)
                controller.save_intent_count += int(decision.intent is TacticalIntent.SAVE)
                controller.distribution_intent_count += int(
                    decision.intent is TacticalIntent.DISTRIBUTE
                )

        positions = {
            controller.cell.agent_id: np.asarray(
                data.qpos[controller.qpos_base : controller.qpos_base + 2], dtype=np.float64
            ).copy()
            for controller in controllers
        }
        for controller in controllers:
            current_decision = controller.decision
            if current_decision is None:
                raise RuntimeError("independent agent has no current decision")
            command = _movement_command(
                controller=controller,
                decision=current_decision,
                positions=positions,
                data=data,
                ball_qpos=ball_qpos,
                config=active,
            )
            local_command = _rotate_z(command, -controller.spec.yaw_rad)
            controller.state.vel_cmd = _normalized_locomotion_command(
                controller.policy, local_command
            )
            _run_locomotion(controller, mirror=bool(local_command[1] < -1.0e-6))
            controller.last_world_command = command.copy()
            controller.active_frames += int(float(np.linalg.norm(command[:2])) >= 0.04)
        for _ in range(_SUBSTEPS):
            # The learned actor updates targets at 50 Hz, while its high-gain
            # PD loop must close at the 500 Hz physics rate.  Holding one
            # torque sample for the full 20 ms control frame destabilizes the
            # G1 even at zero command; this is the same two-rate contract used
            # by the already-qualified shared-world runner.
            for controller in controllers:
                target = np.asarray(controller.output.actions, dtype=np.float64)
                kp = np.asarray(controller.output.kps, dtype=np.float64)
                kd = np.asarray(controller.output.kds, dtype=np.float64)
                q = np.asarray(data.qpos[controller.joint_qpos], dtype=np.float64)
                dq = np.asarray(data.qvel[controller.joint_qvel], dtype=np.float64)
                raw_torque = kp * (target - q) - kd * dq
                projected_torque = _project_joint_safe_torque(
                    joint_position=q,
                    joint_velocity=dq,
                    commanded_torque=raw_torque,
                    joint_ranges=np.asarray(model.jnt_range[controller.joint_ids]),
                    limited=model.jnt_limited[controller.joint_ids].astype(bool),
                )
                torque = np.clip(projected_torque, -guarded_limits, guarded_limits)
                data.ctrl[controller.actuators] = torque
                controller.torque_limit_violation = bool(
                    controller.torque_limit_violation or np.any(np.abs(torque) > hard_limits)
                )
            apply_g1_compliant_goal_net_force(
                data,
                ball_body_id=ball_body,
                ball_qpos=ball_qpos,
                ball_qvel=ball_qvel,
                spec=goal,
                capture_depth_m=max(0.20, 0.80 * goal.depth_m),
                stiffness_n_m=180.0,
                damping_n_s_m=10.0,
                state=net_state,
            )
            mujoco.mj_step(model, data)
            robot_contact_count += _robot_robot_contacts(model, data, controllers)
        peak_ball_speed = max(
            peak_ball_speed,
            float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3])),
        )
        finite = bool(
            np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
            and np.all(np.isfinite(data.ctrl))
        )
        for controller in controllers:
            q = np.asarray(data.qpos[controller.joint_qpos], dtype=np.float64)
            limited = model.jnt_limited[controller.joint_ids].astype(bool)
            ranges = model.jnt_range[controller.joint_ids]
            controller.joint_limit_violation = bool(
                controller.joint_limit_violation
                or np.any(q[limited] < ranges[limited, 0] - 1.0e-5)
                or np.any(q[limited] > ranges[limited, 1] + 1.0e-5)
            )
            pelvis = np.asarray(
                data.qpos[controller.qpos_base : controller.qpos_base + 7], dtype=np.float64
            )
            roll, pitch = _roll_pitch(pelvis[3:7])
            controller.minimum_pelvis_height_m = min(
                controller.minimum_pelvis_height_m, float(pelvis[2])
            )
            controller.maximum_tilt_rad = max(controller.maximum_tilt_rad, abs(roll), abs(pitch))
            _append_player_trace(trace, controller=controller, data=data)
        trace["time"].append(float(data.time))
        trace["ball_pose"].append(data.qpos[ball_qpos : ball_qpos + 7].copy())
        trace["ball_velocity"].append(data.qvel[ball_qvel : ball_qvel + 6].copy())
        trace["coordination_frame_index"].append(current_coordination_index)
        if not finite:
            break

    trajectory = {name: np.asarray(values) for name, values in trace.items()}
    qualities = tuple(
        AgentWorldQuality(
            agent_id=controller.cell.agent_id,
            role=controller.cell.self_model.primary_role,
            active_fraction=controller.active_frames / max(1, len(trace["time"])),
            displacement_m=float(
                np.linalg.norm(
                    np.asarray(
                        data.qpos[controller.qpos_base : controller.qpos_base + 2],
                        dtype=np.float64,
                    )
                    - initial_positions[controller.cell.agent_id][:2]
                )
            ),
            distinct_intent_count=len(controller.seen_intents or ()),
            intent_switch_count=controller.intent_switch_count,
            decision_count=controller.decision_count,
            pass_intent_count=controller.pass_intent_count,
            shot_intent_count=controller.shot_intent_count,
            save_intent_count=controller.save_intent_count,
            distribution_intent_count=controller.distribution_intent_count,
            minimum_pelvis_height_m=controller.minimum_pelvis_height_m,
            maximum_tilt_rad=controller.maximum_tilt_rad,
            joint_limit_violation=controller.joint_limit_violation,
            torque_limit_violation=controller.torque_limit_violation,
            required_minimum_pelvis_height_m=active.minimum_pelvis_height_m,
            allowed_maximum_tilt_rad=active.maximum_tilt_rad,
        )
        for controller in controllers
    )
    result = IndependentTeamWorldResult(
        scenario_hash=scenario.scenario_hash,
        roster_hash=roster.roster_hash,
        config_hash=active.config_hash,
        trajectory_hash=trajectory_digest(trajectory),
        player_count=len(controllers),
        red_player_count=sum(cell.self_model.team_id == "red" for cell in cells),
        blue_player_count=sum(cell.self_model.team_id == "blue" for cell in cells),
        decision_frame_count=len(coordination_hashes),
        coordination_frame_hashes=tuple(coordination_hashes),
        qualities=qualities,
        pass_handshake_count=pass_handshake_count,
        pass_intent_count=intent_counts[TacticalIntent.PASS],
        shot_intent_count=intent_counts[TacticalIntent.SHOOT],
        save_intent_count=intent_counts[TacticalIntent.SAVE],
        distribution_intent_count=intent_counts[TacticalIntent.DISTRIBUTE],
        finite_state=finite,
        robot_robot_contact_count=robot_contact_count,
        rolling_distance_m=float(
            np.linalg.norm(
                np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
                - initial_ball[:2]
            )
        ),
        peak_ball_speed_mps=peak_ball_speed,
    )
    return result, trajectory


def _make_player_controller(
    *,
    model: Any,
    data: Any,
    spec: G1PitchPlayerSpec,
    cell: RosclawSoccerAgentCell,
    pelvis_height: float,
    state_type: Any,
    output_type: Any,
    loco_type: Any,
) -> _PlayerController:
    import mujoco

    prefix = spec.body_prefix
    free_joint = _id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + "floating_base_joint")
    qpos_base = int(model.jnt_qposadr[free_joint])
    qvel_base = int(model.jnt_dofadr[free_joint])
    joint_ids = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_JOINT, prefix + name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    joint_qpos = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int64)
    joint_qvel = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int64)
    actuators = np.asarray(
        [_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, prefix + name) for name in G1_DDS_JOINT_NAMES],
        dtype=np.int64,
    )
    state = state_type(29)
    output = output_type(29)
    with contextlib.redirect_stdout(io.StringIO()):
        policy = loco_type(state, output)
        policy.enter()
    data.qpos[qpos_base : qpos_base + 3] = (
        spec.origin_m[0],
        spec.origin_m[1],
        pelvis_height,
    )
    half_yaw = 0.5 * spec.yaw_rad
    data.qpos[qpos_base + 3 : qpos_base + 7] = (
        math.cos(half_yaw),
        0.0,
        0.0,
        math.sin(half_yaw),
    )
    data.qpos[joint_qpos] = np.asarray(policy.default_angles_reorder, dtype=np.float64)
    pelvis_body = _id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "pelvis")
    return _PlayerController(
        spec=spec,
        cell=cell,
        qpos_base=qpos_base,
        qvel_base=qvel_base,
        joint_ids=joint_ids,
        joint_qpos=joint_qpos,
        joint_qvel=joint_qvel,
        actuators=actuators,
        pelvis_body=pelvis_body,
        torso_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "torso_link"),
        left_ankle_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "left_ankle_roll_link"),
        right_ankle_body=_id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + "right_ankle_roll_link"),
        robot_geoms=_robot_geom_ids(model, pelvis_body),
        state=state,
        output=output,
        policy=policy,
    )


def _fill_locomotion_state(
    controller: _PlayerController, data: Any, ball_body: int, ball_qvel: int
) -> None:
    state = controller.state
    state.q = data.qpos[controller.joint_qpos].copy()
    state.dq = data.qvel[controller.joint_qvel].copy()
    state.tau_est = data.ctrl[controller.actuators].copy()
    state.root_lin_vel_b = data.qvel[controller.qvel_base : controller.qvel_base + 3].copy()
    state.root_ang_vel_b = data.qvel[controller.qvel_base + 3 : controller.qvel_base + 6].copy()
    state.torso_pos_w = data.xpos[controller.torso_body].copy()
    state.torso_quat_w = data.xquat[controller.torso_body].copy()
    state.pelvis_pos_w = data.qpos[controller.qpos_base : controller.qpos_base + 3].copy()
    state.pelvis_quat_w = data.qpos[controller.qpos_base + 3 : controller.qpos_base + 7].copy()
    state.ball_pos_w = data.xpos[ball_body].copy()
    state.ball_vel_w = data.qvel[ball_qvel : ball_qvel + 3].copy()
    state.ball_valid = True
    state.gravity_ori = _gravity_orientation(state.pelvis_quat_w)
    state.ang_vel = state.root_ang_vel_b.copy()


def _physical_state(controller: _PlayerController, data: Any) -> AgentPhysicalState:
    pose = np.asarray(data.qpos[controller.qpos_base : controller.qpos_base + 7], dtype=np.float64)
    velocity = _rotate_z(
        np.asarray(data.qvel[controller.qvel_base : controller.qvel_base + 3]),
        controller.spec.yaw_rad,
    )
    roll, pitch = _roll_pitch(pose[3:7])
    tilt = max(abs(roll), abs(pitch))
    return AgentPhysicalState(
        agent_id=controller.cell.agent_id,
        position_m=(float(pose[0]), float(pose[1]), float(pose[2])),
        velocity_mps=(float(velocity[0]), float(velocity[1]), float(velocity[2])),
        pelvis_height_m=float(pose[2]),
        tilt_rad=tilt,
        stable=bool(pose[2] >= 0.55 and tilt <= 0.80),
    )


def _agent_observation(
    *,
    controller: _PlayerController,
    roster: TeamRoleRoster,
    state_by_id: dict[str, AgentPhysicalState],
    data: Any,
    ball_qpos: int,
    ball_qvel: int,
    goal: G1TrainingGoalSpec,
    left_goal_plane_x_m: float,
    possession_agent_id: str | None,
) -> AgentCellObservation:
    model = roster.agent(controller.cell.agent_id)
    own_goal_x = goal.plane_x_m if model.team_id == "blue" else left_goal_plane_x_m
    opponent_goal_x = goal.plane_x_m if model.team_id == "red" else left_goal_plane_x_m
    ball_position = data.qpos[ball_qpos : ball_qpos + 3]
    ball_velocity = data.qvel[ball_qvel : ball_qvel + 3]
    return AgentCellObservation(
        observer_agent_id=model.agent_id,
        time_sec=float(data.time),
        ball_position_m=(
            float(ball_position[0]),
            float(ball_position[1]),
            float(ball_position[2]),
        ),
        ball_velocity_mps=(
            float(ball_velocity[0]),
            float(ball_velocity[1]),
            float(ball_velocity[2]),
        ),
        own_goal_m=(own_goal_x, 0.0, 0.0),
        opponent_goal_m=(opponent_goal_x, 0.0, 0.0),
        possession_agent_id=possession_agent_id,
        self_state=state_by_id[model.agent_id],
        teammate_states=tuple(state_by_id[agent_id] for agent_id in model.teammate_ids),
        opponent_states=tuple(state_by_id[agent_id] for agent_id in model.opponent_ids),
    )


def _infer_possession(
    *,
    controllers: tuple[_PlayerController, ...],
    data: Any,
    ball_position: NDArray[Any],
    maximum_distance_m: float,
) -> str | None:
    distances = []
    for controller in controllers:
        feet = (
            np.asarray(data.xpos[controller.left_ankle_body], dtype=np.float64),
            np.asarray(data.xpos[controller.right_ankle_body], dtype=np.float64),
        )
        distance = min(float(np.linalg.norm(foot - ball_position)) for foot in feet)
        distances.append((distance, controller.cell.agent_id))
    distance, agent_id = min(distances)
    return agent_id if distance <= maximum_distance_m else None


def _movement_command(
    *,
    controller: _PlayerController,
    decision: AgentCellDecision,
    positions: dict[str, NDArray[np.float64]],
    data: Any,
    ball_qpos: int,
    config: IndependentTeamWorldConfig,
) -> NDArray[np.float64]:
    current = positions[controller.cell.agent_id]
    target = np.asarray(decision.target_position_m[:2], dtype=np.float64)
    ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 2], dtype=np.float64)
    if decision.intent in {
        TacticalIntent.PASS,
        TacticalIntent.SHOOT,
        TacticalIntent.DISTRIBUTE,
    }:
        destination = np.asarray(decision.target_position_m[:2], dtype=np.float64)
        direction = destination - ball
        direction /= max(float(np.linalg.norm(direction)), 1.0e-9)
        target = ball - 0.72 * direction
    error = target - current
    command = np.zeros(3, dtype=np.float64)
    command[:2] = config.position_gain * error
    speed_limit = (
        config.goalkeeper_maximum_speed_mps
        if controller.cell.self_model.primary_role is MatchRole.GOALKEEPER
        else config.maximum_speed_mps
    )
    speed = float(np.linalg.norm(command[:2]))
    if speed > speed_limit:
        command[:2] *= speed_limit / speed
    if float(np.linalg.norm(error)) <= config.arrival_radius_m:
        command[:2] = 0.0
    for other_id, other in positions.items():
        if other_id == controller.cell.agent_id:
            continue
        delta = current - other
        separation = float(np.linalg.norm(delta))
        if 1.0e-9 < separation < config.minimum_player_separation_m:
            correction = min(
                config.maximum_collision_correction_mps,
                config.collision_avoidance_gain * (config.minimum_player_separation_m - separation),
            )
            command[:2] += correction * delta / separation
    previous = (
        np.zeros(3, dtype=np.float64)
        if controller.last_world_command is None
        else controller.last_world_command
    )
    delta = command - previous
    maximum_delta = config.maximum_acceleration_mps2 * _CONTROL_DT
    delta_norm = float(np.linalg.norm(delta[:2]))
    if delta_norm > maximum_delta:
        command[:2] = previous[:2] + delta[:2] * maximum_delta / delta_norm
    speed = float(np.linalg.norm(command[:2]))
    if speed > speed_limit:
        command[:2] *= speed_limit / speed
    return command


def _normalized_locomotion_command(policy: Any, physical: NDArray[Any]) -> NDArray[np.float64]:
    command = np.asarray(physical, dtype=np.float64)
    ranges = np.asarray((policy.range_velx, policy.range_vely, policy.range_velz), dtype=np.float64)
    command = np.clip(command, ranges[:, 0], ranges[:, 1])
    widths = ranges[:, 1] - ranges[:, 0]
    if command.shape != (3,) or np.any(widths <= 0.0):
        raise ValueError("locomotion command contract changed")
    return np.asarray(-1.0 + 2.0 * (command - ranges[:, 0]) / widths, dtype=np.float64)


def _run_locomotion(controller: _PlayerController, *, mirror: bool) -> None:
    """Reuse the qualified sagittal mirror for the actor's weak -y half-space."""

    if not mirror:
        with contextlib.redirect_stdout(io.StringIO()):
            controller.policy.run()
        return
    state = controller.state
    original = {
        name: np.asarray(getattr(state, name), dtype=np.float64).copy()
        for name in ("q", "dq", "gravity_ori", "ang_vel", "vel_cmd")
    }
    try:
        state.q = mirror_g1_joint_positions(original["q"])
        state.dq = mirror_g1_joint_positions(original["dq"])
        gravity = original["gravity_ori"].copy()
        gravity[1] *= -1.0
        state.gravity_ori = gravity
        angular = original["ang_vel"].copy()
        angular[(0, 2),] *= -1.0
        state.ang_vel = angular
        command = original["vel_cmd"].copy()
        command[1] *= -1.0
        state.vel_cmd = command
        with contextlib.redirect_stdout(io.StringIO()):
            controller.policy.run()
        controller.output.actions = mirror_g1_joint_positions(
            np.asarray(controller.output.actions, dtype=np.float64)
        )
        controller.output.kps = mirror_g1_joint_gains(
            np.asarray(controller.output.kps, dtype=np.float64)
        )
        controller.output.kds = mirror_g1_joint_gains(
            np.asarray(controller.output.kds, dtype=np.float64)
        )
    finally:
        for name, value in original.items():
            setattr(state, name, value)


def _append_player_trace(
    trace: dict[str, list[Any]], *, controller: _PlayerController, data: Any
) -> None:
    decision = controller.decision
    command = controller.last_world_command
    if decision is None or command is None:
        raise RuntimeError("cannot trace an undecided independent agent")
    key = _agent_key(controller.cell.agent_id)
    trace[f"{key}_pelvis_pose"].append(
        data.qpos[controller.qpos_base : controller.qpos_base + 7].copy()
    )
    trace[f"{key}_joint_position"].append(data.qpos[controller.joint_qpos].copy())
    trace[f"{key}_joint_velocity"].append(data.qvel[controller.joint_qvel].copy())
    trace[f"{key}_joint_torque"].append(data.ctrl[controller.actuators].copy())
    trace[f"{key}_left_foot_position"].append(data.xpos[controller.left_ankle_body].copy())
    trace[f"{key}_right_foot_position"].append(data.xpos[controller.right_ankle_body].copy())
    trace[f"{key}_intent_code"].append(_INTENT_CODES[decision.intent])
    trace[f"{key}_target_position"].append(decision.target_position_m)
    trace[f"{key}_world_command"].append(command.copy())
    trace[f"{key}_movement_active"].append(float(np.linalg.norm(command[:2])) >= 0.04)


def _robot_robot_contacts(model: Any, data: Any, controllers: tuple[_PlayerController, ...]) -> int:
    owner: dict[int, str] = {}
    for controller in controllers:
        for geom in controller.robot_geoms:
            owner[geom] = controller.cell.agent_id
    count = 0
    for index in range(int(data.ncon)):
        contact = data.contact[index]
        first = owner.get(int(contact.geom1))
        second = owner.get(int(contact.geom2))
        count += int(first is not None and second is not None and first != second)
    return count


def _robot_geom_ids(model: Any, root_body: int) -> frozenset[int]:
    values: set[int] = set()
    for geom in range(int(model.ngeom)):
        body = int(model.geom_bodyid[geom])
        while body > 0 and body != root_body:
            body = int(model.body_parentid[body])
        if body == root_body:
            values.add(geom)
    return frozenset(values)


def _gravity_orientation(quaternion: NDArray[Any]) -> NDArray[np.float64]:
    value = np.asarray(quaternion, dtype=np.float64)
    if value.shape != (4,) or not np.all(np.isfinite(value)):
        raise ValueError("pelvis quaternion is invalid")
    qw, qx, qy, qz = map(float, value)
    return np.asarray(
        (
            2.0 * (-qz * qx + qw * qy),
            -2.0 * (qz * qy + qw * qx),
            1.0 - 2.0 * (qw * qw + qz * qz),
        ),
        dtype=np.float64,
    )


def _project_joint_safe_torque(
    *,
    joint_position: NDArray[Any],
    joint_velocity: NDArray[Any],
    commanded_torque: NDArray[Any],
    joint_ranges: NDArray[Any],
    limited: NDArray[Any],
    margin_rad: float = 0.04,
    prediction_horizon_sec: float = 0.08,
    boundary_kp: float = 80.0,
    boundary_kd: float = 6.0,
) -> NDArray[np.float64]:
    """Project outward torque away from predicted G1 joint-limit threats."""

    q = np.asarray(joint_position, dtype=np.float64)
    dq = np.asarray(joint_velocity, dtype=np.float64)
    torque = np.asarray(commanded_torque, dtype=np.float64)
    ranges = np.asarray(joint_ranges, dtype=np.float64)
    constrained = np.asarray(limited, dtype=bool)
    if (
        q.shape != (29,)
        or dq.shape != (29,)
        or torque.shape != (29,)
        or ranges.shape != (29, 2)
        or constrained.shape != (29,)
        or not all(np.all(np.isfinite(value)) for value in (q, dq, torque, ranges))
    ):
        raise ValueError("joint guard requires finite 29-DoF G1 state")
    projected = torque.copy()
    predicted = q + prediction_horizon_sec * dq
    lower = ranges[:, 0] + margin_rad
    upper = ranges[:, 1] - margin_rad
    lower_threat = constrained & (predicted < lower)
    upper_threat = constrained & (predicted > upper)
    lower_brake = boundary_kp * (lower - q) - boundary_kd * dq
    upper_brake = boundary_kp * (upper - q) - boundary_kd * dq
    projected[lower_threat] = np.maximum(projected[lower_threat], lower_brake[lower_threat])
    projected[upper_threat] = np.minimum(projected[upper_threat], upper_brake[upper_threat])
    return projected


def _roll_pitch(quaternion: NDArray[Any]) -> tuple[float, float]:
    value = np.asarray(quaternion, dtype=np.float64)
    value /= max(float(np.linalg.norm(value)), 1.0e-12)
    w, x, y, z = map(float, value)
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = math.asin(float(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    return roll, pitch


def _rotate_z(vector: NDArray[Any], yaw: float) -> NDArray[np.float64]:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,):
        raise ValueError("planar locomotion vector must be xyz")
    cosine = math.cos(yaw)
    sine = math.sin(yaw)
    return np.asarray(
        (
            cosine * value[0] - sine * value[1],
            sine * value[0] + cosine * value[1],
            value[2],
        ),
        dtype=np.float64,
    )


def _agent_key(agent_id: str) -> str:
    return agent_id.replace(".", "_").replace(":", "_").replace("-", "_")


def _id(model: Any, object_type: Any, name: str) -> int:
    import mujoco

    value = int(mujoco.mj_name2id(model, object_type, name))
    if value < 0:
        raise ValueError(f"multi-player G1 model is missing {name}")
    return value


__all__ = [
    "AgentWorldQuality",
    "IndependentTeamWorldConfig",
    "IndependentTeamWorldResult",
    "IndependentTeamWorldScenario",
    "simulate_independent_team_world",
]
