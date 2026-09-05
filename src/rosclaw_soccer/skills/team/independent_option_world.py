"""Physical contact option execution inside the independent six-G1 arena.

One current autonomous commitment obtains an exclusive, content-bound ball
lease and is executed by a frozen whole-body contact policy.  The other five
ROSClaw cells continue to observe, decide, coordinate, and locomote in the same
MuJoCo clock.  The runner never writes robot roots or ball state after the
initial condition and exposes no ROS, DDS, vendor, or hardware path.
"""

from __future__ import annotations

import contextlib
import importlib
import io
import math
import re
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.independent_agent_cell import (
    RosclawSoccerAgentCell,
    build_team_coordination_frame,
)
from rosclaw_soccer.growth.physical_option_router import (
    PhysicalOptionOutcome,
    PhysicalOptionRequest,
    PhysicalOptionTerminal,
    PhysicalSoccerOption,
    build_physical_option_request,
)
from rosclaw_soccer.growth.role_self_model import MatchRole, TeamRoleRoster
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.providers.g1.mujoco_primitives import (
    adapt_shot_target,
    load_robonaldo,
)
from rosclaw_soccer.sim.contracts import (
    G1_HARD_TORQUE_LIMITS,
    ShotParameters,
    hash_bytes,
    hash_json,
)
from rosclaw_soccer.skills.team import independent_team_world as team_world
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
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")


@dataclass(frozen=True)
class IndependentOptionWorldConfig:
    simulation_duration_sec: float = 9.0
    decision_period_sec: float = 0.10
    option_timeout_sec: float = 7.0
    post_policy_frame: int = 300
    post_policy_blend_frames: int = 50
    minimum_pelvis_height_m: float = 0.55
    maximum_tilt_rad: float = 0.80
    locomotion: team_world.IndependentTeamWorldConfig = team_world.IndependentTeamWorldConfig(
        simulation_duration_sec=9.0
    )
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.independent_option_world_config.v1"

    def __post_init__(self) -> None:
        if (
            not 7.0 <= self.simulation_duration_sec <= 15.0
            or not 0.08 <= self.decision_period_sec <= 0.20
            or not 5.0 <= self.option_timeout_sec <= 10.0
            or not 270 <= self.post_policy_frame <= 360
            or not 20 <= self.post_policy_blend_frames <= 100
            or not 0.45 <= self.minimum_pelvis_height_m <= 0.70
            or not 0.45 <= self.maximum_tilt_rad <= 1.00
            or self.locomotion.simulation_duration_sec != self.simulation_duration_sec
            or self.locomotion.decision_period_sec != self.decision_period_sec
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("independent physical option config violates its SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class IndependentOptionScenario:
    scenario_id: str
    option_agent_id: str
    expected_option: PhysicalSoccerOption
    ball_initial_position_m: tuple[float, float, float]
    ball_initial_velocity_mps: tuple[float, float, float]
    preferred_target_m: tuple[float, float, float]
    parameters: ShotParameters
    seed: int
    schema_version: str = "rosclaw_soccer.independent_option_scenario.v1"

    def __post_init__(self) -> None:
        values = (
            *self.ball_initial_position_m,
            *self.ball_initial_velocity_mps,
            *self.preferred_target_m,
        )
        if (
            not self.scenario_id.startswith("s200.")
            or not _IDENTIFIER.fullmatch(self.option_agent_id)
            or self.expected_option not in {PhysicalSoccerOption.PASS, PhysicalSoccerOption.SHOOT}
            or any(not math.isfinite(value) for value in values)
            or self.ball_initial_position_m[2] <= 0.0
            or isinstance(self.seed, bool)
            or not 0 <= self.seed <= 2**32 - 1
        ):
            raise ValueError("independent physical option scenario is invalid")

    @property
    def scenario_hash(self) -> str:
        value = asdict(self)
        value["expected_option"] = self.expected_option.value
        return str(hash_json(value))


@dataclass(frozen=True)
class IndependentOptionWorldResult:
    scenario_hash: str
    roster_hash: str
    config_hash: str
    trajectory_hash: str
    request: PhysicalOptionRequest
    outcome: PhysicalOptionOutcome
    player_count: int
    decision_frame_count: int
    coordination_frame_hashes: tuple[str, ...]
    all_cells_decided_each_frame: bool
    all_cells_physically_present: bool
    ball_displacement_m: float
    final_ball_position_m: tuple[float, float, float]
    activation_ceiling: str = "SIM_ONLY"
    physics_authority: str = "CPU_MUJOCO"
    hardware_command_sent: bool = False
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.independent_option_world_result.v1"

    def __post_init__(self) -> None:
        hashes = (
            self.scenario_hash,
            self.roster_hash,
            self.config_hash,
            self.trajectory_hash,
        )
        if (
            any(not _HASH.fullmatch(value) for value in hashes)
            or self.request.request_hash != self.outcome.request_hash
            or self.player_count < 6
            or self.decision_frame_count != len(self.coordination_frame_hashes)
            or any(not _HASH.fullmatch(value) for value in self.coordination_frame_hashes)
            or not isinstance(self.all_cells_decided_each_frame, bool)
            or not isinstance(self.all_cells_physically_present, bool)
            or not math.isfinite(self.ball_displacement_m)
            or self.ball_displacement_m < 0.0
            or len(self.final_ball_position_m) != 3
            or any(not math.isfinite(value) for value in self.final_ball_position_m)
            or self.activation_ceiling != "SIM_ONLY"
            or self.physics_authority != "CPU_MUJOCO"
            or self.hardware_command_sent
            or self.pixels_used_for_scoring
        ):
            raise ValueError("independent physical option result contract is invalid")

    @property
    def passed(self) -> bool:
        return bool(
            self.player_count >= 6
            and self.decision_frame_count >= 10
            and self.all_cells_decided_each_frame
            and self.all_cells_physically_present
            and self.request.option in {PhysicalSoccerOption.PASS, PhysicalSoccerOption.SHOOT}
            and self.outcome.contact_time_sec is not None
            and self.outcome.contact_time_sec <= self.request.expires_at_sec
            and self.outcome.physical_contact_success
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
            "request": self.request.to_dict(),
            "request_hash": self.request.request_hash,
            "outcome": self.outcome.to_dict(),
            "player_count": self.player_count,
            "decision_frame_count": self.decision_frame_count,
            "coordination_frame_hashes": list(self.coordination_frame_hashes),
            "all_cells_decided_each_frame": self.all_cells_decided_each_frame,
            "all_cells_physically_present": self.all_cells_physically_present,
            "ball_displacement_m": self.ball_displacement_m,
            "final_ball_position_m": list(self.final_ball_position_m),
            "passed": self.passed,
            "activation_ceiling": self.activation_ceiling,
            "physics_authority": self.physics_authority,
            "hardware_command_sent": self.hardware_command_sent,
            "pixels_used_for_scoring": self.pixels_used_for_scoring,
        }


def physical_option_policy_hash(asset_root: Path) -> str:
    """Bind the exact frozen policy, motion, config, and adapter being executed."""

    root = asset_root.expanduser().resolve()
    relatives = (
        Path("policy/robonaldo/model/policy-obs-aic.onnx"),
        _MOTION_REL,
        Path("policy/robonaldo/config/freekick.yaml"),
        Path("policy/robonaldo/FreeKick.py"),
    )
    paths = tuple(root / relative for relative in relatives)
    if any(not path.is_file() for path in paths):
        raise ValueError("qualified asset root lacks the physical option policy")
    return str(
        hash_json(
            {
                "policy": "robonaldo.freekick.frozen",
                "artifacts": {
                    str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in paths
                },
            }
        )
    )


def simulate_independent_physical_option(
    *,
    asset_root: Path,
    roster: TeamRoleRoster,
    cells: tuple[RosclawSoccerAgentCell, ...],
    players: tuple[G1PitchPlayerSpec, ...],
    scenario: IndependentOptionScenario,
    goal: G1TrainingGoalSpec,
    config: IndependentOptionWorldConfig | None = None,
) -> tuple[IndependentOptionWorldResult, dict[str, NDArray[Any]]]:
    """Execute one causally authorized contact option while all six cells run."""

    active = config or IndependentOptionWorldConfig()
    cell_by_id = {cell.agent_id: cell for cell in cells}
    player_by_id = {player.agent_id: player for player in players}
    roster_ids = {agent.agent_id for agent in roster.agents}
    prepared_spec = player_by_id.get(scenario.option_agent_id)
    if (
        len(roster_ids) < 6
        or set(cell_by_id) != roster_ids
        or set(player_by_id) != roster_ids
        or prepared_spec is None
        or prepared_spec.body_prefix != ""
        or prepared_spec.origin_m != (0.0, 0.0, 0.0)
        or abs(prepared_spec.yaw_rad) > 1.0e-12
    ):
        raise ValueError("physical option requires six bound cells and one canonical option actor")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()

    import mujoco

    model = build_g1_multi_player_stadium_model(asset_root, players=players, spec=goal)
    model.opt.timestep = _PHYSICS_DT
    data = mujoco.MjData(model)
    state_type, output_type, kick_type, mujoco_to_isaac = load_robonaldo(qualification.asset_root)
    loco_type = importlib.import_module("policy.loco_mode.LocoMode").LocoMode
    with np.load(qualification.asset_root / _MOTION_REL, allow_pickle=False) as motion:
        initial_position = np.asarray(motion["body_pos_w"])[0, 0].astype(np.float64)
        initial_quaternion = np.asarray(motion["body_quat_w"])[0, 0].astype(np.float64)
        initial_joints = np.asarray(motion["joint_pos"])[0][mujoco_to_isaac].astype(np.float64)
        pelvis_height = float(initial_position[2])
    controllers = tuple(
        team_world._make_player_controller(
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
    prepared = next(
        value for value in controllers if value.cell.agent_id == scenario.option_agent_id
    )
    data.qpos[prepared.qpos_base : prepared.qpos_base + 3] = initial_position
    data.qpos[prepared.qpos_base + 3 : prepared.qpos_base + 7] = initial_quaternion
    data.qpos[prepared.joint_qpos] = initial_joints
    ball_body = team_world._id(model, mujoco.mjtObj.mjOBJ_BODY, "ball")
    ball_geom = team_world._id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    ball_joint = team_world._id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free")
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    data.qpos[ball_qpos : ball_qpos + 3] = scenario.ball_initial_position_m
    data.qpos[ball_qpos + 3 : ball_qpos + 7] = (1.0, 0.0, 0.0, 0.0)
    data.qvel[ball_qvel : ball_qvel + 3] = scenario.ball_initial_velocity_mps
    data.qvel[ball_qvel + 3 : ball_qvel + 6] = 0.0
    mujoco.mj_forward(model, data)
    for controller in controllers:
        team_world._fill_locomotion_state(controller, data, ball_body, ball_qvel)
    kick_output = output_type(29)
    with contextlib.redirect_stdout(io.StringIO()):
        kick_policy = kick_type(prepared.state, kick_output)
    kick_policy.target_pos_w = np.asarray(scenario.preferred_target_m, dtype=np.float32)
    with contextlib.redirect_stdout(io.StringIO()):
        kick_policy.enter()

    total_frames = int(round(active.simulation_duration_sec / _CONTROL_DT))
    decision_stride = max(1, int(round(active.decision_period_sec / _CONTROL_DT)))
    hard_limits = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    guarded_limits = 0.85 * hard_limits
    left_foot_geoms = team_world._robot_geom_ids(model, prepared.left_ankle_body)
    right_foot_geoms = team_world._robot_geom_ids(model, prepared.right_ankle_body)
    target_controller = (
        None
        if scenario.expected_option is not PhysicalSoccerOption.PASS
        else next(
            value
            for value in controllers
            if value.cell.agent_id in prepared.cell.self_model.teammate_ids
            and value.cell.self_model.primary_role is MatchRole.FINISHER
        )
    )
    initial_ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64).copy()
    coordination_hashes: list[str] = []
    request: PhysicalOptionRequest | None = None
    current_coordination_index = -1
    finite = True
    robot_contact_count = 0
    contact_time: float | None = None
    contact_link: str | None = None
    pre_contact_peak_speed = float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3]))
    post_contact_peak_speed = 0.0
    target_distance = math.inf
    target_agent_contact = False
    goalkeeper_contact = False
    goal_crossed = False
    phase_code = 0
    handoff_start_policy_frame: int | None = None
    net_state = G1CompliantGoalNetState()
    trace: dict[str, list[Any]] = {
        "time": [],
        "ball_pose": [],
        "ball_velocity": [],
        "coordination_frame_index": [],
        "option_phase": [],
        "option_contact": [],
    }
    for controller in controllers:
        key = team_world._agent_key(controller.cell.agent_id)
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

    for frame in range(total_frames):
        for controller in controllers:
            team_world._fill_locomotion_state(controller, data, ball_body, ball_qvel)
        if frame % decision_stride == 0:
            physical_states = tuple(
                team_world._physical_state(controller, data) for controller in controllers
            )
            state_by_id = {state.agent_id: state for state in physical_states}
            possession = team_world._infer_possession(
                controllers=controllers,
                data=data,
                ball_position=np.asarray(data.qpos[ball_qpos : ball_qpos + 3]),
                maximum_distance_m=active.locomotion.possession_radius_m,
            )
            observations = tuple(
                team_world._agent_observation(
                    controller=controller,
                    roster=roster,
                    state_by_id=state_by_id,
                    data=data,
                    ball_qpos=ball_qpos,
                    ball_qvel=ball_qvel,
                    goal=goal,
                    left_goal_plane_x_m=active.locomotion.left_goal_plane_x_m,
                    possession_agent_id=possession,
                )
                for controller in controllers
            )
            # Finishing is aimed at the declared scoring target, not the goal
            # centre.  This target is therefore inside the decision hash and
            # cannot be swapped by the physical runner after authorization.
            if scenario.expected_option is PhysicalSoccerOption.SHOOT:
                observations = tuple(
                    value
                    if value.observer_agent_id != scenario.option_agent_id
                    else replace(value, opponent_goal_m=scenario.preferred_target_m)
                    for value in observations
                )
            decisions = tuple(
                cell_by_id[value.observer_agent_id].decide(value) for value in observations
            )
            coordination = build_team_coordination_frame(
                roster=roster,
                cells=cells,
                observations=observations,
                decisions=decisions,
                frame_index=len(coordination_hashes),
            )
            coordination_hashes.append(coordination.frame_hash)
            current_coordination_index += 1
            for decision in decisions:
                controller = next(
                    item for item in controllers if item.cell.agent_id == decision.agent_id
                )
                controller.decision = decision
                controller.decision_count += 1
            if request is None:
                prepared_decision = next(
                    value for value in decisions if value.agent_id == scenario.option_agent_id
                )
                request = build_physical_option_request(
                    cell=prepared.cell,
                    decision=prepared_decision,
                    coordination=coordination,
                    option_policy_hash=physical_option_policy_hash(asset_root),
                    timeout_sec=active.option_timeout_sec,
                )
                if request.option is not scenario.expected_option:
                    raise ValueError("autonomous option does not match the scenario contract")
                kick_policy.target_pos_w = np.asarray(request.target_position_m, dtype=np.float32)

        positions = {
            controller.cell.agent_id: np.asarray(
                data.qpos[controller.qpos_base : controller.qpos_base + 2], dtype=np.float64
            ).copy()
            for controller in controllers
        }
        if request is None:
            raise RuntimeError("physical option request was not authorized at frame zero")
        current_request = request
        for controller in controllers:
            if controller.decision is None:
                raise RuntimeError("independent physical option cell has no current decision")
            command = team_world._movement_command(
                controller=controller,
                decision=controller.decision,
                positions=positions,
                data=data,
                ball_qpos=ball_qpos,
                config=active.locomotion,
            )
            if controller is prepared:
                command[:] = 0.0
            local = team_world._rotate_z(command, -controller.spec.yaw_rad)
            controller.state.vel_cmd = team_world._normalized_locomotion_command(
                controller.policy, local
            )
            team_world._run_locomotion(controller, mirror=bool(local[1] < -1.0e-6))
            controller.last_world_command = command.copy()
        policy_frame = int(kick_policy.time_step)
        # A time-only handoff can interrupt a late contact and was observed to
        # topple the finisher.  Recovery is therefore event gated: after a
        # contact, the strike policy retains authority until the minimum
        # learned motion phase completes.  A missing contact is different: at
        # request expiry it loses authority and blends out immediately rather
        # than continuing to act outside its lease.
        handoff_due = bool(
            handoff_start_policy_frame is not None
            or (contact_time is not None and policy_frame >= active.post_policy_frame)
            or (contact_time is None and float(data.time) >= current_request.expires_at_sec)
        )
        if not handoff_due:
            with contextlib.redirect_stdout(io.StringIO()):
                kick_policy.run()
            kick_target = adapt_shot_target(
                target=np.asarray(kick_output.actions, dtype=np.float64),
                default=np.asarray(kick_policy.default_q_mj, dtype=np.float64),
                parameters=scenario.parameters,
                policy_frame=policy_frame,
            )
            option_target = kick_target
            option_kp = np.asarray(kick_output.kps, dtype=np.float64)
            option_kd = np.asarray(kick_output.kds, dtype=np.float64)
            phase_code = 1
        else:
            if handoff_start_policy_frame is None:
                handoff_start_policy_frame = policy_frame
            blend = min(
                1.0,
                (policy_frame - handoff_start_policy_frame + 1) / active.post_policy_blend_frames,
            )
            option_target = (1.0 - blend) * np.asarray(
                kick_output.actions, dtype=np.float64
            ) + blend * np.asarray(prepared.output.actions, dtype=np.float64)
            option_kp = (1.0 - blend) * np.asarray(
                kick_output.kps, dtype=np.float64
            ) + blend * np.asarray(prepared.output.kps, dtype=np.float64)
            option_kd = (1.0 - blend) * np.asarray(
                kick_output.kds, dtype=np.float64
            ) + blend * np.asarray(prepared.output.kds, dtype=np.float64)
            kick_policy.time_step += 1
            phase_code = 2 if blend < 1.0 else 3

        for _ in range(_SUBSTEPS):
            for controller in controllers:
                if controller is prepared:
                    target, kp, kd = option_target, option_kp, option_kd
                else:
                    target = np.asarray(controller.output.actions, dtype=np.float64)
                    kp = np.asarray(controller.output.kps, dtype=np.float64)
                    kd = np.asarray(controller.output.kds, dtype=np.float64)
                q = np.asarray(data.qpos[controller.joint_qpos], dtype=np.float64)
                dq = np.asarray(data.qvel[controller.joint_qvel], dtype=np.float64)
                raw_torque = kp * (target - q) - kd * dq
                projected = team_world._project_joint_safe_torque(
                    joint_position=q,
                    joint_velocity=dq,
                    commanded_torque=raw_torque,
                    joint_ranges=np.asarray(model.jnt_range[controller.joint_ids]),
                    limited=model.jnt_limited[controller.joint_ids].astype(bool),
                )
                data.ctrl[controller.actuators] = np.clip(
                    projected, -guarded_limits, guarded_limits
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
            robot_contact_count += team_world._robot_robot_contacts(model, data, controllers)
            for index in range(int(data.ncon)):
                contact = data.contact[index]
                pair = {int(contact.geom1), int(contact.geom2)}
                if ball_geom not in pair:
                    continue
                other = next(value for value in pair if value != ball_geom)
                if contact_time is None and other in left_foot_geoms | right_foot_geoms:
                    contact_time = float(data.time)
                    contact_link = "left_foot" if other in left_foot_geoms else "right_foot"
                if target_controller is not None and other in target_controller.robot_geoms:
                    target_agent_contact = True
                for controller in controllers:
                    if (
                        controller.cell.self_model.primary_role is MatchRole.GOALKEEPER
                        and other in controller.robot_geoms
                    ):
                        goalkeeper_contact = True
            speed = float(np.linalg.norm(data.qvel[ball_qvel : ball_qvel + 3]))
            if contact_time is None:
                pre_contact_peak_speed = max(pre_contact_peak_speed, speed)
            else:
                post_contact_peak_speed = max(post_contact_peak_speed, speed)
            ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
            if contact_time is not None:
                target_distance = min(
                    target_distance,
                    float(np.linalg.norm(ball - np.asarray(current_request.target_position_m))),
                )
                goal_crossed = bool(
                    goal_crossed
                    or (
                        ball[0] >= goal.plane_x_m
                        and abs(ball[1]) <= goal.width_m / 2.0
                        and goal.ball_radius_m <= ball[2] <= goal.height_m
                    )
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
            roll, pitch = team_world._roll_pitch(pelvis[3:7])
            controller.minimum_pelvis_height_m = min(
                controller.minimum_pelvis_height_m, float(pelvis[2])
            )
            controller.maximum_tilt_rad = max(controller.maximum_tilt_rad, abs(roll), abs(pitch))
            team_world._append_player_trace(trace, controller=controller, data=data)
        trace["time"].append(float(data.time))
        trace["ball_pose"].append(data.qpos[ball_qpos : ball_qpos + 7].copy())
        trace["ball_velocity"].append(data.qvel[ball_qvel : ball_qvel + 6].copy())
        trace["coordination_frame_index"].append(current_coordination_index)
        trace["option_phase"].append(phase_code)
        trace["option_contact"].append(contact_time is not None)
        if not finite:
            break

    if request is None:
        raise RuntimeError("physical option request was never authorized")
    minimum_height = min(value.minimum_pelvis_height_m for value in controllers)
    maximum_tilt = max(value.maximum_tilt_rad for value in controllers)
    joint_violation = any(value.joint_limit_violation for value in controllers)
    final_ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64)
    outcome = PhysicalOptionOutcome(
        request_hash=request.request_hash,
        terminal=(
            PhysicalOptionTerminal.COMPLETE
            if contact_time is not None
            else PhysicalOptionTerminal.ABORTED
        ),
        contact_observed=contact_time is not None,
        contact_time_sec=contact_time,
        contact_link=contact_link,
        pre_contact_ball_speed_mps=pre_contact_peak_speed,
        post_contact_peak_ball_speed_mps=post_contact_peak_speed,
        target_delivery_distance_m=None if math.isinf(target_distance) else target_distance,
        target_agent_contact_observed=target_agent_contact,
        goal_crossed=goal_crossed,
        goalkeeper_contact_observed=goalkeeper_contact,
        finite_state=finite,
        minimum_pelvis_height_m=minimum_height,
        maximum_tilt_rad=maximum_tilt,
        required_minimum_pelvis_height_m=active.minimum_pelvis_height_m,
        allowed_maximum_tilt_rad=active.maximum_tilt_rad,
        joint_limit_violation=joint_violation,
        torque_limit_violation=False,
        robot_robot_contact_count=robot_contact_count,
        option_started_from_current_commitment=True,
        recovery_handoff_completed=phase_code == 3,
    )
    trajectory = {name: np.asarray(values) for name, values in trace.items()}
    result = IndependentOptionWorldResult(
        scenario_hash=scenario.scenario_hash,
        roster_hash=roster.roster_hash,
        config_hash=active.config_hash,
        trajectory_hash=trajectory_digest(trajectory),
        request=request,
        outcome=outcome,
        player_count=len(controllers),
        decision_frame_count=len(coordination_hashes),
        coordination_frame_hashes=tuple(coordination_hashes),
        all_cells_decided_each_frame=all(
            value.decision_count == len(coordination_hashes) for value in controllers
        ),
        all_cells_physically_present=len(controllers) == 6 and model.nu == 6 * 29,
        ball_displacement_m=float(np.linalg.norm(final_ball[:2] - initial_ball[:2])),
        final_ball_position_m=(
            float(final_ball[0]),
            float(final_ball[1]),
            float(final_ball[2]),
        ),
    )
    return result, trajectory


__all__ = [
    "IndependentOptionScenario",
    "IndependentOptionWorldConfig",
    "IndependentOptionWorldResult",
    "physical_option_policy_hash",
    "simulate_independent_physical_option",
]
