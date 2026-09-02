"""CPU MuJoCo tactical plane for learning why to pass in a 2v1.

The environment deliberately sits above the G1 whole-body controller.  A
small physical effector represents a frozen, previously qualified kick/first-
touch option while the learner owns only the 10 Hz PASS/SHOOT decision.  Ball
motion, defender interceptions, reception and goal crossing are MuJoCo state,
not scripted outcome labels.  This module is SIM-only and has no ROS or
hardware transport.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.tactical_2v1 import (
    MatchedTacticalRollout,
    TacticalAction,
    TacticalRewardWeights,
    TwoVsOneDecisionEvidence,
    TwoVsOneState,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

_HASH = re.compile(r"^sha256:[0-9a-f]{64}$")
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_ACTIONS = (TacticalAction.PASS, TacticalAction.SHOOT)


def _require_hash(value: str, label: str) -> str:
    if not _HASH.fullmatch(value):
        raise ValueError(f"{label} must be a SHA-256 content commitment")
    return value


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _point_segment_distance(
    point: NDArray[np.float64], start: NDArray[np.float64], end: NDArray[np.float64]
) -> float:
    segment = end - start
    denominator = float(np.dot(segment, segment))
    if denominator <= 1.0e-12:
        return float(np.linalg.norm(point - start))
    fraction = float(np.clip(np.dot(point - start, segment) / denominator, 0.0, 1.0))
    return float(np.linalg.norm(point - (start + fraction * segment)))


@dataclass(frozen=True)
class FrozenTacticalSkillBundle:
    """Content identities of the low-level options hidden from the learner."""

    body_hash: str
    athlete_foundation_hash: str
    first_touch_actor_hash: str
    pass_skill_hash: str
    shoot_skill_hash: str
    schema_version: str = "rosclaw_soccer.frozen_tactical_skill_bundle.v1"

    def __post_init__(self) -> None:
        for label, value in (
            ("body_hash", self.body_hash),
            ("athlete_foundation_hash", self.athlete_foundation_hash),
            ("first_touch_actor_hash", self.first_touch_actor_hash),
            ("pass_skill_hash", self.pass_skill_hash),
            ("shoot_skill_hash", self.shoot_skill_hash),
        ):
            _require_hash(value, label)

    @property
    def bundle_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class TwoVsOnePhysicsConfig:
    physics_timestep_sec: float = 0.002
    skill_control_hz: float = 50.0
    tactical_control_hz: float = 10.0
    maximum_duration_sec: float = 4.0
    maximum_ball_speed_mps: float = 8.0
    maximum_effector_force_n: float = 450.0
    maximum_defender_force_n: float = 650.0
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.two_vs_one_physics_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.physics_timestep_sec,
            self.skill_control_hz,
            self.tactical_control_hz,
            self.maximum_duration_sec,
            self.maximum_ball_speed_mps,
            self.maximum_effector_force_n,
            self.maximum_defender_force_n,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("2v1 physics config must be finite")
        if (
            not 0.001 <= self.physics_timestep_sec <= 0.005
            or not 25.0 <= self.skill_control_hz <= 50.0
            or not 5.0 <= self.tactical_control_hz <= 10.0
            or self.tactical_control_hz >= self.skill_control_hz
            or not 2.0 <= self.maximum_duration_sec <= 8.0
            or not 4.0 <= self.maximum_ball_speed_mps <= 10.0
            or not 200.0 <= self.maximum_effector_force_n <= 500.0
            or not 300.0 <= self.maximum_defender_force_n <= 800.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("2v1 physics config violates hierarchy or safety bounds")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class TwoVsOnePhysicsScenario:
    scenario_id: str
    seed: int
    defender_commitment: float
    finisher_lateral_m: float
    defender_reaction_delay_sec: float = 0.35
    defender_speed_scale: float = 1.0
    ball_ground_friction: float = 0.35
    schema_version: str = "rosclaw_soccer.two_vs_one_physics_scenario.v1"

    def __post_init__(self) -> None:
        if not _IDENTIFIER.fullmatch(self.scenario_id):
            raise ValueError("2v1 physics scenario identity is invalid")
        if isinstance(self.seed, bool) or not 0 <= self.seed <= 2**32 - 1:
            raise ValueError("2v1 physics seed is invalid")
        values = (
            self.defender_commitment,
            self.finisher_lateral_m,
            self.defender_reaction_delay_sec,
            self.defender_speed_scale,
            self.ball_ground_friction,
        )
        if any(not math.isfinite(value) for value in values):
            raise ValueError("2v1 physics scenario values must be finite")
        if (
            not 0.0 <= self.defender_commitment <= 1.0
            or not 1.0 <= abs(self.finisher_lateral_m) <= 1.8
            or not 0.25 <= self.defender_reaction_delay_sec <= 0.50
            or not 0.80 <= self.defender_speed_scale <= 1.20
            or not 0.25 <= self.ball_ground_friction <= 0.50
        ):
            raise ValueError("2v1 physics scenario exceeds its curriculum")

    @property
    def scenario_hash(self) -> str:
        return str(hash_json(asdict(self)))

    @property
    def carrier_position(self) -> NDArray[np.float64]:
        return np.asarray((2.0, 0.0), dtype=np.float64)

    @property
    def ball_position(self) -> NDArray[np.float64]:
        return np.asarray((2.36, 0.0), dtype=np.float64)

    @property
    def finisher_position(self) -> NDArray[np.float64]:
        return np.asarray((4.0, self.finisher_lateral_m), dtype=np.float64)

    @property
    def goal_position(self) -> NDArray[np.float64]:
        return np.asarray((9.5, 0.0), dtype=np.float64)

    @property
    def defender_position(self) -> NDArray[np.float64]:
        side = math.copysign(1.0, self.finisher_lateral_m)
        cover = np.asarray((3.80, 1.05 * side), dtype=np.float64)
        press = np.asarray((3.05, 0.05 * side), dtype=np.float64)
        return cover * (1.0 - self.defender_commitment) + press * self.defender_commitment

    def observations(self) -> tuple[float, float, float, float, float]:
        defender = self.defender_position
        ball = self.ball_position
        pressure_distance = float(np.linalg.norm(defender - self.carrier_position))
        pass_clearance = _point_segment_distance(defender, ball, self.finisher_position)
        shot_clearance = _point_segment_distance(defender, ball, self.goal_position)
        return (
            _clamp01(1.0 - pressure_distance / 2.4),
            _clamp01(pass_clearance / 1.20),
            _clamp01(shot_clearance / 1.20),
            _clamp01(ball[0] / self.goal_position[0]),
            _clamp01(self.finisher_position[0] / self.goal_position[0]),
        )

    def state(
        self,
        *,
        skill_bundle: FrozenTacticalSkillBundle,
        config: TwoVsOnePhysicsConfig,
    ) -> TwoVsOneState:
        pressure, pass_open, shot_open, goal_progress, teammate_progress = self.observations()
        world = {
            "carrier": self.carrier_position.tolist(),
            "ball": self.ball_position.tolist(),
            "finisher": self.finisher_position.tolist(),
            "defender": self.defender_position.tolist(),
            "goal": self.goal_position.tolist(),
        }
        defender_policy = {
            "reaction_delay_sec": self.defender_reaction_delay_sec,
            "speed_scale": self.defender_speed_scale,
            "smooth_commitment": self.defender_commitment,
            "controller": "bounded_predictive_interceptor_v1",
        }
        return TwoVsOneState(
            state_id=f"state.{self.scenario_id}",
            seed=self.seed,
            self_state_hash=hash_json({"carrier": world["carrier"], "ball": world["ball"]}),
            world_state_hash=hash_json(world),
            scenario_hash=self.scenario_hash,
            environment_hash=hash_json(
                {"physics_config_hash": config.config_hash, "world": "planar_2v1_mujoco_v1"}
            ),
            frozen_foundation_hash=skill_bundle.athlete_foundation_hash,
            frozen_skill_bundle_hash=skill_bundle.bundle_hash,
            frozen_defender_hash=hash_json(defender_policy),
            carrier_pressure=pressure,
            teammate_lane_openness=pass_open,
            shot_lane_openness=shot_open,
            goal_progress=goal_progress,
            teammate_progress=teammate_progress,
        )


@dataclass(frozen=True)
class TwoVsOnePhysicsResult:
    action: TacticalAction
    focal_agent_present: bool
    goal_scored: bool
    pass_completed: bool
    intercepted: bool
    first_ball_contact_role: str | None
    first_ball_contact_time_sec: float | None
    terminal_time_sec: float
    maximum_ball_speed_mps: float
    maximum_ball_progress_m: float
    finite_state: bool
    team_reward: float
    role_reward: float
    possession_progress: float
    safety_cost: float
    trajectory_hash: str
    action_trace_hash: str
    schema_version: str = "rosclaw_soccer.two_vs_one_physics_result.v1"

    @property
    def safe(self) -> bool:
        return self.safety_cost == 0.0 and self.finite_state

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["action"] = self.action.value
        value["safe"] = self.safe
        return value


def _model_xml(scenario: TwoVsOnePhysicsScenario, *, focal_agent_present: bool) -> str:
    finisher = scenario.finisher_position
    defender = scenario.defender_position
    finisher_contact = (
        'contype="1" conaffinity="1"' if focal_agent_present else 'contype="0" conaffinity="0"'
    )
    return f"""
<mujoco model="rosclaw_soccer_2v1">
  <option timestep="0.002" gravity="0 0 -9.81" integrator="RK4"/>
  <default><geom condim="4" friction="0.8 0.02 0.002" solref="0.006 1"/></default>
  <worldbody>
    <geom name="floor" type="plane" size="12 6 .1" rgba=".08 .42 .16 1"/>
    <body name="carrier" pos="2 0 .4">
      <joint name="carrier_x" type="slide" axis="1 0 0" damping="30"/>
      <joint name="carrier_y" type="slide" axis="0 1 0" damping="30"/>
      <geom name="carrier_geom" type="cylinder" size=".22 .4" mass="60"
            contype="2" conaffinity="2" rgba=".85 .12 .12 1"/>
    </body>
    <body name="carrier_foot" pos="2.02 0 .11">
      <joint name="carrier_foot_x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="carrier_foot_y" type="slide" axis="0 1 0" damping="2"/>
      <geom name="carrier_foot_geom" type="sphere" size=".105" mass="4" rgba="1 .28 .18 1"/>
    </body>
    <body name="finisher" pos="{finisher[0]:.12g} {finisher[1]:.12g} .4">
      <joint name="finisher_x" type="slide" axis="1 0 0" damping="30"/>
      <joint name="finisher_y" type="slide" axis="0 1 0" damping="30"/>
      <geom name="finisher_geom" type="cylinder" size=".22 .4" mass="60"
            contype="2" conaffinity="2" rgba=".95 .35 .12 1"/>
    </body>
    <body name="finisher_foot" pos="{finisher[0]:.12g} {finisher[1]:.12g} .11">
      <joint name="finisher_foot_x" type="slide" axis="1 0 0" damping="2"/>
      <joint name="finisher_foot_y" type="slide" axis="0 1 0" damping="2"/>
      <geom name="finisher_foot_geom" type="sphere" size=".105" mass="4"
            {finisher_contact} rgba="1 .48 .18 1"/>
    </body>
    <body name="defender" pos="{defender[0]:.12g} {defender[1]:.12g} .3">
      <joint name="defender_x" type="slide" axis="1 0 0" damping="30"/>
      <joint name="defender_y" type="slide" axis="0 1 0" damping="30"/>
      <geom name="defender_geom" type="cylinder" size=".26 .3" mass="70" rgba=".08 .18 .86 1"/>
    </body>
    <body name="ball" pos="2.36 0 .11">
      <freejoint name="ball_free"/>
      <geom name="ball_geom" type="sphere" size=".11" mass=".43"
            friction="{scenario.ball_ground_friction:.12g} .01 .001"
            rgba=".95 .95 .95 1"/>
    </body>
  </worldbody>
</mujoco>
"""


def _slide_addresses(model: Any, prefix: str) -> tuple[int, int]:
    import mujoco

    addresses: list[int] = []
    for suffix in ("x", "y"):
        joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"{prefix}_{suffix}"))
        if joint < 0:
            raise RuntimeError(f"missing tactical plane joint: {prefix}_{suffix}")
        addresses.append(int(model.jnt_dofadr[joint]))
    if addresses[1] != addresses[0] + 1:
        raise RuntimeError("tactical plane slide joints must be contiguous")
    return addresses[0], addresses[1]


def simulate_two_vs_one_physics(
    *,
    scenario: TwoVsOnePhysicsScenario,
    action: TacticalAction,
    skill_bundle: FrozenTacticalSkillBundle,
    config: TwoVsOnePhysicsConfig | None = None,
    focal_agent_present: bool = True,
) -> tuple[TwoVsOnePhysicsResult, dict[str, NDArray[Any]]]:
    """Execute one deterministic physical action or focal-player ablation."""

    import mujoco

    if action not in _ACTIONS:
        raise ValueError("2v1 physics currently qualifies only PASS and SHOOT")
    active = config or TwoVsOnePhysicsConfig()
    model = mujoco.MjModel.from_xml_string(
        _model_xml(scenario, focal_agent_present=focal_agent_present)
    )
    model.opt.timestep = active.physics_timestep_sec
    data = mujoco.MjData(model)
    prefixes = ("carrier", "carrier_foot", "finisher", "finisher_foot", "defender")
    addresses = {prefix: _slide_addresses(model, prefix)[0] for prefix in prefixes}
    origins = {
        "carrier": scenario.carrier_position,
        "carrier_foot": np.asarray((2.02, 0.0), dtype=np.float64),
        "finisher": scenario.finisher_position,
        "finisher_foot": scenario.finisher_position,
        "defender": scenario.defender_position,
    }
    ball_joint = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "ball_free"))
    ball_qpos = int(model.jnt_qposadr[ball_joint])
    ball_qvel = int(model.jnt_dofadr[ball_joint])
    geoms = {
        name: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name))
        for name in (
            "carrier_foot_geom",
            "finisher_foot_geom",
            "defender_geom",
            "ball_geom",
        )
    }
    mujoco.mj_forward(model, data)

    trace_rows: dict[str, list[Any]] = {
        "time": [],
        "ball_pose": [],
        "ball_velocity": [],
        "carrier_position": [],
        "finisher_position": [],
        "defender_position": [],
        "carrier_foot_position": [],
        "finisher_foot_position": [],
        "ball_contact_role": [],
        "control_force": [],
        "focal_agent_present": [],
    }
    first_role: str | None = None
    first_time: float | None = None
    carrier_contact = False
    pass_completed = False
    intercepted = False
    goal_scored = False
    finite = True
    maximum_speed = 0.0
    maximum_progress = float(scenario.ball_position[0])
    skill_steps = max(1, round(1.0 / (active.skill_control_hz * active.physics_timestep_sec)))
    origins_vector = {key: value.copy() for key, value in origins.items()}
    initial_ball = scenario.ball_position
    destination = (
        scenario.finisher_position if action is TacticalAction.PASS else scenario.goal_position
    )
    direction = destination - initial_ball
    direction /= np.linalg.norm(direction)
    behind = initial_ball - direction * 0.27
    latest_forces = np.zeros((5, 2), dtype=np.float64)

    def position(prefix: str) -> NDArray[np.float64]:
        start = addresses[prefix]
        offset = np.asarray(data.qpos[start : start + 2], dtype=np.float64).copy()
        return origins_vector[prefix] + offset

    def pd_force(
        prefix: str,
        target: NDArray[np.float64],
        *,
        maximum: float,
        kp: float,
        kd: float,
    ) -> NDArray[np.float64]:
        start = addresses[prefix]
        velocity = np.asarray(data.qvel[start : start + 2], dtype=np.float64)
        force = np.clip(kp * (target - position(prefix)) - kd * velocity, -maximum, maximum)
        data.qfrc_applied[start : start + 2] = force
        latest_forces[prefixes.index(prefix)] = force
        return force

    terminal = False
    maximum_steps = math.ceil(active.maximum_duration_sec / active.physics_timestep_sec)
    for step in range(maximum_steps):
        time_sec = float(data.time)
        ball = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64).copy()
        ball_velocity = np.asarray(data.qvel[ball_qvel : ball_qvel + 6], dtype=np.float64).copy()
        if step % skill_steps == 0:
            data.qfrc_applied[:] = 0.0
            latest_forces[:] = 0.0
            if time_sec < 0.24:
                carrier_foot_target = behind
            elif time_sec < 0.40:
                fraction = (time_sec - 0.24) / 0.16
                carrier_foot_target = behind + direction * 0.70 * fraction
            elif time_sec < 0.65:
                fraction = (time_sec - 0.40) / 0.25
                carrier_foot_target = behind + direction * 0.70 * (1.0 - fraction)
            else:
                carrier_foot_target = behind
            pd_force(
                "carrier_foot",
                carrier_foot_target,
                maximum=active.maximum_effector_force_n,
                kp=800.0,
                kd=55.0,
            )
            pd_force(
                "carrier",
                scenario.carrier_position,
                maximum=600.0,
                kp=600.0,
                kd=100.0,
            )
            pd_force(
                "finisher",
                scenario.finisher_position + np.asarray((0.20, 0.0)),
                maximum=600.0,
                kp=600.0,
                kd=100.0,
            )
            pd_force(
                "finisher_foot",
                scenario.finisher_position,
                maximum=350.0,
                kp=600.0,
                kd=60.0,
            )
            commitment = scenario.defender_commitment
            response = (1.0 - commitment) if action is TacticalAction.PASS else commitment
            if time_sec < scenario.defender_reaction_delay_sec:
                defender_target = scenario.defender_position
            else:
                predicted = ball[:2] + ball_velocity[:2] * 0.10
                defender_target = (
                    scenario.defender_position * (1.0 - response) + predicted * response
                )
            pd_force(
                "defender",
                defender_target,
                maximum=active.maximum_defender_force_n
                * scenario.defender_speed_scale
                * (0.35 + 0.65 * response),
                kp=500.0,
                kd=100.0,
            )

        mujoco.mj_step(model, data)
        contact_code = 0
        for contact_index in range(int(data.ncon)):
            contact = data.contact[contact_index]
            pair = {int(contact.geom1), int(contact.geom2)}
            if geoms["ball_geom"] not in pair:
                continue
            role: str | None = None
            if geoms["carrier_foot_geom"] in pair:
                role = "carrier"
                carrier_contact = True
                contact_code = 1
            elif focal_agent_present and geoms["finisher_foot_geom"] in pair:
                role = "finisher"
                pass_completed = action is TacticalAction.PASS and carrier_contact
                contact_code = 2
            elif geoms["defender_geom"] in pair:
                role = "defender"
                intercepted = carrier_contact
                contact_code = -1
            if role is not None and first_role is None:
                first_role = role
                first_time = float(data.time)

        ball_after = np.asarray(data.qpos[ball_qpos : ball_qpos + 3], dtype=np.float64).copy()
        velocity_after = np.asarray(data.qvel[ball_qvel : ball_qvel + 6], dtype=np.float64).copy()
        maximum_speed = max(maximum_speed, float(np.linalg.norm(velocity_after[:3])))
        maximum_progress = max(maximum_progress, float(ball_after[0]))
        finite = finite and all(
            np.all(np.isfinite(value))
            for value in (data.qpos, data.qvel, data.qacc, data.ctrl, data.qfrc_applied)
        )
        goal_scored = bool(
            ball_after[0] >= scenario.goal_position[0] and abs(ball_after[1]) <= 1.83
        )
        out_of_bounds = bool(
            ball_after[0] < -0.5
            or ball_after[0] > 11.0
            or abs(ball_after[1]) > 4.5
            or ball_after[2] > 3.0
        )
        terminal = bool(
            not finite
            or pass_completed
            or intercepted
            or goal_scored
            or out_of_bounds
            or maximum_speed > active.maximum_ball_speed_mps
        )
        trace_rows["time"].append(float(data.time))
        trace_rows["ball_pose"].append(data.qpos[ball_qpos : ball_qpos + 7].copy())
        trace_rows["ball_velocity"].append(velocity_after)
        trace_rows["carrier_position"].append(position("carrier").copy())
        trace_rows["finisher_position"].append(position("finisher").copy())
        trace_rows["defender_position"].append(position("defender").copy())
        trace_rows["carrier_foot_position"].append(position("carrier_foot").copy())
        trace_rows["finisher_foot_position"].append(position("finisher_foot").copy())
        trace_rows["ball_contact_role"].append(contact_code)
        trace_rows["control_force"].append(latest_forces.reshape(-1).copy())
        trace_rows["focal_agent_present"].append(int(focal_agent_present))
        if terminal:
            break

    trajectory: dict[str, NDArray[Any]] = {
        key: np.asarray(value) for key, value in trace_rows.items()
    }
    progress = _clamp01(
        (maximum_progress - scenario.ball_position[0])
        / (scenario.goal_position[0] - scenario.ball_position[0])
    )
    safe = bool(
        finite and maximum_speed <= active.maximum_ball_speed_mps and trajectory["time"].size > 1
    )
    team_reward = (
        1.50 * float(goal_scored)
        + 0.90 * float(pass_completed)
        + 0.20 * progress
        - 0.80 * float(intercepted)
    )
    role_reward = float(
        (action is TacticalAction.PASS and pass_completed)
        or (action is TacticalAction.SHOOT and goal_scored)
    )
    action_trace_hash = hash_json(
        {
            "action": action.value,
            "focal_agent_present": focal_agent_present,
            "skill_bundle_hash": skill_bundle.bundle_hash,
            "control_force_digest": hash_bytes(
                np.ascontiguousarray(trajectory["control_force"]).tobytes()
            ),
        }
    )
    result = TwoVsOnePhysicsResult(
        action=action,
        focal_agent_present=focal_agent_present,
        goal_scored=goal_scored,
        pass_completed=pass_completed,
        intercepted=intercepted,
        first_ball_contact_role=first_role,
        first_ball_contact_time_sec=first_time,
        terminal_time_sec=float(trajectory["time"][-1]),
        maximum_ball_speed_mps=maximum_speed,
        maximum_ball_progress_m=maximum_progress,
        finite_state=finite,
        team_reward=team_reward,
        role_reward=role_reward,
        possession_progress=progress,
        safety_cost=0.0 if safe else 1.0,
        trajectory_hash=trajectory_digest(trajectory),
        action_trace_hash=action_trace_hash,
    )
    return result, trajectory


def matched_two_vs_one_decision(
    *,
    scenario: TwoVsOnePhysicsScenario,
    action: TacticalAction,
    policy_hash: str,
    skill_bundle: FrozenTacticalSkillBundle,
    config: TwoVsOnePhysicsConfig | None = None,
    weights: TacticalRewardWeights | None = None,
) -> tuple[
    TwoVsOneDecisionEvidence,
    TwoVsOnePhysicsResult,
    TwoVsOnePhysicsResult,
    dict[str, NDArray[Any]],
    dict[str, NDArray[Any]],
]:
    """Run one action and its matched focal-player removal."""

    _require_hash(policy_hash, "policy_hash")
    active = config or TwoVsOnePhysicsConfig()
    primary, primary_trace = simulate_two_vs_one_physics(
        scenario=scenario,
        action=action,
        skill_bundle=skill_bundle,
        config=active,
        focal_agent_present=True,
    )
    ablated, ablated_trace = simulate_two_vs_one_physics(
        scenario=scenario,
        action=action,
        skill_bundle=skill_bundle,
        config=active,
        focal_agent_present=False,
    )
    state = scenario.state(skill_bundle=skill_bundle, config=active)
    rollout = MatchedTacticalRollout(
        state_hash=state.state_hash,
        policy_hash=policy_hash,
        action=action,
        action_trace_hash=primary.action_trace_hash,
        trajectory_hash=primary.trajectory_hash,
        ablation_action_trace_hash=ablated.action_trace_hash,
        ablated_trajectory_hash=ablated.trajectory_hash,
        team_reward=primary.team_reward,
        role_reward=primary.role_reward,
        ablated_team_reward=ablated.team_reward,
        possession_progress=primary.possession_progress,
        safety_cost=primary.safety_cost,
    )
    evidence = TwoVsOneDecisionEvidence(
        state=state,
        rollout=rollout,
        weights=weights or TacticalRewardWeights(),
    )
    return evidence, primary, ablated, primary_trace, ablated_trace


def persist_matched_two_vs_one_decision(
    *,
    output_dir: Path,
    source_checkout: Path,
    scenario: TwoVsOnePhysicsScenario,
    action: TacticalAction,
    policy_hash: str,
    skill_bundle: FrozenTacticalSkillBundle,
    config: TwoVsOnePhysicsConfig | None = None,
    weights: TacticalRewardWeights | None = None,
) -> dict[str, Any]:
    """Persist one content-bound decision and both physical trajectories."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("2v1 evidence output must be new and outside the checkout")
    evidence, primary, ablated, primary_trace, ablated_trace = matched_two_vs_one_decision(
        scenario=scenario,
        action=action,
        policy_hash=policy_hash,
        skill_bundle=skill_bundle,
        config=config,
        weights=weights,
    )
    destination.mkdir(parents=True)
    artifacts: dict[str, dict[str, str]] = {}
    for label, trajectory in (("primary", primary_trace), ("ablated", ablated_trace)):
        path = destination / f"{label}-trajectory.npz"
        temporary = path.with_suffix(".npz.tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
        os.replace(temporary, path)
        artifacts[label] = {
            "file": path.name,
            "file_hash": hash_bytes(path.read_bytes()),
            "trajectory_digest": trajectory_digest(trajectory),
        }
    active = config or TwoVsOnePhysicsConfig()
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.two_vs_one_physics_evidence.v1",
        "status": "PASS_MATCHED_TACTICAL_ROLLOUT"
        if evidence.promotion_eligible
        else "REJECTED_MATCHED_TACTICAL_ROLLOUT",
        "scenario": asdict(scenario),
        "scenario_hash": scenario.scenario_hash,
        "physics_config": asdict(active),
        "physics_config_hash": active.config_hash,
        "skill_bundle": asdict(skill_bundle),
        "skill_bundle_hash": skill_bundle.bundle_hash,
        "decision_evidence": evidence.to_dict(),
        "decision_evidence_hash": evidence.evidence_hash,
        "primary_result": primary.to_dict(),
        "ablated_result": ablated.to_dict(),
        "artifacts": artifacts,
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "tactical_plane_only": True,
            "g1_whole_body_rollout_claimed": False,
            "pixels_used_for_scoring": False,
            "promotion_eligible": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    path = destination / "two-vs-one-report.json"
    temporary_report = path.with_suffix(".json.tmp")
    temporary_report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary_report, path)
    return report


__all__ = [
    "FrozenTacticalSkillBundle",
    "TwoVsOnePhysicsConfig",
    "TwoVsOnePhysicsResult",
    "TwoVsOnePhysicsScenario",
    "matched_two_vs_one_decision",
    "persist_matched_two_vs_one_decision",
    "simulate_two_vs_one_physics",
]
