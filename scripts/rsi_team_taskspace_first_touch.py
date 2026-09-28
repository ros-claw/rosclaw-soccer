"""Paired SIM_ONLY shared-world physical first-touch task-space mechanism probe."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets, trajectory_digest
from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.taskspace_swing_evidence import LEG_NAMES, audit_taskspace_swing_trace
from rosclaw_soccer.rsi.taskspace_swing_probe import (
    choose_swing_side,
    release_joint_delta,
    swing_joint_delta,
)
from rosclaw_soccer.rsi.team_adaptive_intercept_navigation import TeamAdaptiveInterceptNavigation
from rosclaw_soccer.rsi.team_context_phase_navigation import TeamContextPhaseNavigation
from rosclaw_soccer.rsi.team_contextual_nav_policy import TeamContextualNavigationMemory
from rosclaw_soccer.rsi.team_foot_velocity_chooser import TeamFootVelocityChooser
from rosclaw_soccer.rsi.team_intercept_navigation import TeamInterceptNavigation
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldConfig,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.skills.team.motor_option import (
    TeamMotorObservation,
    TeamMotorPhysicsObservation,
    TeamMotorTarget,
)
from rosclaw_soccer.training.independent_team_growth import build_independent_three_vs_three_fixture


class TeamSwingMotor:
    needs_foot_kinematics = True
    needs_contact_velocity = True

    def __init__(
        self,
        agent_id: str,
        enabled: bool,
        action: dict[str, Any],
        activation_selector: TeamContextualNavigationMemory | TeamFootVelocityChooser | None = None,
    ) -> None:
        self.agent_id = agent_id
        self.enabled = enabled
        self.action = action
        self.activation_selector = activation_selector
        self.next_frame = 0
        self.side = -1
        self.first_contact_frame: int | None = None
        self.contact_delta = np.zeros(6)
        self.last_residual = np.zeros(29)
        self.observations: dict[str, list[np.ndarray[Any, Any]]] = {
            key: []
            for key in (
                "feet",
                "foot_velocity",
                "jacobian",
                "side",
                "residual",
                "baseline",
                "executed",
                "ball",
                "qpos",
                "qvel",
                "predicted_baseline",
            )
        }
        self.joint_limits: np.ndarray[Any, Any] | None = None
        self.contact_event_times: list[float] = []
        self.own_foot_force_peak_n: list[float] = []
        self.own_foot_contact_point_w: list[tuple[float, float, float]] = []
        self.own_foot_contact_normal_w: list[tuple[float, float, float]] = []
        self.own_foot_relative_velocity_w: list[tuple[float, float, float]] = []
        self.own_foot_normal_impulse_ns: list[float] = []
        self.own_foot_impulse_on_ball_w_ns: list[np.ndarray[Any, Any]] = []
        self.last_physics_time_sec = 0.0
        self.contract_hash = hash_json(
            {
                "schema": "rsi_team_swing_motor_v13",
                "agent": agent_id,
                "enabled": enabled,
                "action": action,
                "activation_selector_hash": (
                    None if activation_selector is None else activation_selector.contract_hash
                ),
                "activation_ceiling": "SIM_ONLY",
            }
        )

    def propose(self, observation: TeamMotorObservation) -> TeamMotorTarget:
        if (
            observation.agent_id != self.agent_id
            or observation.frame != self.next_frame
            or observation.foot_kinematics is None
            or observation.foundation is None
        ):
            raise ValueError("same-player measured foot and foundation target required")
        self.next_frame += 1
        self.own_foot_force_peak_n.append(0.0)
        self.own_foot_contact_point_w.append((0.0, 0.0, 0.0))
        self.own_foot_contact_normal_w.append((0.0, 0.0, 0.0))
        self.own_foot_relative_velocity_w.append((0.0, 0.0, 0.0))
        self.own_foot_normal_impulse_ns.append(0.0)
        self.own_foot_impulse_on_ball_w_ns.append(np.zeros(3))
        kinematics = observation.foot_kinematics
        if kinematics.foot_linear_velocity_world_mps is None:
            raise ValueError("measured same-frame foot velocity required")
        feet = np.asarray(kinematics.foot_position_world_m, dtype=float)
        foot_velocity = np.asarray(kinematics.foot_linear_velocity_world_mps, dtype=float)
        jacobian = np.asarray(kinematics.foot_linear_jacobian_world, dtype=float)
        limits = np.asarray(kinematics.leg_joint_limits_rad, dtype=float)
        if self.joint_limits is None:
            self.joint_limits = np.tile(np.asarray((-10.0, 10.0)), (29, 1))
            self.joint_limits[:12] = limits.reshape(12, 2)
        elif not np.array_equal(self.joint_limits[:12], limits.reshape(12, 2)):
            raise ValueError("physical leg limits changed during episode")
        q = np.asarray(observation.qpos, dtype=float)
        ball = q[36:39]
        baseline = np.asarray(observation.foundation.target.target_rad, dtype=float)
        target = baseline.copy()
        residual = np.zeros(29)
        selected = (
            self.activation_selector is None or self.activation_selector.selected_arm is not None
        )
        if self.enabled and selected and observation.frame >= self.action["entry_frame"]:
            if self.first_contact_frame is None:
                self.side = choose_swing_side(
                    feet,
                    ball,
                    self.side,
                    acquisition_max_gap_m=self.action["swing_foot_acquisition_gap_m"],
                )
            if self.side >= 0:
                ids = list(range(self.side * 6, self.side * 6 + 6))
                if self.first_contact_frame is not None:
                    delta = release_joint_delta(
                        self.contact_delta, observation.frame - self.first_contact_frame
                    )
                else:
                    delta = swing_joint_delta(
                        feet[self.side],
                        ball,
                        jacobian[self.side],
                        baseline[ids],
                        limits[self.side],
                        forward_cap_m=self.action["forward_cap_m"],
                        lateral_cap_m=self.action["lateral_cap_m"],
                        vertical_offset_m=self.action["vertical_offset_m"],
                        strike_through_m=self.action.get("strike_through_m", 0.0),
                    )
                target[ids] = baseline[ids] + delta
                residual[ids] = target[ids] - baseline[ids]
        self.last_residual = residual.copy()
        self.observations["feet"].append(feet.copy())
        self.observations["foot_velocity"].append(foot_velocity.copy())
        self.observations["jacobian"].append(jacobian.copy())
        self.observations["side"].append(np.asarray(self.side))
        self.observations["residual"].append(residual.copy())
        self.observations["baseline"].append(baseline.copy())
        self.observations["executed"].append(target.copy())
        self.observations["ball"].append(ball.copy())
        self.observations["qpos"].append(q.copy())
        self.observations["qvel"].append(np.asarray(observation.qvel, dtype=float).copy())
        self.observations["predicted_baseline"].append(baseline.copy())
        return TeamMotorTarget(
            tuple(float(value) for value in target),
            observation.foundation.target.kp,
            observation.foundation.target.kd,
        )

    def observe_physics(self, observation: TeamMotorPhysicsObservation) -> None:
        if observation.observer_agent_id != self.agent_id or not observation.world_bodies_safe:
            raise ValueError("unsafe or foreign physical motor observation")
        dt = observation.time_sec - self.last_physics_time_sec
        if not 0 < dt <= 0.1:
            raise ValueError("non-monotonic or skipped contact physics clock")
        self.last_physics_time_sec = observation.time_sec
        own_contacts = tuple(
            contact
            for contact in observation.ball_contacts
            if contact.agent_id == self.agent_id and contact.effector in {"left_foot", "right_foot"}
        )
        for contact in own_contacts:
            if contact.normal_force_n <= 0:
                continue
            normal = contact.normal_ball_to_counterpart_world
            if normal is None:
                raise ValueError("complete measured ball-foot contact normal required")
            impulse = contact.normal_force_n * dt
            self.own_foot_normal_impulse_ns[-1] += impulse
            self.own_foot_impulse_on_ball_w_ns[-1] -= impulse * np.asarray(normal)
        strongest = max(own_contacts, key=lambda contact: contact.normal_force_n, default=None)
        own_force = 0.0 if strongest is None else strongest.normal_force_n
        if own_force > 1.0 and (
            strongest is None
            or strongest.contact_position_world_m is None
            or strongest.normal_ball_to_counterpart_world is None
            or strongest.counterpart_minus_ball_velocity_world_mps is None
        ):
            raise ValueError("complete measured ball-foot contact kinematics required")
        if own_force > self.own_foot_force_peak_n[-1]:
            self.own_foot_force_peak_n[-1] = own_force
            if (
                strongest is not None
                and strongest.contact_position_world_m is not None
                and strongest.normal_ball_to_counterpart_world is not None
                and strongest.counterpart_minus_ball_velocity_world_mps is not None
            ):
                self.own_foot_contact_point_w[-1] = strongest.contact_position_world_m
                self.own_foot_contact_normal_w[-1] = strongest.normal_ball_to_counterpart_world
                self.own_foot_relative_velocity_w[-1] = (
                    strongest.counterpart_minus_ball_velocity_world_mps
                )
        if self.first_contact_frame is not None:
            return
        if own_force > 1.0:
            self.first_contact_frame = self.next_frame - 1
            self.contact_event_times.append(observation.time_sec)
            if self.side >= 0:
                ids = list(range(self.side * 6, self.side * 6 + 6))
                self.contact_delta = self.last_residual[ids].copy()


def world_contact_code(agent_id: str, player_ids: tuple[str, ...]) -> int:
    """Match the shared world's sorted-roster contact-code contract."""
    if len(set(player_ids)) != len(player_ids) or agent_id not in player_ids:
        raise ValueError("focal agent missing or duplicate player identity")
    return sorted(player_ids).index(agent_id) + 1


def _run_one(
    *,
    mode: str,
    asset_root: Path,
    output_dir: Path,
    fixture: Any,
    scenario: IndependentTeamWorldScenario,
    protocol: dict[str, Any],
    navigation_policy: (
        TeamInterceptNavigation
        | TeamContextPhaseNavigation
        | TeamContextualNavigationMemory
        | TeamFootVelocityChooser
        | TeamAdaptiveInterceptNavigation
        | None
    ) = None,
    motor_option: TeamSwingMotor | None = None,
) -> dict[str, Any]:
    enabled = mode == "candidate"
    motor = motor_option or TeamSwingMotor(
        protocol["focal_agent_id"], enabled, protocol["candidate_action"]
    )
    if motor.agent_id != protocol["focal_agent_id"] or motor.enabled != enabled:
        raise ValueError("same-player candidate/parent motor required")
    result, trace = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=fixture.cells,
        players=fixture.players,
        scenario=scenario,
        goal=fixture.goal,
        config=IndependentTeamWorldConfig(simulation_duration_sec=protocol["frames"] * 0.02),
        motor_options={motor.agent_id: motor},
        navigation_policies=(
            {motor.agent_id: navigation_policy} if navigation_policy is not None else None
        ),
    )
    if motor.next_frame != protocol["frames"] or motor.joint_limits is None:
        raise ValueError("incomplete paired motor episode")
    folder = output_dir / mode
    folder.mkdir(parents=True)
    physics_path = folder / "trajectory.npz"
    np.savez_compressed(physics_path, **trace)  # type: ignore[arg-type]
    force = np.asarray(trace["ball_contact_force_n"])
    agent_codes = np.asarray(trace["ball_contact_agent_code"])
    effectors = np.asarray(trace["ball_contact_effector_code"])
    # Shared-world contact codes follow sorted roster IDs, not fixture layout.
    focal_code = world_contact_code(
        motor.agent_id, tuple(player.agent_id for player in fixture.players)
    )
    focal_foot = (force > 1.0) & (agent_codes == focal_code) & np.isin(effectors, (1, 2))
    observed_own_foot_force = np.asarray(motor.own_foot_force_peak_n)
    action_path = folder / "taskspace_trace.npz"
    np.savez_compressed(
        action_path,
        pre_step_foot_link_position_w=np.asarray(motor.observations["feet"])[:, None],
        pre_step_foot_linear_velocity_w=np.asarray(motor.observations["foot_velocity"])[:, None],
        pre_step_foot_linear_jacobian_w=np.asarray(motor.observations["jacobian"])[:, None],
        taskspace_selected_side=np.asarray(motor.observations["side"])[:, None],
        applied_taskspace_joint_delta_rad=np.asarray(motor.observations["residual"])[:, None],
        baseline_taskspace_joint_target_rad=np.asarray(motor.observations["baseline"])[:, None],
        executed_taskspace_joint_target_rad=np.asarray(motor.observations["executed"])[:, None],
        taskspace_joint_limits_rad=motor.joint_limits[None],
        pre_step_ball_position_local_m=np.asarray(motor.observations["ball"])[:, None],
        pre_step_focal_qpos=np.asarray(motor.observations["qpos"])[:, None],
        pre_step_focal_qvel=np.asarray(motor.observations["qvel"])[:, None],
        observed_ball_body_contact_force_peak_n=np.repeat(
            observed_own_foot_force[:, None, None], 6, axis=2
        ),
        observed_own_foot_contact_position_w=np.asarray(motor.own_foot_contact_point_w)[:, None],
        observed_own_foot_contact_normal_ball_to_foot_w=np.asarray(motor.own_foot_contact_normal_w)[
            :, None
        ],
        observed_own_foot_counterpart_minus_ball_velocity_w=np.asarray(
            motor.own_foot_relative_velocity_w
        )[:, None],
        observed_own_foot_normal_impulse_ns=np.asarray(motor.own_foot_normal_impulse_ns)[:, None],
        observed_own_foot_impulse_on_ball_w_ns=np.asarray(motor.own_foot_impulse_on_ball_w_ns)[
            :, None
        ],
        predicted_baseline_joint_target_rad=np.asarray(motor.observations["predicted_baseline"])[
            :, None
        ],
    )
    audit_report = {
        "taskspace_forward_m": protocol["candidate_action"]["forward_cap_m"],
        "taskspace_lateral_cap_m": protocol["candidate_action"]["lateral_cap_m"],
        "taskspace_vertical_offset_m": protocol["candidate_action"]["vertical_offset_m"],
        "taskspace_strike_through_m": protocol["candidate_action"].get("strike_through_m", 0.0),
        "taskspace_acquisition_max_gap_m": protocol["candidate_action"][
            "swing_foot_acquisition_gap_m"
        ],
        "taskspace_leg_joint_names": [list(row) for row in LEG_NAMES],
        "taskspace_joint_order": list(G1_DDS_JOINT_NAMES),
        "selected_taskspace_mask": [
            enabled
            and (
                motor.activation_selector is None
                or motor.activation_selector.selected_arm is not None
            )
        ],
    }
    with np.load(action_path, allow_pickle=False) as action_trace:
        action_audit = audit_taskspace_swing_trace(
            action_trace, audit_report, frames=protocol["frames"], count=1
        )
    force_mask = force > 1.0
    focal_nonfoot = (np.asarray(trace["ball_nonfoot_contact_force_n"]) > 1.0) & (
        np.asarray(trace["ball_nonfoot_contact_agent_code"]) == focal_code
    )
    contact_frames = np.flatnonzero(force_mask)
    report = {
        "schema": "rsi_team_taskspace_first_touch_episode_v13",
        "mode": mode,
        "activation_ceiling": "SIM_ONLY",
        "scenario_hash": scenario.scenario_hash,
        "motor_contract_hash": motor.contract_hash,
        "navigation_contract_hash": (
            navigation_policy.contract_hash if navigation_policy is not None else None
        ),
        "trace_hash": hash_bytes(physics_path.read_bytes()),
        "action_trace_hash": hash_bytes(action_path.read_bytes()),
        "trajectory_digest": trajectory_digest(trace),
        "world_result": result.to_dict(),
        "action_audit": action_audit,
        "first_any_ball_contact_frame": int(contact_frames[0]) if len(contact_frames) else None,
        "ball_contact_frame_count": int(len(contact_frames)),
        "focal_foot_contact_frames": np.flatnonzero(focal_foot).tolist(),
        "motor_observed_own_foot_contact_frames": np.flatnonzero(
            observed_own_foot_force > 1.0
        ).tolist(),
        "focal_nonfoot_contact_frames": np.flatnonzero(focal_nonfoot).tolist(),
        "ball_contact_agent_codes": np.unique(agent_codes[force_mask]).tolist(),
        "ball_contact_effector_codes": np.unique(effectors[force_mask]).tolist(),
        "motor_first_contact_frame": motor.first_contact_frame,
        "motor_contact_event_times_sec": motor.contact_event_times,
        "policy_intervention_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (folder / "report.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    args = parser.parse_args()
    if args.output_dir.exists():
        parser.error("output already exists")
    protocol = json.loads(args.protocol.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_team_taskspace_first_touch_protocol_v13"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("frames") != 250
    ):
        raise ValueError("uncommitted SIM_ONLY mechanism protocol")
    qualification = qualify_g1_assets(args.asset_root)
    qualification.require_eligible()
    fixture = build_independent_three_vs_three_fixture(args.asset_root)
    if protocol["focal_agent_id"] not in {player.agent_id for player in fixture.players}:
        raise ValueError("focal player missing")
    course = protocol["scenario"]
    scenario = IndependentTeamWorldScenario(
        course["scenario_id"],
        tuple(course["ball_initial_position_m"]),
        tuple(course["ball_initial_velocity_mps"]),
        course["seed"],
    )
    source_paths = (
        Path(__file__),
        Path(__file__).parents[1] / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        Path(__file__).parents[1] / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(Path(__file__).parents[1])): hash_bytes(path.read_bytes())
        for path in source_paths
    }
    args.output_dir.mkdir(parents=True)
    reports = [
        _run_one(
            mode=mode,
            asset_root=args.asset_root,
            output_dir=args.output_dir,
            fixture=fixture,
            scenario=scenario,
            protocol=protocol,
        )
        for mode in ("parent", "candidate")
    ]
    if source_hashes != {
        str(path.relative_to(Path(__file__).parents[1])): hash_bytes(path.read_bytes())
        for path in source_paths
    }:
        raise ValueError("source changed during paired physics")
    verdict = {
        "schema": "rsi_team_taskspace_first_touch_pair_v13",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "asset_body_hash": qualification.body_hash,
        "source_hashes": source_hashes,
        "parent_report_hash": reports[0]["report_hash"],
        "candidate_report_hash": reports[1]["report_hash"],
        "parent_ball_contact_frames": reports[0]["ball_contact_frame_count"],
        "candidate_ball_contact_frames": reports[1]["ball_contact_frame_count"],
        "parent_focal_foot_contact_frames": reports[0]["focal_foot_contact_frames"],
        "candidate_focal_foot_contact_frames": reports[1]["focal_foot_contact_frames"],
        "parent_focal_nonfoot_contact_frames": reports[0]["focal_nonfoot_contact_frames"],
        "candidate_focal_nonfoot_contact_frames": reports[1]["focal_nonfoot_contact_frames"],
        "development_only": True,
        "promotion_authorized": False,
    }
    verdict["verdict_hash"] = hash_json(verdict)
    (args.output_dir / "pair_report.json").write_text(
        json.dumps(verdict, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print("RSI_TEAM_TASKSPACE_PAIR=" + json.dumps(verdict, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
