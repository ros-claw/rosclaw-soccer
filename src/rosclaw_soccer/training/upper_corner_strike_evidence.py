"""Strict evidence loop for the shooter round of alternating team Growth."""

from __future__ import annotations

import hashlib
import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.upper_corner_strike import UpperCornerStrikePolicy
from rosclaw_soccer.providers.g1.asset_qualification import (
    qualify_g1_assets,
    trajectory_digest,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import (
    G1GoalkeeperConfig,
    G1SharedWorldResult,
    simulate_shared_world,
)
from rosclaw_soccer.training.regulation_dead_corner_save import (
    RegulationDeadCornerLane,
    regulation_dead_corner_lane_kwargs,
    regulation_dead_corner_lanes,
)


@dataclass(frozen=True)
class UpperCornerStrikeEvidenceConfig:
    policy: UpperCornerStrikePolicy = UpperCornerStrikePolicy()
    target_height_m: float = 1.75
    minimum_crossing_height_m: float = 1.70
    maximum_target_error_m: float = 0.12
    maximum_post_surface_clearance_m: float = 0.15
    minimum_height_improvement_m: float = 0.40
    # 0.0975 is retained as an S96-v1 failure memory: the left crossing
    # reached only 1.691 m.  The replacement remains disjoint from nominal.
    sealed_holdout_friction: tuple[float, ...] = (0.0900, 0.1025)
    minimum_passer_pelvis_height_m: float = 0.60
    minimum_shooter_pelvis_height_m: float = 0.60
    minimum_goalkeeper_pelvis_height_m: float = 0.60
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.upper_corner_strike_evidence_config.v1"

    def __post_init__(self) -> None:
        values = (
            self.target_height_m,
            self.minimum_crossing_height_m,
            self.maximum_target_error_m,
            self.maximum_post_surface_clearance_m,
            self.minimum_height_improvement_m,
            *self.sealed_holdout_friction,
            self.minimum_passer_pelvis_height_m,
            self.minimum_shooter_pelvis_height_m,
            self.minimum_goalkeeper_pelvis_height_m,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("upper-corner strike evidence config must be finite")
        if not 1.65 <= self.minimum_crossing_height_m <= self.target_height_m <= 2.15:
            raise ValueError("upper-corner target must lie in the regulation upper region")
        if not 0.05 <= self.maximum_target_error_m <= 0.15:
            raise ValueError("upper-corner target-error gate is invalid")
        if not 0.05 <= self.maximum_post_surface_clearance_m <= 0.15:
            raise ValueError("upper-corner post-clearance gate is invalid")
        if not 0.20 <= self.minimum_height_improvement_m <= 0.80:
            raise ValueError("upper-corner improvement gate is invalid")
        if len(set(self.sealed_holdout_friction)) < 2 or 0.10 in self.sealed_holdout_friction:
            raise ValueError("upper-corner holdout friction must be disjoint from discovery")
        if any(not 0.08 <= value <= 0.12 for value in self.sealed_holdout_friction):
            raise ValueError("upper-corner holdout friction is outside the qualified domain")
        if (
            min(
                self.minimum_passer_pelvis_height_m,
                self.minimum_shooter_pelvis_height_m,
                self.minimum_goalkeeper_pelvis_height_m,
            )
            < 0.55
        ):
            raise ValueError("upper-corner stability gate is too low")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("upper-corner evidence must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def run_upper_corner_strike_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    output_dir: Path,
    source_checkout: Path,
    config: UpperCornerStrikeEvidenceConfig | None = None,
) -> dict[str, Any]:
    """Compare the frozen S93 parent to one bounded shooter-only child."""

    active = config or UpperCornerStrikeEvidenceConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("upper-corner evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    artifacts = {
        "striker_actor": striker_actor_path,
        "goalkeeper_actor": goalkeeper_actor_path,
        "gmt_model": gmt_model_path,
        "gmt_skill": gmt_skill_path,
    }
    if any(not path.expanduser().resolve().is_file() for path in artifacts.values()):
        raise ValueError("upper-corner evidence artifact is unavailable")
    dive = dive_source_checkout.expanduser().resolve()
    if not dive.is_dir() or not (dive / "LICENSE").is_file():
        raise ValueError("upper-corner goalkeeper reference source is incomplete")
    output.mkdir(parents=True)
    request = {
        "schema_version": "rosclaw_soccer.upper_corner_strike_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "policy_hash": active.policy.artifact_hash,
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "artifacts": {
            name + "_hash": hash_bytes(path.expanduser().resolve().read_bytes())
            for name, path in artifacts.items()
        },
        "dive_source_commit": _git_head(dive),
        "source_commit": _git_head(checkout),
        "growth_contract": {
            "plastic_role": "shooter",
            "frozen_roles": ["passer", "goalkeeper"],
            "discovery_friction": 0.10,
            "holdout_friction": list(active.sealed_holdout_friction),
        },
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
    }
    _write_json(output / "request.json", request)

    lanes: dict[str, Any] = {}
    for lane in regulation_dead_corner_lanes():
        nominal = _lane_kwargs(
            lane=lane,
            active=active,
            striker_actor_path=striker_actor_path,
            goalkeeper_actor_path=goalkeeper_actor_path,
            gmt_model_path=gmt_model_path,
            gmt_skill_path=gmt_skill_path,
            dive_source_checkout=dive,
        )
        parent_kwargs = dict(nominal)
        parent_kwargs["goalkeeper_config"] = None
        parent_kwargs["shooter_ballistic_contact_torque_config"] = (
            regulation_dead_corner_lane_kwargs(
                lane=lane,
                striker_actor_path=striker_actor_path,
                goalkeeper_actor_path=goalkeeper_actor_path,
                gmt_model_path=gmt_model_path,
                gmt_skill_path=gmt_skill_path,
                dive_source_checkout=dive,
            )["shooter_ballistic_contact_torque_config"]
        )
        parent_kwargs["shooter_parameter_overrides"] = {
            **parent_kwargs["shooter_parameter_overrides"],
            "foot_yaw_offset": 0.01,
        }
        candidate_kwargs = dict(nominal)
        keeper = candidate_kwargs.pop("goalkeeper_config")
        parent_result, parent_trajectory = simulate_shared_world(asset_root, **parent_kwargs)
        candidate_result, candidate_trajectory = simulate_shared_world(
            asset_root, **candidate_kwargs
        )
        discovery = _evaluate_lane(
            result=candidate_result,
            goal=candidate_kwargs["goal_spec"],
            config=active,
        )
        parent_height = parent_result.goal_crossing_z_m
        candidate_height = candidate_result.goal_crossing_z_m
        discovery["gates"]["height_improvement"] = bool(
            parent_height is not None
            and candidate_height is not None
            and candidate_height - parent_height >= active.minimum_height_improvement_m
        )
        discovery["passed"] = bool(all(discovery["gates"].values()))
        nominal_path = output / f"{lane.lane_id}-upper-corner-trajectory.npz"
        np.savez_compressed(nominal_path, **candidate_trajectory)  # type: ignore[arg-type]

        holdouts: dict[str, Any] = {}
        for friction in active.sealed_holdout_friction:
            holdout_kwargs = dict(candidate_kwargs)
            holdout_kwargs["ball_ground_friction"] = friction
            first_result, first_trajectory = simulate_shared_world(asset_root, **holdout_kwargs)
            replay_result, replay_trajectory = simulate_shared_world(asset_root, **holdout_kwargs)
            evaluation = _evaluate_lane(
                result=first_result,
                goal=holdout_kwargs["goal_spec"],
                config=active,
            )
            strict = bool(
                first_result.to_dict() == replay_result.to_dict()
                and trajectory_digest(first_trajectory) == trajectory_digest(replay_trajectory)
            )
            evaluation["gates"]["strict_replay"] = strict
            evaluation["passed"] = bool(all(evaluation["gates"].values()))
            holdouts[f"friction-{friction:.4f}"] = {
                **evaluation,
                "friction": friction,
                "trajectory_digest": trajectory_digest(first_trajectory),
            }

        teammate_kwargs = dict(nominal)
        teammate_kwargs["goalkeeper_config"] = G1GoalkeeperConfig(
            depth_from_goal_line_m=keeper.depth_from_goal_line_m,
            initial_lateral_position_m=keeper.initial_lateral_position_m,
            regulation_goal_positioning_enabled=True,
            anticipation_enabled=True,
            joint_guard_margin_rad=0.10,
            joint_guard_prediction_horizon_sec=0.20,
            joint_guard_boundary_kp=200.0,
            joint_guard_boundary_kd=20.0,
        )
        teammate_result, teammate_trajectory = simulate_shared_world(asset_root, **teammate_kwargs)
        teammate_gates = {
            "three_agents_present": teammate_result.goalkeeper_enabled,
            "ordered_pass_and_shot": bool(
                teammate_result.pass_contact_time_sec is not None
                and teammate_result.shot_contact_time_sec is not None
                and teammate_result.pass_contact_time_sec < teammate_result.shot_contact_time_sec
            ),
            "frozen_passer_stable": (
                teammate_result.passer_min_pelvis_height_m >= active.minimum_passer_pelvis_height_m
            ),
            "frozen_goalkeeper_stable": bool(
                teammate_result.goalkeeper_min_pelvis_height_m is not None
                and teammate_result.goalkeeper_min_pelvis_height_m
                >= active.minimum_goalkeeper_pelvis_height_m
            ),
            "joint_limits": not teammate_result.joint_limit_violation,
            "torque_limits": not teammate_result.torque_limit_violation,
            "zero_actuator_saturation": not teammate_result.actuator_saturation,
        }
        lanes[lane.lane_id] = {
            "lane": asdict(lane),
            "action": asdict(active.policy.action(lane.lane_id)),
            "passed": bool(
                discovery["passed"]
                and all(item["passed"] for item in holdouts.values())
                and all(teammate_gates.values())
            ),
            "parent": {
                "result": parent_result.to_dict(),
                "trajectory_digest": trajectory_digest(parent_trajectory),
            },
            "discovery": discovery,
            "holdouts": holdouts,
            "teammate_replay": {
                "passed": bool(all(teammate_gates.values())),
                "gates": teammate_gates,
                "result": teammate_result.to_dict(),
                "trajectory_digest": trajectory_digest(teammate_trajectory),
            },
            "nominal_trajectory_file": nominal_path.name,
            "nominal_trajectory_hash": hash_bytes(nominal_path.read_bytes()),
            "nominal_trajectory_digest": trajectory_digest(candidate_trajectory),
        }

    gates = {
        "both_regulation_upper_corners": all(item["passed"] for item in lanes.values()),
        "sealed_holdout_is_disjoint": 0.10 not in active.sealed_holdout_friction,
        "one_plastic_role": True,
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.upper_corner_strike_evidence.v1",
        "passed": bool(all(gates.values())),
        "promotion_status": (
            "FROZEN_RESEARCH_DEMO" if all(gates.values()) else "REJECTED_DEVELOPMENT"
        ),
        "claim": "BILATERAL_REGULATION_UPPER_CORNER_STRIKE_ONLY_IF_ALL_PHYSICS_GATES_PASS",
        "gates": gates,
        "policy": active.policy.to_dict(),
        "policy_hash": active.policy.artifact_hash,
        "lanes": lanes,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "fresh_training_performed": True,
    }
    report["evidence_hash"] = hash_json(report)
    _write_json(output / "evidence.json", report)
    return report


def validate_upper_corner_strike_evidence(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("upper-corner evidence must be a JSON object")
    evidence_hash = payload.pop("evidence_hash", None)
    try:
        if (
            payload.get("schema_version") != "rosclaw_soccer.upper_corner_strike_evidence.v1"
            or payload.get("passed") is not True
            or payload.get("promotion_status") != "FROZEN_RESEARCH_DEMO"
            or payload.get("physics_authority") != "CPU_MUJOCO"
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or not all(payload.get("gates", {}).values())
            or hash_json(payload) != evidence_hash
        ):
            raise ValueError("upper-corner evidence authority contract is invalid")
        root = source.parent
        for lane in payload.get("lanes", {}).values():
            trajectory = root / lane["nominal_trajectory_file"]
            if (
                not trajectory.is_file()
                or hash_bytes(trajectory.read_bytes()) != lane["nominal_trajectory_hash"]
            ):
                raise ValueError("upper-corner trajectory binding changed")
    finally:
        if evidence_hash is not None:
            payload["evidence_hash"] = evidence_hash
    return payload


def _lane_kwargs(
    *,
    lane: RegulationDeadCornerLane,
    active: UpperCornerStrikeEvidenceConfig,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
) -> dict[str, Any]:
    kwargs = regulation_dead_corner_lane_kwargs(
        lane=lane,
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        dive_source_checkout=dive_source_checkout,
    )
    goal = replace(kwargs["goal_spec"], target_z_m=active.target_height_m)
    action = active.policy.action(lane.lane_id)
    kwargs.update(
        goal_spec=goal,
        shooter_target=(goal.plane_x_m, goal.target_y_m, goal.target_z_m),
        shooter_parameter_overrides={
            **kwargs["shooter_parameter_overrides"],
            "foot_yaw_offset": action.foot_yaw_offset_rad,
        },
        shooter_ballistic_contact_torque_config=active.policy.torque_config(),
    )
    return kwargs


def _evaluate_lane(
    *,
    result: G1SharedWorldResult,
    goal: Any,
    config: UpperCornerStrikeEvidenceConfig,
) -> dict[str, Any]:
    crossing_y = result.goal_crossing_y_m
    crossing_z = result.goal_crossing_z_m
    clearance = (
        math.inf
        if crossing_y is None
        else goal.width_m / 2.0 - goal.post_radius_m - goal.ball_radius_m - abs(crossing_y)
    )
    gates = {
        "finite_state": result.finite_state,
        "ordered_pass_and_shot": bool(
            result.pass_contact_time_sec is not None
            and result.shot_contact_time_sec is not None
            and result.pass_contact_time_sec < result.shot_contact_time_sec
        ),
        "precise_pass": bool(
            result.pass_delivery_error_m is not None and result.pass_delivery_error_m <= 0.05
        ),
        "inside_regulation_goal": bool(result.goal_plane_crossed and result.goal_crossed),
        "upper_region": bool(
            crossing_z is not None
            and config.minimum_crossing_height_m
            <= crossing_z
            <= goal.height_m - goal.post_radius_m - goal.ball_radius_m
        ),
        "post_hugging": 0.0 <= clearance <= config.maximum_post_surface_clearance_m,
        "target_precision": bool(
            result.target_error_m is not None
            and result.target_error_m <= config.maximum_target_error_m
        ),
        "attacker_stability": bool(
            result.passer_min_pelvis_height_m >= config.minimum_passer_pelvis_height_m
            and result.shooter_min_pelvis_height_m >= config.minimum_shooter_pelvis_height_m
        ),
        "joint_limits": not result.joint_limit_violation,
        "torque_limits": not result.torque_limit_violation,
        "zero_actuator_saturation": not result.actuator_saturation,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "goal_crossing_y_m": crossing_y,
        "goal_crossing_z_m": crossing_z,
        "post_surface_clearance_m": clearance,
        "target_error_m": result.target_error_m,
        "result": result.to_dict(),
    }


def _git_head(checkout: Path) -> str:
    return subprocess.run(
        ["git", "-C", str(checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    ).stdout.strip()


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (Path(__file__), Path(__file__).parents[1] / "growth" / "upper_corner_strike.py"):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


__all__ = [
    "UpperCornerStrikeEvidenceConfig",
    "run_upper_corner_strike_evidence",
    "validate_upper_corner_strike_evidence",
]
