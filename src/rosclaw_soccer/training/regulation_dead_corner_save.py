"""Counterfactual, regulation-goal dead-corner save evidence.

Each lane first proves that the same three-G1 pass and shot scores when the
goalkeeper is absent.  The goalkeeper replay must then make an anatomical,
collision-faithful glove contact, keep the ball out, complete the existing
contact-grounded take-off/landing exam, and reproduce bit-for-bit.  Rendered
pixels never contribute to any gate.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any, cast

import numpy as np

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
from rosclaw_soccer.training.dynamic_corner_save import (
    DynamicCornerSaveLane,
    dynamic_corner_lane_kwargs,
    expanded_dynamic_corner_lanes,
)
from rosclaw_soccer.training.dynamic_takeoff_exam import evaluate_dynamic_takeoff_save

_CLAIM = "STRICT_REGULATION_LATERAL_DEAD_CORNER_SAVE_PAIR"


@dataclass(frozen=True)
class RegulationDeadCornerLane:
    """One wide attacking angle and its causally angle-aligned keeper stance."""

    lane_id: str
    label: str
    source_lane: DynamicCornerSaveLane
    world_translation_m: float
    target_y_m: float
    expected_glove_side: str
    schema_version: str = "rosclaw_soccer.regulation_dead_corner_lane.v1"

    def __post_init__(self) -> None:
        if not self.lane_id or not self.lane_id.replace("-", "").isalnum():
            raise ValueError("dead-corner lane id must be kebab-case text")
        if not self.label or len(self.label) > 96:
            raise ValueError("dead-corner lane label is invalid")
        if not all(math.isfinite(value) for value in (self.world_translation_m, self.target_y_m)):
            raise ValueError("dead-corner lane geometry must be finite")
        if abs(self.world_translation_m) > 3.44 or abs(self.target_y_m) > 3.545:
            raise ValueError("dead-corner lane must stay inside the regulation goal")
        if self.expected_glove_side not in {"left", "right"}:
            raise ValueError("dead-corner glove side is invalid")


def regulation_dead_corner_lanes() -> tuple[RegulationDeadCornerLane, ...]:
    """Return the frozen bilateral lanes selected from S92 failure replay."""

    sources = expanded_dynamic_corner_lanes()
    left_source = replace(
        sources[0],
        takeoff_config=replace(
            sources[0].takeoff_config,
            lunge_config=replace(
                sources[0].takeoff_config.lunge_config,
                lower_body_scale=0.83,
            ),
        ),
    )
    right_source = replace(
        sources[-1],
        takeoff_config=replace(
            sources[-1].takeoff_config,
            lunge_config=replace(
                sources[-1].takeoff_config.lunge_config,
                lower_body_scale=0.86,
            ),
        ),
    )
    return (
        RegulationDeadCornerLane(
            lane_id="left-post",
            label="LEFT REGULATION POST · AIRBORNE GLOVE SAVE",
            source_lane=left_source,
            world_translation_m=-3.05,
            target_y_m=-3.35,
            expected_glove_side="left",
        ),
        RegulationDeadCornerLane(
            lane_id="right-post",
            label="RIGHT REGULATION POST · AIRBORNE GLOVE SAVE",
            source_lane=right_source,
            world_translation_m=2.84,
            target_y_m=3.35,
            expected_glove_side="right",
        ),
    )


@dataclass(frozen=True)
class RegulationDeadCornerConfig:
    """Fail-closed contract for two regulation lateral dead corners."""

    lanes: tuple[RegulationDeadCornerLane, ...] = field(
        default_factory=regulation_dead_corner_lanes
    )
    maximum_post_surface_clearance_m: float = 0.15
    minimum_baseline_crossing_height_m: float = 1.20
    minimum_glove_contact_abs_y_m: float = 3.25
    minimum_glove_contact_height_m: float = 1.35
    maximum_glove_surface_separation_m: float = 0.001
    maximum_glove_surface_penetration_m: float = 0.018
    minimum_keeper_pelvis_height_m: float = 0.64
    minimum_contact_span_m: float = 6.50
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.regulation_dead_corner_config.v1"

    def __post_init__(self) -> None:
        if len(self.lanes) != 2 or {lane.expected_glove_side for lane in self.lanes} != {
            "left",
            "right",
        }:
            raise ValueError("dead-corner exam requires one lane per glove side")
        values = (
            self.maximum_post_surface_clearance_m,
            self.minimum_baseline_crossing_height_m,
            self.minimum_glove_contact_abs_y_m,
            self.minimum_glove_contact_height_m,
            self.maximum_glove_surface_separation_m,
            self.maximum_glove_surface_penetration_m,
            self.minimum_keeper_pelvis_height_m,
            self.minimum_contact_span_m,
        )
        if not all(math.isfinite(value) and value >= 0.0 for value in values):
            raise ValueError("dead-corner gates must be finite and non-negative")
        if not 0.05 <= self.maximum_post_surface_clearance_m <= 0.20:
            raise ValueError("dead-corner post clearance is invalid")
        if not 1.0 <= self.minimum_baseline_crossing_height_m <= 1.60:
            raise ValueError("dead-corner crossing-height gate is invalid")
        if not 3.0 <= self.minimum_glove_contact_abs_y_m <= 3.50:
            raise ValueError("dead-corner contact lateral gate is invalid")
        if not 1.25 <= self.minimum_glove_contact_height_m <= 1.80:
            raise ValueError("dead-corner contact-height gate is invalid")
        if not 0.0 <= self.maximum_glove_surface_separation_m <= 0.002:
            raise ValueError("dead-corner glove separation gate is invalid")
        if not 0.002 <= self.maximum_glove_surface_penetration_m <= 0.020:
            raise ValueError("dead-corner glove penetration gate is invalid")
        if not 0.60 <= self.minimum_keeper_pelvis_height_m <= 0.75:
            raise ValueError("dead-corner keeper stability gate is invalid")
        if not 6.0 <= self.minimum_contact_span_m <= 7.0:
            raise ValueError("dead-corner bilateral span gate is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("dead-corner evidence must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def regulation_dead_corner_lane_kwargs(
    *,
    lane: RegulationDeadCornerLane,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
) -> dict[str, Any]:
    """Place a qualified local save at a real regulation post."""

    kwargs = dynamic_corner_lane_kwargs(
        lane=lane.source_lane,
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        dive_source_checkout=dive_source_checkout,
    )
    goal = kwargs.get("goal_spec")
    goalkeeper = kwargs.get("goalkeeper_config")
    if goal is None or not isinstance(goalkeeper, G1GoalkeeperConfig):
        raise RuntimeError("dead-corner parent configuration is incomplete")
    if not math.isclose(goal.width_m, 7.32) or not math.isclose(goal.height_m, 2.44):
        raise RuntimeError("dead-corner exam requires a regulation goal")
    translated: dict[str, tuple[float, float, float]] = {}
    for name in ("shooter_origin", "passer_origin", "pass_reception_target_m"):
        value = kwargs.get(name)
        if not isinstance(value, tuple) or len(value) != 3:
            raise RuntimeError(f"dead-corner {name} is incomplete")
        translated[name] = (
            float(value[0]),
            float(value[1] + lane.world_translation_m),
            float(value[2]),
        )
    kwargs.update(
        goal_spec=replace(goal, target_y_m=lane.target_y_m),
        shooter_target=(goal.plane_x_m, lane.target_y_m, goal.target_z_m),
        goalkeeper_config=replace(
            goalkeeper,
            initial_lateral_position_m=lane.world_translation_m,
            regulation_goal_positioning_enabled=True,
            joint_guard_margin_rad=0.10,
            joint_guard_prediction_horizon_sec=0.20,
            joint_guard_boundary_kp=200.0,
            joint_guard_boundary_kd=20.0,
            joint_guard_impact_lead_sec=0.08,
            # This frozen exam translates an already qualified local save.
            # The canonical mirror remains an independently tested growth
            # feature, but is not allowed to perturb the strict replay.
            canonical_locomotion_mirror_enabled=False,
        ),
        **translated,
    )
    return kwargs


def _post_surface_clearance_m(*, result: G1SharedWorldResult, goal: Any) -> float:
    if result.goal_crossing_y_m is None:
        return math.inf
    return float(
        goal.width_m / 2.0
        - goal.post_radius_m
        - goal.ball_radius_m
        - abs(result.goal_crossing_y_m)
    )


def evaluate_dead_corner_baseline(
    *,
    result: G1SharedWorldResult,
    goal: Any,
    config: RegulationDeadCornerConfig,
) -> dict[str, Any]:
    """Prove the unopposed physical shot is a legal post-hugging goal."""

    clearance = _post_surface_clearance_m(result=result, goal=goal)
    crossing_z = result.goal_crossing_z_m
    gates = {
        "finite_state": result.finite_state,
        "ordered_pass_and_shot": bool(
            result.pass_contact_time_sec is not None
            and result.shot_contact_time_sec is not None
            and result.pass_contact_time_sec < result.shot_contact_time_sec
        ),
        "precise_pass": bool(
            result.pass_delivery_error_m is not None
            and result.pass_delivery_error_m <= 0.05
            and result.pass_delivery_lateral_error_m is not None
            and result.pass_delivery_lateral_error_m <= 0.03
        ),
        "unopposed_goal": bool(result.goal_plane_crossed and result.goal_crossed),
        "regulation_post_dead_corner": bool(
            0.0 <= clearance <= config.maximum_post_surface_clearance_m
        ),
        "raised_shot": bool(
            crossing_z is not None
            and config.minimum_baseline_crossing_height_m
            <= crossing_z
            <= goal.height_m - goal.post_radius_m - goal.ball_radius_m
        ),
        "attacker_stability": bool(
            result.passer_min_pelvis_height_m >= 0.60
            and result.shooter_min_pelvis_height_m >= 0.60
        ),
        "joint_limits": not result.joint_limit_violation,
        "torque_limits": not result.torque_limit_violation,
        "zero_actuator_saturation": not result.actuator_saturation,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "goal_crossing_y_m": result.goal_crossing_y_m,
        "goal_crossing_z_m": crossing_z,
        "post_surface_clearance_m": clearance,
        "result": result.to_dict(),
    }


def evaluate_dead_corner_save(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    lane: RegulationDeadCornerLane,
    config: RegulationDeadCornerConfig,
) -> dict[str, Any]:
    """Require physical glove authority plus the inherited airborne exam."""

    takeoff = evaluate_dynamic_takeoff_save(
        result=result,
        trajectory=trajectory,
        config=lane.source_lane.takeoff_config,
    )
    position = result.goalkeeper_glove_contact_position_m
    surface = result.goalkeeper_glove_contact_surface_distance_m
    gates = {
        "inherited_airborne_save": takeoff.get("passed") is True,
        "anatomical_expected_glove": bool(
            result.goalkeeper_glove_contact_side == lane.expected_glove_side
            and (
                result.goalkeeper_left_glove_contact_observed
                or result.goalkeeper_right_glove_contact_observed
            )
        ),
        "collision_faithful_glove": bool(
            surface is not None
            and -config.maximum_glove_surface_penetration_m
            <= surface
            <= config.maximum_glove_surface_separation_m
        ),
        "post_zone_high_contact": bool(
            position is not None
            and abs(position[1]) >= config.minimum_glove_contact_abs_y_m
            and position[2] >= config.minimum_glove_contact_height_m
        ),
        "physical_save": bool(result.goalkeeper_save_observed and not result.goal_crossed),
        "keeper_stability": bool(
            result.goalkeeper_min_pelvis_height_m is not None
            and result.goalkeeper_min_pelvis_height_m
            >= config.minimum_keeper_pelvis_height_m
        ),
        "joint_limits": not result.joint_limit_violation,
        "torque_limits": not result.torque_limit_violation,
        "zero_actuator_saturation": not result.actuator_saturation,
    }
    return {
        "passed": bool(all(gates.values())),
        "gates": gates,
        "takeoff_exam": takeoff,
        "glove_contact_position_m": position,
        "glove_surface_distance_m": surface,
        "result": result.to_dict(),
    }


def run_regulation_dead_corner_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    output_dir: Path,
    source_checkout: Path,
    config: RegulationDeadCornerConfig | None = None,
) -> dict[str, Any]:
    """Run unopposed/save strict replays for both regulation posts."""

    active = config or RegulationDeadCornerConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    dive_source = dive_source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("dead-corner evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    paths = (striker_actor_path, goalkeeper_actor_path, gmt_model_path, gmt_skill_path)
    if any(not path.expanduser().resolve().is_file() for path in paths):
        raise ValueError("dead-corner artifacts must be readable files")
    if not dive_source.is_dir() or not (dive_source / "LICENSE").is_file():
        raise ValueError("dead-corner dive source is incomplete")
    lane_kwargs = {
        lane.lane_id: regulation_dead_corner_lane_kwargs(
            lane=lane,
            striker_actor_path=striker_actor_path,
            goalkeeper_actor_path=goalkeeper_actor_path,
            gmt_model_path=gmt_model_path,
            gmt_skill_path=gmt_skill_path,
            dive_source_checkout=dive_source,
        )
        for lane in active.lanes
    }
    artifacts = {
        "striker_actor_hash": hash_bytes(striker_actor_path.read_bytes()),
        "goalkeeper_actor_hash": hash_bytes(goalkeeper_actor_path.read_bytes()),
        "gmt_model_hash": hash_bytes(gmt_model_path.read_bytes()),
        "gmt_skill_hash": hash_bytes(gmt_skill_path.read_bytes()),
        "dive_source_commit": _git_head(dive_source),
        "dive_source_license_hash": hash_bytes((dive_source / "LICENSE").read_bytes()),
    }
    request = {
        "schema_version": "rosclaw_soccer.regulation_dead_corner_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "goal_specs": {
            lane_id: asdict(kwargs["goal_spec"]) for lane_id, kwargs in lane_kwargs.items()
        },
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "artifacts": artifacts,
        "source_commit": _git_head(checkout),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
    }
    output.mkdir(parents=True)
    _write_json(output / "request.json", request)
    cases: dict[str, Any] = {}
    contact_y: list[float] = []
    for lane in active.lanes:
        save_kwargs = lane_kwargs[lane.lane_id]
        baseline_kwargs = dict(save_kwargs)
        baseline_kwargs["goalkeeper_config"] = None
        baseline_first_result, baseline_first_trajectory = simulate_shared_world(
            asset_root, **baseline_kwargs
        )
        baseline_replay_result, baseline_replay_trajectory = simulate_shared_world(
            asset_root, **baseline_kwargs
        )
        save_first_result, save_first_trajectory = simulate_shared_world(
            asset_root, **save_kwargs
        )
        save_replay_result, save_replay_trajectory = simulate_shared_world(
            asset_root, **save_kwargs
        )
        goal = save_kwargs["goal_spec"]
        baseline_first = evaluate_dead_corner_baseline(
            result=baseline_first_result, goal=goal, config=active
        )
        baseline_replay = evaluate_dead_corner_baseline(
            result=baseline_replay_result, goal=goal, config=active
        )
        save_first = evaluate_dead_corner_save(
            result=save_first_result,
            trajectory=save_first_trajectory,
            lane=lane,
            config=active,
        )
        save_replay = evaluate_dead_corner_save(
            result=save_replay_result,
            trajectory=save_replay_trajectory,
            lane=lane,
            config=active,
        )
        baseline_strict = bool(
            baseline_first_result.to_dict() == baseline_replay_result.to_dict()
            and trajectory_digest(baseline_first_trajectory)
            == trajectory_digest(baseline_replay_trajectory)
        )
        save_strict = bool(
            save_first_result.to_dict() == save_replay_result.to_dict()
            and trajectory_digest(save_first_trajectory)
            == trajectory_digest(save_replay_trajectory)
        )
        baseline_path = output / f"{lane.lane_id}-unopposed-trajectory.npz"
        save_path = output / f"{lane.lane_id}-save-trajectory.npz"
        np.savez_compressed(baseline_path, **baseline_replay_trajectory)  # type: ignore[arg-type]
        np.savez_compressed(save_path, **save_replay_trajectory)  # type: ignore[arg-type]
        position = save_replay_result.goalkeeper_glove_contact_position_m
        if position is not None:
            contact_y.append(float(position[1]))
        passed = bool(
            baseline_first.get("passed")
            and baseline_replay.get("passed")
            and save_first.get("passed")
            and save_replay.get("passed")
            and baseline_strict
            and save_strict
        )
        cases[lane.lane_id] = {
            "lane": asdict(lane),
            "passed": passed,
            "baseline_strict_replay": baseline_strict,
            "save_strict_replay": save_strict,
            "baseline_trajectory_file": baseline_path.name,
            "baseline_trajectory_hash": hash_bytes(baseline_path.read_bytes()),
            "save_trajectory_file": save_path.name,
            "save_trajectory_hash": hash_bytes(save_path.read_bytes()),
            "baseline_first": baseline_first,
            "baseline_replay": baseline_replay,
            "save_first": save_first,
            "save_replay": save_replay,
        }
    contact_span = max(contact_y) - min(contact_y) if len(contact_y) == len(active.lanes) else 0.0
    portfolio_gates = {
        "all_lanes_passed": all(case["passed"] for case in cases.values()),
        "all_unopposed_shots_score": all(
            case["baseline_replay"]["gates"]["unopposed_goal"] for case in cases.values()
        ),
        "all_saves_are_physical": all(
            case["save_replay"]["gates"]["physical_save"] for case in cases.values()
        ),
        "both_glove_sides": {
            case["save_replay"]["result"]["goalkeeper_glove_contact_side"]
            for case in cases.values()
        }
        == {"left", "right"},
        "bilateral_contact_span": contact_span >= active.minimum_contact_span_m,
    }
    passed = bool(all(portfolio_gates.values()))
    report = {
        "schema_version": "rosclaw_soccer.regulation_dead_corner_evidence.v1",
        "passed": passed,
        "promotion_status": "FROZEN_RESEARCH_DEMO" if passed else "REJECTED_DEVELOPMENT",
        "claim": _CLAIM,
        "portfolio_gates": portfolio_gates,
        "contact_span_m": contact_span,
        "cases": cases,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "single_shared_ball_per_case": True,
        "simultaneous_three_body_physics_per_save_case": True,
        "counterfactual_unopposed_baseline_per_lane": True,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "implementation_hash": _implementation_hash(),
        "artifacts": artifacts,
    }
    _write_json(output / "evidence.json", report)
    return report


def validate_regulation_dead_corner_evidence(path: Path) -> dict[str, Any]:
    """Validate frozen hashes and authority without trusting media."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    request = source.parent / "request.json"
    cases = payload.get("cases") if isinstance(payload, dict) else None
    gates = payload.get("portfolio_gates") if isinstance(payload, dict) else None
    if not (
        isinstance(payload, dict)
        and payload.get("schema_version")
        == "rosclaw_soccer.regulation_dead_corner_evidence.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "FROZEN_RESEARCH_DEMO"
        and payload.get("claim") == _CLAIM
        and payload.get("implementation_hash") == _implementation_hash()
        and payload.get("physics_authority") == "CPU_MUJOCO"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("pixels_used_for_scoring") is False
        and isinstance(cases, dict)
        and len(cases) == 2
        and isinstance(gates, dict)
        and all(gates.values())
        and request.is_file()
        and hash_bytes(request.read_bytes()) == payload.get("request_hash")
    ):
        raise ValueError("dead-corner evidence authority contract is invalid")
    for case in cases.values():
        if not isinstance(case, dict) or case.get("passed") is not True:
            raise ValueError("dead-corner evidence contains a rejected lane")
        for prefix in ("baseline", "save"):
            name = case.get(f"{prefix}_trajectory_file")
            if not isinstance(name, str) or Path(name).name != name:
                raise ValueError("dead-corner trajectory name is invalid")
            trajectory = source.parent / name
            if not trajectory.is_file() or hash_bytes(trajectory.read_bytes()) != case.get(
                f"{prefix}_trajectory_hash"
            ):
                raise ValueError("dead-corner trajectory binding changed")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    package = Path(__file__).parents[1]
    return str(
        hash_json(
            {
                "dead_corner": hash_bytes(Path(__file__).read_bytes()),
                "dynamic_corner": hash_bytes(
                    (package / "training" / "dynamic_corner_save.py").read_bytes()
                ),
                "takeoff_exam": hash_bytes(
                    (package / "training" / "dynamic_takeoff_exam.py").read_bytes()
                ),
                "shared_world": hash_bytes(
                    (package / "skills" / "team" / "shared_world.py").read_bytes()
                ),
            }
        )
    )


def _git_head(checkout: Path) -> str:
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


__all__ = [
    "RegulationDeadCornerConfig",
    "RegulationDeadCornerLane",
    "evaluate_dead_corner_baseline",
    "evaluate_dead_corner_save",
    "regulation_dead_corner_lane_kwargs",
    "regulation_dead_corner_lanes",
    "run_regulation_dead_corner_evidence",
    "validate_regulation_dead_corner_evidence",
]
