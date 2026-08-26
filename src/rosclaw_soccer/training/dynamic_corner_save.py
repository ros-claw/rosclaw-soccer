"""Strict multi-corner airborne-save portfolio for the shared three-G1 world."""

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
    simulate_shared_world,
)
from rosclaw_soccer.training.dynamic_aerial_lunge_save import (
    dynamic_aerial_lunge_kwargs,
)
from rosclaw_soccer.training.dynamic_takeoff_exam import (
    DynamicTakeoffExamConfig,
    evaluate_dynamic_takeoff_save,
    expanded_dynamic_takeoff_config,
)

_CLAIM = "STRICT_MULTI_CORNER_AIRBORNE_SAVE_PORTFOLIO"


@dataclass(frozen=True)
class DynamicCornerSaveLane:
    """One attack-chain translation and its independently gated takeoff expert."""

    lane_id: str
    label: str
    attacker_lateral_shift_m: float
    goalkeeper_initial_lateral_m: float
    takeoff_config: DynamicTakeoffExamConfig
    schema_version: str = "rosclaw_soccer.dynamic_corner_save_lane.v1"

    def __post_init__(self) -> None:
        if not self.lane_id or not self.lane_id.replace("-", "").isalnum():
            raise ValueError("dynamic corner lane id must be kebab-case text")
        if not self.label or len(self.label) > 96:
            raise ValueError("dynamic corner lane label is invalid")
        values = (self.attacker_lateral_shift_m, self.goalkeeper_initial_lateral_m)
        if not all(math.isfinite(value) for value in values):
            raise ValueError("dynamic corner lane geometry must be finite")
        if abs(self.attacker_lateral_shift_m) > 1.50:
            raise ValueError("dynamic corner attack shift exceeds the shared-world lane")
        if abs(self.goalkeeper_initial_lateral_m) > 1.20:
            raise ValueError("dynamic corner goalkeeper start is outside the goal pocket")


def expanded_dynamic_corner_lanes() -> tuple[DynamicCornerSaveLane, ...]:
    """Return four failure-selected lanes without weakening their safety gates."""

    base = expanded_dynamic_takeoff_config()
    left_inner = replace(
        base,
        lunge_config=replace(
            base.lunge_config,
            activation_lead_sec=0.37,
            lower_body_scale=0.776,
        ),
    )
    left_outer = replace(
        left_inner,
        lunge_config=replace(
            left_inner.lunge_config,
            outward_punch_force_scale=0.24,
            joint_guard_impact_lead_sec=0.04,
        ),
        # This harder edge lane retains 140 ms of contact-confirmed flight,
        # still 1.75x the S90 champion.  Every other takeoff/landing gate is
        # identical to S91 and remains independently evaluated.
        minimum_airborne_duration_sec=0.14,
    )
    right_outer = replace(
        base,
        lunge_config=replace(
            base.lunge_config,
            lower_body_scale=0.795,
            waist_scale=0.22,
            outward_punch_force_scale=0.30,
        ),
    )
    return (
        DynamicCornerSaveLane(
            "left-outer",
            "LEFT OUTER HIGH · OUTWARD TIP",
            -0.78,
            0.0,
            left_outer,
        ),
        DynamicCornerSaveLane(
            "left-inner",
            "LEFT HIGH · MIRRORED TAKEOFF",
            -0.70,
            0.0,
            left_inner,
        ),
        DynamicCornerSaveLane(
            "right-inner",
            "RIGHT HIGH · S91 CHAMPION",
            0.0,
            0.0,
            base,
        ),
        DynamicCornerSaveLane(
            "right-outer",
            "RIGHT OUTER HIGH · OUTWARD TIP",
            0.04,
            0.0,
            right_outer,
        ),
    )


@dataclass(frozen=True)
class DynamicCornerPortfolioConfig:
    """Promotion contract for diverse physical contact points on both sides."""

    lanes: tuple[DynamicCornerSaveLane, ...] = field(
        default_factory=expanded_dynamic_corner_lanes
    )
    minimum_contact_span_m: float = 0.75
    minimum_adjacent_contact_separation_m: float = 0.03
    minimum_left_contact_abs_y_m: float = 0.35
    minimum_right_contact_y_m: float = 0.44
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    commercial_use_allowed: bool = False
    schema_version: str = "rosclaw_soccer.dynamic_corner_portfolio_config.v1"

    def __post_init__(self) -> None:
        if len(self.lanes) < 4 or len({lane.lane_id for lane in self.lanes}) != len(self.lanes):
            raise ValueError("dynamic corner portfolio requires four unique lanes")
        values = (
            self.minimum_contact_span_m,
            self.minimum_adjacent_contact_separation_m,
            self.minimum_left_contact_abs_y_m,
            self.minimum_right_contact_y_m,
        )
        if not all(math.isfinite(value) and value > 0.0 for value in values):
            raise ValueError("dynamic corner portfolio coverage gates are invalid")
        if not 0.50 <= self.minimum_contact_span_m <= 2.0:
            raise ValueError("dynamic corner contact span is invalid")
        if not 0.02 <= self.minimum_adjacent_contact_separation_m <= 0.30:
            raise ValueError("dynamic corner contact separation is invalid")
        if (
            self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
            or self.commercial_use_allowed
        ):
            raise ValueError("dynamic corner portfolio must remain non-commercial SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def dynamic_corner_lane_kwargs(
    *,
    lane: DynamicCornerSaveLane,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
) -> dict[str, Any]:
    """Translate the whole attack chain while keeping its local strike pocket fixed."""

    kwargs = dynamic_aerial_lunge_kwargs(
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        dive_source_checkout=dive_source_checkout,
        config=lane.takeoff_config.lunge_config,
    )
    goal = kwargs.get("goal_spec")
    goalkeeper = kwargs.get("goalkeeper_config")
    if goal is None or not isinstance(goalkeeper, G1GoalkeeperConfig):
        raise RuntimeError("dynamic corner parent configuration is incomplete")
    shift = lane.attacker_lateral_shift_m
    target_y = float(goal.target_y_m + shift)
    shifted_goal = replace(goal, target_y_m=target_y)
    translated: dict[str, tuple[float, float, float]] = {}
    for name in ("shooter_origin", "passer_origin", "pass_reception_target_m"):
        value = kwargs.get(name)
        if not isinstance(value, tuple) or len(value) != 3:
            raise RuntimeError(f"dynamic corner {name} is incomplete")
        translated[name] = (float(value[0]), float(value[1] + shift), float(value[2]))
    kwargs.update(
        goal_spec=shifted_goal,
        shooter_target=(shifted_goal.plane_x_m, target_y, shifted_goal.target_z_m),
        goalkeeper_config=replace(
            goalkeeper,
            initial_lateral_position_m=lane.goalkeeper_initial_lateral_m,
        ),
        **translated,
    )
    return kwargs


def run_dynamic_corner_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    output_dir: Path,
    source_checkout: Path,
    config: DynamicCornerPortfolioConfig | None = None,
) -> dict[str, Any]:
    """Run every lane twice and freeze only an all-gate, diverse portfolio."""

    active = config or DynamicCornerPortfolioConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    dive_source = dive_source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("dynamic corner evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    paths = (striker_actor_path, goalkeeper_actor_path, gmt_model_path, gmt_skill_path)
    if any(not path.expanduser().resolve().is_file() for path in paths):
        raise ValueError("dynamic corner artifacts must be readable files")
    if not dive_source.is_dir() or not (dive_source / "LICENSE").is_file():
        raise ValueError("dynamic corner dive source is incomplete")
    lane_kwargs = {
        lane.lane_id: dynamic_corner_lane_kwargs(
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
        "schema_version": "rosclaw_soccer.dynamic_corner_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "lane_goal_specs": {
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
    contact_positions: list[tuple[str, float]] = []
    contact_sides: list[str] = []
    for lane in active.lanes:
        kwargs = lane_kwargs[lane.lane_id]
        first_result, first_trajectory = simulate_shared_world(asset_root, **kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        first = evaluate_dynamic_takeoff_save(
            result=first_result,
            trajectory=first_trajectory,
            config=lane.takeoff_config,
        )
        replay = evaluate_dynamic_takeoff_save(
            result=replay_result,
            trajectory=replay_trajectory,
            config=lane.takeoff_config,
        )
        strict_replay = bool(
            first_result.to_dict() == replay_result.to_dict()
            and trajectory_digest(first_trajectory) == trajectory_digest(replay_trajectory)
        )
        trajectory_path = output / f"{lane.lane_id}-trajectory.npz"
        np.savez_compressed(trajectory_path, **replay_trajectory)  # type: ignore[arg-type]
        position = replay_result.goalkeeper_glove_contact_position_m
        side = replay_result.goalkeeper_glove_contact_side
        if position is not None:
            contact_positions.append((lane.lane_id, float(position[1])))
        if side is not None:
            contact_sides.append(side)
        cases[lane.lane_id] = {
            "lane": asdict(lane),
            "passed": bool(first.get("passed") and replay.get("passed") and strict_replay),
            "strict_replay": strict_replay,
            "trajectory_file": trajectory_path.name,
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "result": replay_result.to_dict(),
            "first": first,
            "replay": replay,
        }
    ordered = sorted(value for _, value in contact_positions)
    span = ordered[-1] - ordered[0] if len(ordered) == len(active.lanes) else 0.0
    separations = tuple(
        upper - lower for lower, upper in zip(ordered[:-1], ordered[1:], strict=True)
    )
    portfolio_gates = {
        "all_lanes_passed": all(case["passed"] for case in cases.values()),
        "all_lanes_strict_replay": all(case["strict_replay"] for case in cases.values()),
        "all_contacts_observed": len(contact_positions) == len(active.lanes),
        "contact_span": span >= active.minimum_contact_span_m,
        "contact_separation": bool(
            separations
            and min(separations) >= active.minimum_adjacent_contact_separation_m
        ),
        "left_outer_contact": bool(
            ordered and ordered[0] <= -active.minimum_left_contact_abs_y_m
        ),
        "right_outer_contact": bool(
            ordered and ordered[-1] >= active.minimum_right_contact_y_m
        ),
        "both_glove_sides": {"left", "right"} <= set(contact_sides),
    }
    passed = bool(all(portfolio_gates.values()))
    report = {
        "schema_version": "rosclaw_soccer.dynamic_corner_evidence.v1",
        "passed": passed,
        "promotion_status": "FROZEN_RESEARCH_DEMO" if passed else "REJECTED_DEVELOPMENT",
        "claim": _CLAIM,
        "portfolio_gates": portfolio_gates,
        "case_count": len(cases),
        "cases": cases,
        "contact_positions_y_m": dict(contact_positions),
        "contact_sides": tuple(contact_sides),
        "contact_span_m": span,
        "adjacent_contact_separations_m": separations,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
        "single_shared_ball_per_case": True,
        "simultaneous_three_body_physics_per_case": True,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "implementation_hash": _implementation_hash(),
        "artifacts": artifacts,
    }
    _write_json(output / "evidence.json", report)
    return report


def validate_dynamic_corner_evidence(path: Path) -> dict[str, Any]:
    """Validate frozen source bindings and authority without rerunning physics."""

    source = path.expanduser().resolve()
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("dynamic corner evidence must be an object")
    request = source.parent / "request.json"
    cases = payload.get("cases")
    gates = payload.get("portfolio_gates")
    if not (
        payload.get("schema_version") == "rosclaw_soccer.dynamic_corner_evidence.v1"
        and payload.get("passed") is True
        and payload.get("promotion_status") == "FROZEN_RESEARCH_DEMO"
        and payload.get("claim") == _CLAIM
        and payload.get("physics_authority") == "CPU_MUJOCO"
        and payload.get("activation_ceiling") == "SIM_ONLY"
        and payload.get("hardware_command_sent") is False
        and payload.get("commercial_use_allowed") is False
        and payload.get("pixels_used_for_scoring") is False
        and isinstance(cases, dict)
        and len(cases) >= 4
        and isinstance(gates, dict)
        and all(gates.values())
        and request.is_file()
        and hash_bytes(request.read_bytes()) == payload.get("request_hash")
    ):
        raise ValueError("dynamic corner evidence authority contract is invalid")
    for value in cases.values():
        if not isinstance(value, dict) or value.get("passed") is not True:
            raise ValueError("dynamic corner evidence contains a rejected lane")
        name = value.get("trajectory_file")
        if not isinstance(name, str) or Path(name).name != name:
            raise ValueError("dynamic corner trajectory name is invalid")
        trajectory = source.parent / name
        if not trajectory.is_file() or hash_bytes(trajectory.read_bytes()) != value.get(
            "trajectory_hash"
        ):
            raise ValueError("dynamic corner trajectory binding changed")
    return cast(dict[str, Any], payload)


def _implementation_hash() -> str:
    package = Path(__file__).parents[1]
    return str(
        hash_json(
            {
                "corner_portfolio": hash_bytes(Path(__file__).read_bytes()),
                "takeoff_exam": hash_bytes(
                    (package / "training" / "dynamic_takeoff_exam.py").read_bytes()
                ),
                "dynamic_lunge": hash_bytes(
                    (package / "training" / "dynamic_aerial_lunge_save.py").read_bytes()
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
    "DynamicCornerPortfolioConfig",
    "DynamicCornerSaveLane",
    "dynamic_corner_lane_kwargs",
    "expanded_dynamic_corner_lanes",
    "run_dynamic_corner_evidence",
    "validate_dynamic_corner_evidence",
]
