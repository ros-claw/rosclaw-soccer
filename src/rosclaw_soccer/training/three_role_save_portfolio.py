"""Strict multi-lane three-G1 aerial-save portfolio.

Every retained lane runs the complete pass -> front-G1 strike -> goalkeeper
contact chain in one CPU MuJoCo world with one immutable physical ball.  The
portfolio changes world-space attacking lanes while preserving each policy's
qualified local strike pocket; rendered pixels never participate in scoring.
"""

from __future__ import annotations

import json
import math
import subprocess
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any

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
from rosclaw_soccer.training.three_role_aerial_save import (
    ThreeRoleAerialSaveConfig,
    evaluate_three_role_aerial_save,
    three_role_aerial_save_kwargs,
)


@dataclass(frozen=True)
class ThreeRoleSaveLane:
    """One world-space attacking lane and goalkeeper starting pocket."""

    lane_id: str
    label: str
    attacker_lateral_offset_m: float
    goalkeeper_initial_lateral_m: float

    def __post_init__(self) -> None:
        if not self.lane_id or not self.lane_id.replace("-", "").isalnum():
            raise ValueError("save lane id must be non-empty kebab-case text")
        if not self.label or len(self.label) > 80:
            raise ValueError("save lane label must contain at most 80 characters")
        values = (self.attacker_lateral_offset_m, self.goalkeeper_initial_lateral_m)
        if not all(math.isfinite(value) and abs(value) <= 1.20 for value in values):
            raise ValueError("save lane offsets must be finite and inside +/-1.20 m")
        relative = self.attacker_lateral_offset_m - self.goalkeeper_initial_lateral_m
        if not -0.50 <= relative <= 0.50:
            raise ValueError("save lane must remain inside the qualified goalkeeper pocket")


_DEFAULT_LANES = (
    ThreeRoleSaveLane("right-channel", "+0.43 m · RIGHT-GLOVE SAVE", 0.0, 0.0),
    ThreeRoleSaveLane("center-channel", "CENTER-LINE · RIGHT-GLOVE SAVE", -0.45, -0.35),
    ThreeRoleSaveLane("left-channel", "-0.26 m · LEFT-GLOVE SAVE", -0.70, -0.25),
    ThreeRoleSaveLane("far-left-channel", "-0.46 m · RIGHT-GLOVE SAVE", -0.90, -0.90),
)


@dataclass(frozen=True)
class ThreeRoleSavePortfolioConfig:
    """Frozen multi-lane evidence contract selected from failure replay."""

    lanes: tuple[ThreeRoleSaveLane, ...] = _DEFAULT_LANES
    aerial_config: ThreeRoleAerialSaveConfig = ThreeRoleAerialSaveConfig()
    minimum_contact_span_m: float = 0.75
    minimum_adjacent_contact_separation_m: float = 0.15
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.three_role_save_portfolio_config.v2"

    def __post_init__(self) -> None:
        if len(self.lanes) < 3:
            raise ValueError("save portfolio requires at least three lanes")
        lane_ids = tuple(lane.lane_id for lane in self.lanes)
        if len(set(lane_ids)) != len(lane_ids):
            raise ValueError("save portfolio lane ids must be unique")
        if not 0.50 <= self.minimum_contact_span_m <= 2.0:
            raise ValueError("save portfolio contact span must be in [0.50, 2.0] m")
        if not 0.08 <= self.minimum_adjacent_contact_separation_m <= 0.50:
            raise ValueError("save portfolio contact separation must be in [0.08, 0.50] m")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("save portfolio is SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def three_role_save_lane_kwargs(
    *,
    lane: ThreeRoleSaveLane,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    aerial_config: ThreeRoleAerialSaveConfig | None = None,
) -> dict[str, Any]:
    """Translate one qualified local skill chain into a world-space lane."""

    active = aerial_config or ThreeRoleAerialSaveConfig()
    kwargs: dict[str, Any] = three_role_aerial_save_kwargs(
        striker_actor_path=striker_actor_path,
        goalkeeper_actor_path=goalkeeper_actor_path,
        gmt_model_path=gmt_model_path,
        gmt_skill_path=gmt_skill_path,
        config=active,
    )
    passer_origin = kwargs.get("passer_origin")
    goalkeeper = kwargs.get("goalkeeper_config")
    goal = kwargs.get("goal_spec")
    if (
        not isinstance(passer_origin, tuple)
        or len(passer_origin) != 3
        or not isinstance(goalkeeper, G1GoalkeeperConfig)
        or goal is None
    ):
        raise RuntimeError("three-role save portfolio parent configuration is incomplete")
    attacker_offset = lane.attacker_lateral_offset_m
    target_y = float(goal.target_y_m + attacker_offset)
    if abs(target_y) > goal.width_m / 2.0 - goal.post_radius_m:
        raise ValueError("save portfolio lane target leaves the regulation goal mouth")
    kwargs.update(
        goal_spec=replace(goal, target_y_m=target_y),
        shooter_target=(goal.plane_x_m, target_y, goal.target_z_m),
        # Policy target remains shooter-local; only the world lane changes.
        shooter_origin=(0.0, attacker_offset, 0.0),
        passer_origin=(
            float(passer_origin[0]),
            float(passer_origin[1] + attacker_offset),
            float(passer_origin[2]),
        ),
        pass_reception_target_m=(
            active.pass_reception_target_m[0],
            active.pass_reception_target_m[1] + attacker_offset,
            active.pass_reception_target_m[2],
        ),
        goalkeeper_config=replace(
            goalkeeper,
            initial_lateral_position_m=lane.goalkeeper_initial_lateral_m,
        ),
    )
    return kwargs


def evaluate_three_role_save_lane(
    *,
    result: G1SharedWorldResult,
    trajectory: dict[str, np.ndarray],
    config: ThreeRoleAerialSaveConfig,
) -> dict[str, Any]:
    """Add contact geometry to the strict S84 physical-state evaluation."""

    report: dict[str, Any] = evaluate_three_role_aerial_save(
        result=result,
        trajectory=trajectory,
        config=config,
    )
    if (
        result.goalkeeper_glove_contact_time_sec is None
        or result.goalkeeper_glove_contact_position_m is None
    ):
        return report
    report["glove_contact_position_m"] = result.goalkeeper_glove_contact_position_m
    report["glove_contact_time_sec"] = result.goalkeeper_glove_contact_time_sec
    report["glove_contact_side"] = result.goalkeeper_glove_contact_side
    report["glove_contact_surface_distance_m"] = (
        result.goalkeeper_glove_contact_surface_distance_m
    )
    return report


def run_three_role_save_portfolio_evidence(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    output_dir: Path,
    source_checkout: Path,
    config: ThreeRoleSavePortfolioConfig | None = None,
) -> dict[str, Any]:
    """Run every lane twice and freeze only a diverse all-pass portfolio."""

    active = config or ThreeRoleSavePortfolioConfig()
    output = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("save portfolio evidence must be new and outside the checkout")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    paths = (striker_actor_path, goalkeeper_actor_path, gmt_model_path, gmt_skill_path)
    if any(not path.expanduser().resolve().is_file() for path in paths):
        raise ValueError("save portfolio artifacts must be readable files")
    lane_kwargs = {
        lane.lane_id: three_role_save_lane_kwargs(
            lane=lane,
            striker_actor_path=striker_actor_path,
            goalkeeper_actor_path=goalkeeper_actor_path,
            gmt_model_path=gmt_model_path,
            gmt_skill_path=gmt_skill_path,
            aerial_config=active.aerial_config,
        )
        for lane in active.lanes
    }
    artifacts = {
        "striker_actor_hash": hash_bytes(striker_actor_path.read_bytes()),
        "goalkeeper_actor_hash": hash_bytes(goalkeeper_actor_path.read_bytes()),
        "gmt_model_hash": hash_bytes(gmt_model_path.read_bytes()),
        "gmt_skill_hash": hash_bytes(gmt_skill_path.read_bytes()),
    }
    request = {
        "schema_version": "rosclaw_soccer.three_role_save_portfolio_request.v1",
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
    }
    output.mkdir(parents=True)
    _write_json(output / "request.json", request)
    cases: dict[str, Any] = {}
    contact_positions_y: list[float] = []
    contact_sides: list[str] = []
    for lane in active.lanes:
        kwargs = lane_kwargs[lane.lane_id]
        first_result, first_trajectory = simulate_shared_world(asset_root, **kwargs)
        replay_result, replay_trajectory = simulate_shared_world(asset_root, **kwargs)
        first = evaluate_three_role_save_lane(
            result=first_result,
            trajectory=first_trajectory,
            config=active.aerial_config,
        )
        replay = evaluate_three_role_save_lane(
            result=replay_result,
            trajectory=replay_trajectory,
            config=active.aerial_config,
        )
        strict_replay = bool(
            first_result.to_dict() == replay_result.to_dict()
            and trajectory_digest(first_trajectory) == trajectory_digest(replay_trajectory)
        )
        trajectory_path = output / f"{lane.lane_id}-trajectory.npz"
        np.savez_compressed(trajectory_path, **replay_trajectory)  # type: ignore[arg-type]
        position = replay.get("glove_contact_position_m")
        if isinstance(position, tuple) and len(position) == 3:
            contact_positions_y.append(float(position[1]))
        side = replay.get("glove_contact_side")
        if isinstance(side, str):
            contact_sides.append(side)
        cases[lane.lane_id] = {
            "lane": asdict(lane),
            "passed": bool(first.get("passed") and replay.get("passed") and strict_replay),
            "strict_replay": strict_replay,
            "trajectory_file": trajectory_path.name,
            "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
            "first": first,
            "replay": replay,
        }
    ordered_y = sorted(contact_positions_y)
    span_m = ordered_y[-1] - ordered_y[0] if len(ordered_y) == len(active.lanes) else 0.0
    separations = tuple(
        right - left
        for left, right in zip(ordered_y[:-1], ordered_y[1:], strict=True)
    )
    portfolio_gates = {
        "all_lanes_passed": all(case["passed"] for case in cases.values()),
        "all_lanes_strict_replay": all(case["strict_replay"] for case in cases.values()),
        "contact_span": span_m >= active.minimum_contact_span_m,
        "contact_separation": bool(
            separations
            and min(separations) >= active.minimum_adjacent_contact_separation_m
        ),
        "both_sides_of_center": bool(ordered_y and ordered_y[0] < 0.0 < ordered_y[-1]),
        "both_glove_sides": {"left", "right"} <= set(contact_sides),
    }
    passed = bool(all(portfolio_gates.values()))
    report = {
        "schema_version": "rosclaw_soccer.three_role_save_portfolio_evidence.v1",
        "passed": passed,
        "promotion_status": "FROZEN_SIM_DEMO" if passed else "REJECTED_DEVELOPMENT",
        "portfolio_gates": portfolio_gates,
        "glove_contact_positions_y_m": tuple(contact_positions_y),
        "glove_contact_sides": tuple(contact_sides),
        "contact_span_m": span_m,
        "adjacent_contact_separations_m": separations,
        "case_count": len(cases),
        "cases": cases,
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "pixels_used_for_scoring": False,
        "simultaneous_three_body_physics_per_case": True,
        "single_shared_ball_per_case": True,
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "implementation_hash": _implementation_hash(),
        "artifacts": artifacts,
    }
    _write_json(output / "evidence.json", report)
    return report


def _implementation_hash() -> str:
    package = Path(__file__).parents[1]
    return str(
        hash_json(
            {
                "portfolio": hash_bytes(Path(__file__).read_bytes()),
                "aerial_save": hash_bytes(
                    (package / "training" / "three_role_aerial_save.py").read_bytes()
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
    "ThreeRoleSaveLane",
    "ThreeRoleSavePortfolioConfig",
    "evaluate_three_role_save_lane",
    "run_three_role_save_portfolio_evidence",
    "three_role_save_lane_kwargs",
]
