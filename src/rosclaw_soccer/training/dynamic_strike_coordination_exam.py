"""Seal the S209 learned quick-strike comparison and contested outcome."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.role_self_model import TacticalIntent
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    build_continuous_competitive_fixture,
)
from rosclaw_soccer.training.dynamic_strike_coordination_learning import (
    load_dynamic_strike_coordination_actor_artifact,
)
from rosclaw_soccer.training.dynamic_strike_coordination_probe import (
    _project_shot,
    validate_dynamic_strike_coordination_probe,
)
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    validate_phase_conditioned_strike_growth,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec
from rosclaw_soccer.world.multi_player import build_g1_multi_player_stadium_model

_STATUS_PASS = "PASS_FIXED_CONTEXT_LEARNED_QUICK_STRIKE_BLOCK"
_STATUS_REJECTED = "REJECTED_FIXED_CONTEXT_LEARNED_QUICK_STRIKE_BLOCK"
_EXPECTED_PHASES = [1, 2, 3, 4, 5, 6]


def run_dynamic_strike_coordination_exam(
    *,
    evidence_dir: Path,
    asset_root: Path,
    parent_exam_path: Path,
    actor_artifact_dir: Path,
    learned_primary_probe_path: Path,
    learned_replay_probe_path: Path,
) -> dict[str, Any]:
    """Evaluate immutable source artifacts and persist one fail-closed report."""

    root = evidence_dir.expanduser().resolve()
    checkout = Path(__file__).parents[3]
    if root == checkout or checkout in root.parents:
        raise ValueError("dynamic strike exam evidence must remain outside the checkout")
    if root.exists() and any(root.iterdir()):
        raise ValueError("dynamic strike exam evidence directory must be empty")
    root.mkdir(parents=True, exist_ok=True)
    report = _evaluate(
        asset_root=asset_root.expanduser().resolve(),
        parent_exam_path=parent_exam_path.expanduser().resolve(),
        actor_artifact_dir=actor_artifact_dir.expanduser().resolve(),
        learned_primary_probe_path=learned_primary_probe_path.expanduser().resolve(),
        learned_replay_probe_path=learned_replay_probe_path.expanduser().resolve(),
    )
    report["report_hash"] = hash_json(report)
    destination = root / "dynamic-strike-coordination-exam.json"
    _atomic_json(destination, report)
    return validate_dynamic_strike_coordination_exam(destination)


def validate_dynamic_strike_coordination_exam(path: Path) -> dict[str, Any]:
    """Re-evaluate the source evidence, implementation, and promotion boundary."""

    resolved = path.expanduser().resolve()
    value = _read_json(resolved)
    declared = value.pop("report_hash", None)
    try:
        paths = value.get("source_paths")
        if not isinstance(paths, dict):
            raise ValueError("dynamic strike exam source paths are absent")
        expected = _evaluate(
            asset_root=Path(str(value.get("asset_root"))).expanduser().resolve(),
            parent_exam_path=Path(str(paths.get("parent_exam"))).expanduser().resolve(),
            actor_artifact_dir=Path(str(paths.get("actor_artifact"))).expanduser().resolve(),
            learned_primary_probe_path=Path(str(paths.get("learned_primary_probe")))
            .expanduser()
            .resolve(),
            learned_replay_probe_path=Path(str(paths.get("learned_replay_probe")))
            .expanduser()
            .resolve(),
        )
        if declared != hash_json(value) or value != expected:
            raise ValueError("dynamic strike exam evidence or implementation changed")
    finally:
        value["report_hash"] = declared
    return value


def _evaluate(
    *,
    asset_root: Path,
    parent_exam_path: Path,
    actor_artifact_dir: Path,
    learned_primary_probe_path: Path,
    learned_replay_probe_path: Path,
) -> dict[str, Any]:
    parent = validate_phase_conditioned_strike_growth(parent_exam_path)
    actor, learning = load_dynamic_strike_coordination_actor_artifact(actor_artifact_dir)
    primary = validate_dynamic_strike_coordination_probe(learned_primary_probe_path)
    replay = validate_dynamic_strike_coordination_probe(learned_replay_probe_path)
    parent_artifact = cast(dict[str, Any], parent["primary_artifact"])
    parent_trajectory_path = parent_exam_path.parent / str(parent_artifact["file"])
    primary_artifact = cast(dict[str, Any], primary["trajectory_artifact"])
    replay_artifact = cast(dict[str, Any], replay["trajectory_artifact"])
    primary_trajectory_path = learned_primary_probe_path.parent / str(primary_artifact["file"])
    replay_trajectory_path = learned_replay_probe_path.parent / str(replay_artifact["file"])
    parent_trajectory = _load_trajectory(parent_trajectory_path)
    primary_trajectory = _load_trajectory(primary_trajectory_path)
    goal = G1TrainingGoalSpec(**cast(dict[str, Any], primary["goal_spec"]))
    baseline_assessment = cast(dict[str, Any], parent["primary_assessment"])
    learned_assessment = cast(dict[str, Any], primary["assessment"])
    baseline_strike_frame = int(baseline_assessment["events"]["strike_frame"])
    baseline_projection = _projection_dict(
        trajectory=parent_trajectory,
        strike_frame=baseline_strike_frame,
        goal=goal,
    )
    learned_projection = cast(dict[str, Any], primary["shot_projection"])
    contact = _first_contested_contact(
        trajectory=primary_trajectory,
        strike_frame=int(learned_assessment["events"]["strike_frame"]),
        agent_ids=tuple(str(value) for value in primary["agent_ids"]),
        asset_root=asset_root,
        goal=goal,
    )
    actual_crossing = _actual_goal_crossing(primary_trajectory, goal=goal)
    exact_learned_replay = bool(
        primary["trajectory_digest"] == replay["trajectory_digest"]
        and primary["assessment"] == replay["assessment"]
        and primary["world_result"] == replay["world_result"]
        and primary["shot_projection"] == replay["shot_projection"]
    )
    baseline_metrics = cast(dict[str, Any], baseline_assessment["metrics"])
    learned_metrics = cast(dict[str, Any], learned_assessment["metrics"])
    baseline_gap = float(baseline_metrics["receive_to_strike_sec"])
    learned_gap = float(learned_metrics["receive_to_strike_sec"])
    baseline_speed = float(baseline_metrics["peak_shot_speed_mps"])
    learned_speed = float(learned_metrics["peak_shot_speed_mps"])
    baseline_error = float(cast(float, baseline_projection["target_error_m"]))
    learned_error = float(cast(float, learned_projection["target_error_m"]))
    improvement = {
        "receive_to_strike_reduction_sec": baseline_gap - learned_gap,
        "receive_to_strike_reduction_fraction": (baseline_gap - learned_gap) / baseline_gap,
        "peak_shot_speed_gain_mps": learned_speed - baseline_speed,
        "peak_shot_speed_gain_fraction": (learned_speed - baseline_speed) / baseline_speed,
        "ballistic_target_error_reduction_m": baseline_error - learned_error,
        "ballistic_target_error_reduction_fraction": (baseline_error - learned_error)
        / baseline_error,
    }
    learned_gates = cast(dict[str, bool], learned_assessment["gates"])
    world_result = cast(dict[str, Any], primary["world_result"])
    gates = {
        "parent_exact_replay": parent.get("exact_replay") is True,
        "learned_exact_replay": exact_learned_replay,
        "dataset_bound_learned_actor": bool(
            actor.policy_type == "learned_linear"
            and actor.dataset_snapshot_hash == learning.get("dataset_snapshot_hash")
            and actor.actor_hash == primary.get("actor_hash") == replay.get("actor_hash")
        ),
        "world_safe": primary.get("world_safe") is True and replay.get("world_safe") is True,
        "complete_phase_chain": learned_assessment.get("phase_sequence") == _EXPECTED_PHASES,
        "physical_pass_received": learned_gates.get("physical_teammate_pass_received") is True,
        "physical_foot_strike": learned_gates.get("physical_foot_strike_in_strike_phase") is True,
        "stable_recovery": learned_gates.get("stable_recovery_completed") is True,
        "ballistic_whole_ball_on_target": learned_projection.get("whole_ball_inside_goal") is True,
        "receive_to_strike_improved_200ms": improvement["receive_to_strike_reduction_sec"] >= 0.20,
        "peak_speed_improved_20pct": improvement["peak_shot_speed_gain_fraction"] >= 0.20,
        "target_error_improved_25pct": improvement["ballistic_target_error_reduction_fraction"]
        >= 0.25,
        "opponent_lower_leg_block_before_goal": bool(
            contact["opponent_contact"]
            and contact["lower_leg_contact"]
            and contact["ball_x_m"] < goal.plane_x_m
        ),
        "block_changed_velocity": float(contact["velocity_change_mps"]) >= 0.50,
        "no_robot_robot_contact": world_result.get("robot_robot_contact_count") == 0,
        "sim_only_no_hardware": bool(
            primary.get("activation_ceiling") == "SIM_ONLY"
            and replay.get("activation_ceiling") == "SIM_ONLY"
            and primary.get("hardware_command_sent") is False
            and replay.get("hardware_command_sent") is False
        ),
    }
    passed = bool(gates and all(gates.values()))
    source_paths = {
        "parent_exam": str(parent_exam_path),
        "actor_artifact": str(actor_artifact_dir),
        "learned_primary_probe": str(learned_primary_probe_path),
        "learned_replay_probe": str(learned_replay_probe_path),
    }
    source_files = {
        str(parent_exam_path): hash_bytes(parent_exam_path.read_bytes()),
        str(parent_trajectory_path): hash_bytes(parent_trajectory_path.read_bytes()),
        str(actor_artifact_dir / "learning.json"): hash_bytes(
            (actor_artifact_dir / "learning.json").read_bytes()
        ),
        str(actor_artifact_dir / "dataset.npz"): hash_bytes(
            (actor_artifact_dir / "dataset.npz").read_bytes()
        ),
        str(learned_primary_probe_path): hash_bytes(learned_primary_probe_path.read_bytes()),
        str(primary_trajectory_path): hash_bytes(primary_trajectory_path.read_bytes()),
        str(learned_replay_probe_path): hash_bytes(learned_replay_probe_path.read_bytes()),
        str(replay_trajectory_path): hash_bytes(replay_trajectory_path.read_bytes()),
    }
    implementation_files = _implementation_files()
    return {
        "schema_version": "rosclaw_soccer.dynamic_strike_coordination_exam.v1",
        "status": _STATUS_PASS if passed else _STATUS_REJECTED,
        "passed": passed,
        "promotion_eligible": False,
        "why_not_general_promotion": (
            "fixed-context exact replay passed, but cross-target learned evaluation did not"
            " satisfy the safe on-target gate"
        ),
        "asset_root": str(asset_root),
        "fixture_hash": parent["fixture_hash"],
        "actor_hash": actor.actor_hash,
        "dataset_snapshot_hash": actor.dataset_snapshot_hash,
        "learning_report_hash": learning["report_hash"],
        "parent_status": parent["status"],
        "parent_passed_under_current_environment": parent["passed"],
        "parent_exact_replay": parent["exact_replay"],
        "parent_rejection_context": (
            "current runtime records simultaneous goalkeeper glove and body contact; the frozen"
            " glove-dominance gate rejects it"
        ),
        "baseline": {
            "trajectory_digest": parent_artifact["trajectory_digest"],
            "assessment": baseline_assessment,
            "ballistic_projection": baseline_projection,
        },
        "learned": {
            "primary_trajectory_digest": primary["trajectory_digest"],
            "replay_trajectory_digest": replay["trajectory_digest"],
            "assessment": learned_assessment,
            "ballistic_projection": learned_projection,
            "first_contested_contact": contact,
            "actual_goal_crossing": actual_crossing,
            "exact_replay": exact_learned_replay,
        },
        "improvement": improvement,
        "gates": gates,
        "claim": {
            "qualified": "DATA_BOUND_QUICK_STRIKE_WITH_PHYSICAL_OPPONENT_SHIN_BLOCK",
            "not_claimed": [
                "GOAL",
                "GOALKEEPER_SAVE",
                "CROSS_TARGET_GENERALIZATION",
                "REAL_ROBOT_TRANSFER",
            ],
            "opponent_intent_at_contact": contact["opponent_intent"],
            "intentional_block_claimed": False,
        },
        "evidence_boundary": {
            "physics_authority": "CPU_MUJOCO",
            "activation_ceiling": "SIM_ONLY",
            "whole_body_g1_count": 6,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
        "source_paths": source_paths,
        "source_files": source_files,
        "implementation_files": implementation_files,
        "implementation_hash": hash_json(implementation_files),
    }


def _projection_dict(
    *, trajectory: dict[str, NDArray[Any]], strike_frame: int, goal: G1TrainingGoalSpec
) -> dict[str, float | int | bool | str | None]:
    y_value, z_value, error, sample_frame = _project_shot(
        trajectory=trajectory,
        strike_frame=strike_frame,
        goal_plane_x_m=goal.plane_x_m,
        target_y_m=goal.target_y_m,
        target_z_m=goal.target_z_m,
    )
    return {
        "method": "peak_forward_velocity_within_120ms_ballistic_v1",
        "sample_frame": sample_frame,
        "goal_plane_y_m": y_value,
        "goal_plane_z_m": z_value,
        "target_error_m": error,
        "whole_ball_inside_goal": bool(
            y_value is not None
            and z_value is not None
            and abs(y_value) <= goal.width_m / 2.0 - goal.ball_radius_m
            and goal.ball_radius_m <= z_value <= goal.height_m - goal.ball_radius_m
        ),
    }


def _first_contested_contact(
    *,
    trajectory: dict[str, NDArray[Any]],
    strike_frame: int,
    agent_ids: tuple[str, ...],
    asset_root: Path,
    goal: G1TrainingGoalSpec,
) -> dict[str, Any]:
    contact_agent = np.asarray(trajectory["ball_nonfoot_contact_agent_code"], dtype=np.int64)
    frames = np.flatnonzero((np.arange(len(contact_agent)) > strike_frame) & (contact_agent > 0))
    if len(frames) == 0:
        raise ValueError("learned quick strike has no contested physical contact")
    frame = int(frames[0])
    agent_code = int(contact_agent[frame])
    if not 1 <= agent_code <= len(agent_ids):
        raise ValueError("contested contact agent code is invalid")
    defender_id = agent_ids[agent_code - 1]
    shooter_code = int(trajectory["strike_phase_agent_code"][strike_frame])
    if not 1 <= shooter_code <= len(agent_ids):
        raise ValueError("strike owner code is invalid")
    shooter_id = agent_ids[shooter_code - 1]
    fixture = build_continuous_competitive_fixture(asset_root)
    model = build_g1_multi_player_stadium_model(asset_root, players=fixture.players, spec=goal)
    geom_id = int(trajectory["ball_nonfoot_contact_geom_id"][frame])
    geom_name = str(model.geom(geom_id).name)
    velocity = np.asarray(trajectory["ball_velocity"], dtype=np.float64)
    velocity_change = float(
        np.linalg.norm(
            velocity[min(len(velocity) - 1, frame + 1), :3] - velocity[max(0, frame - 1), :3]
        )
    )
    key = defender_id.replace(".", "_") + "_intent_code"
    intent_code = int(trajectory[key][frame])
    intents = tuple(TacticalIntent)
    intent = intents[intent_code].value if 0 <= intent_code < len(intents) else "unknown"
    ball_position = np.asarray(trajectory["ball_pose"][frame, :3], dtype=np.float64)
    lower_leg = any(token in geom_name for token in ("_shin", "_ankle", "_foot"))
    return {
        "frame": frame,
        "time_sec": float(trajectory["time"][frame]),
        "shooter_agent_id": shooter_id,
        "defender_agent_id": defender_id,
        "defender_geom_id": geom_id,
        "defender_geom_name": geom_name,
        "opponent_intent": intent,
        "contact_force_n": float(trajectory["ball_nonfoot_contact_force_n"][frame]),
        "velocity_change_mps": velocity_change,
        "ball_x_m": float(ball_position[0]),
        "opponent_contact": shooter_id.split(".", 1)[0] != defender_id.split(".", 1)[0],
        "lower_leg_contact": lower_leg,
    }


def _actual_goal_crossing(
    trajectory: dict[str, NDArray[Any]], *, goal: G1TrainingGoalSpec
) -> dict[str, float | int | bool | None]:
    pose = np.asarray(trajectory["ball_pose"], dtype=np.float64)
    frames = np.flatnonzero((pose[:-1, 0] < goal.plane_x_m) & (pose[1:, 0] >= goal.plane_x_m))
    if len(frames) == 0:
        return {"frame": None, "goal_plane_y_m": None, "goal_plane_z_m": None, "inside": False}
    frame = int(frames[0])
    denominator = float(pose[frame + 1, 0] - pose[frame, 0])
    alpha = 0.0 if abs(denominator) <= 1.0e-12 else (goal.plane_x_m - pose[frame, 0]) / denominator
    crossing = pose[frame, :3] + alpha * (pose[frame + 1, :3] - pose[frame, :3])
    inside = bool(
        abs(float(crossing[1])) <= goal.width_m / 2.0 - goal.ball_radius_m
        and goal.ball_radius_m <= float(crossing[2]) <= goal.height_m - goal.ball_radius_m
    )
    return {
        "frame": frame,
        "goal_plane_y_m": float(crossing[1]),
        "goal_plane_z_m": float(crossing[2]),
        "inside": inside,
    }


def _load_trajectory(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    if trajectory_digest(trajectory) == "":  # pragma: no cover - defensive type guard
        raise ValueError("trajectory digest is unavailable")
    return trajectory


def _implementation_files() -> dict[str, str]:
    root = Path(__file__).parents[1]
    files = {
        "actor": root / "growth" / "dynamic_strike_coordination.py",
        "assessment": root / "growth" / "phase_conditioned_strike_assessment.py",
        "world": root / "skills" / "team" / "independent_team_world.py",
        "learning": root / "training" / "dynamic_strike_coordination_learning.py",
        "probe": root / "training" / "dynamic_strike_coordination_probe.py",
        "exam": Path(__file__),
    }
    return {name: hash_bytes(path.read_bytes()) for name, path in files.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("dynamic strike exam report must be an object")
    return value


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        dir=path.parent, suffix=".json", mode="w", encoding="utf-8", delete=False
    ) as stream:
        temporary = Path(stream.name)
        try:
            json.dump(value, stream, ensure_ascii=False, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--evidence-dir", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--parent-exam", required=True, type=Path)
    parser.add_argument("--actor-artifact", required=True, type=Path)
    parser.add_argument("--learned-primary-probe", required=True, type=Path)
    parser.add_argument("--learned-replay-probe", required=True, type=Path)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = run_dynamic_strike_coordination_exam(
        evidence_dir=arguments.evidence_dir,
        asset_root=arguments.asset_root,
        parent_exam_path=arguments.parent_exam,
        actor_artifact_dir=arguments.actor_artifact,
        learned_primary_probe_path=arguments.learned_primary_probe,
        learned_replay_probe_path=arguments.learned_replay_probe,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "run_dynamic_strike_coordination_exam",
    "validate_dynamic_strike_coordination_exam",
]
