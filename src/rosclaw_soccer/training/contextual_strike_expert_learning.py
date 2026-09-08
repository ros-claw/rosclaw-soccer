"""Build a conservative contextual strike memory from verified S209 probes."""

from __future__ import annotations

import argparse
import json
import math
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.contextual_strike_experts import (
    ContextualStrikeExpert,
    ContextualStrikeExpertMemory,
    ContextualStrikeNegative,
    StrikeTaskContext,
    build_strike_task_context,
)
from rosclaw_soccer.growth.dynamic_strike_coordination import (
    DynamicStrikeCoordinationActor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.dynamic_strike_coordination_probe import (
    validate_dynamic_strike_coordination_probe,
)


def train_contextual_strike_expert_memory(
    *,
    success_probe_paths: tuple[Path, ...],
    failure_probe_paths: tuple[Path, ...],
    output_dir: Path,
    context_scale: tuple[float, ...] = (0.05, 0.04, 0.25, 0.08, 0.05),
    acceptance_radius: float = 0.75,
    negative_radius: float = 0.75,
) -> dict[str, Any]:
    """Persist one data-bound memory; source trajectories remain external."""

    root = output_dir.expanduser().resolve()
    checkout = Path(__file__).parents[3]
    if root == checkout or checkout in root.parents:
        raise ValueError("contextual strike memory must remain outside the checkout")
    if root.exists():
        raise FileExistsError("contextual strike memory output already exists")
    root.mkdir(parents=True)
    report = _derive_training_report(
        success_probe_paths=tuple(path.expanduser().resolve() for path in success_probe_paths),
        failure_probe_paths=tuple(path.expanduser().resolve() for path in failure_probe_paths),
        context_scale=context_scale,
        acceptance_radius=acceptance_radius,
        negative_radius=negative_radius,
    )
    report["report_hash"] = hash_json(report)
    _atomic_json(root / "training.json", report)
    return validate_contextual_strike_expert_memory(root)


def load_contextual_strike_expert_memory(
    artifact_dir: Path,
) -> tuple[ContextualStrikeExpertMemory, dict[str, Any]]:
    report = validate_contextual_strike_expert_memory(artifact_dir)
    memory = ContextualStrikeExpertMemory.from_mapping(report.get("memory"))
    if memory.memory_hash != report.get("memory_hash"):
        raise ValueError("contextual strike memory hash changed")
    return memory, report


def validate_contextual_strike_expert_memory(artifact_dir: Path) -> dict[str, Any]:
    root = artifact_dir.expanduser().resolve()
    source = root / "training.json"
    value = _read_json(source)
    declared = value.pop("report_hash", None)
    try:
        success_paths = value.get("success_probe_paths")
        failure_paths = value.get("failure_probe_paths")
        context_scale = value.get("context_scale")
        if (
            not isinstance(success_paths, list)
            or not isinstance(failure_paths, list)
            or not isinstance(context_scale, list)
            or not isinstance(value.get("acceptance_radius"), int | float)
            or not isinstance(value.get("negative_radius"), int | float)
        ):
            raise ValueError("contextual strike training sources are absent")
        expected = _derive_training_report(
            success_probe_paths=tuple(Path(path).expanduser().resolve() for path in success_paths),
            failure_probe_paths=tuple(Path(path).expanduser().resolve() for path in failure_paths),
            context_scale=tuple(float(item) for item in context_scale),
            acceptance_radius=float(value["acceptance_radius"]),
            negative_radius=float(value["negative_radius"]),
        )
        if declared != hash_json(value) or value != expected:
            raise ValueError("contextual strike training evidence or implementation changed")
    finally:
        value["report_hash"] = declared
    return value


def _derive_training_report(
    *,
    success_probe_paths: tuple[Path, ...],
    failure_probe_paths: tuple[Path, ...],
    context_scale: tuple[float, ...],
    acceptance_radius: float,
    negative_radius: float,
) -> dict[str, Any]:
    if len(success_probe_paths) < 3 or not failure_probe_paths:
        raise ValueError("contextual strike training needs successes and failures")
    success_rows = [_probe_row(path, require_success=True) for path in success_probe_paths]
    failure_rows = [_probe_row(path, require_success=False) for path in failure_probe_paths]
    targets = [float(row["target_y_m"]) for row in success_rows]
    if len(set(targets)) != len(targets):
        raise ValueError("contextual strike expert targets must be unique")
    ordered = sorted(success_rows, key=lambda row: float(row["target_y_m"]))
    experts = tuple(
        ContextualStrikeExpert(
            expert_id=f"target-{round(1000 * float(row['target_y_m'])):04d}",
            context_center=tuple(
                float(value) for value in cast(StrikeTaskContext, row["context"]).vector()
            ),
            actor=cast(DynamicStrikeCoordinationActor, row["actor"]),
            source_report_hash=str(row["report_hash"]),
            source_trajectory_digest=str(row["trajectory_digest"]),
        )
        for row in ordered
    )
    corrections: list[dict[str, Any]] = []
    unresolved: list[ContextualStrikeNegative] = []
    for index, failure in enumerate(failure_rows):
        target_y = float(failure["target_y_m"])
        ball_y = float(failure["ball_initial_y_m"])
        correction_index = next(
            (
                candidate
                for candidate, success in enumerate(ordered)
                if math.isclose(float(success["target_y_m"]), target_y, abs_tol=1.0e-12)
                and math.isclose(float(success["ball_initial_y_m"]), ball_y, abs_tol=1.0e-12)
            ),
            None,
        )
        if correction_index is None:
            unresolved.append(
                ContextualStrikeNegative(
                    boundary_id=f"unresolved-{index:02d}",
                    context_center=tuple(
                        float(value)
                        for value in cast(StrikeTaskContext, failure["context"]).vector()
                    ),
                    source_report_hash=str(failure["report_hash"]),
                    source_trajectory_digest=str(failure["trajectory_digest"]),
                )
            )
        corrections.append(
            {
                "failure_report_hash": failure["report_hash"],
                "failure_target_y_m": target_y,
                "failure_world_safe": failure["world_safe"],
                "failure_whole_ball_on_target": failure["whole_ball_on_target"],
                "corrected_by_expert_id": (
                    None if correction_index is None else experts[correction_index].expert_id
                ),
            }
        )
    memory = ContextualStrikeExpertMemory.build(
        experts=experts,
        negative_boundaries=tuple(unresolved),
        context_scale=context_scale,
        acceptance_radius=acceptance_radius,
        negative_radius=negative_radius,
    )
    gates = {
        "at_least_five_verified_experts": len(experts) >= 5,
        "target_span_at_least_20cm": max(targets) - min(targets) >= 0.20 - 1.0e-12,
        "dagger_failure_has_correction": any(
            row["corrected_by_expert_id"] is not None for row in corrections
        ),
        "unresolved_failure_becomes_negative_boundary": bool(unresolved),
        "all_source_actors_are_bounded_parameter_teachers": all(
            expert.actor.policy_type == "parameter"
            and expert.actor.activation_ceiling == "SIM_ONLY"
            and not expert.actor.hardware_authorized
            for expert in experts
        ),
    }
    if not all(gates.values()):
        raise ValueError("contextual strike training corpus is incomplete")
    source_paths = tuple((*success_probe_paths, *failure_probe_paths))
    source_files: dict[str, str] = {}
    for path in source_paths:
        report = _read_json(path)
        artifact = cast(dict[str, Any], report["trajectory_artifact"])
        trajectory_path = path.parent / str(artifact["file"])
        source_files[str(path)] = hash_bytes(path.read_bytes())
        source_files[str(trajectory_path)] = hash_bytes(trajectory_path.read_bytes())
    implementation_files = _implementation_files()
    return {
        "schema_version": "rosclaw_soccer.contextual_strike_expert_training.v1",
        "status": "TRAINED_SIM_ONLY_CONTEXTUAL_EXPERT_MEMORY",
        "success_probe_paths": [str(path) for path in success_probe_paths],
        "failure_probe_paths": [str(path) for path in failure_probe_paths],
        "context_scale": list(context_scale),
        "acceptance_radius": acceptance_radius,
        "negative_radius": negative_radius,
        "dagger_corrections": corrections,
        "gates": gates,
        "memory": memory.to_dict(),
        "memory_hash": memory.memory_hash,
        "dataset_snapshot_hash": memory.dataset_snapshot_hash,
        "source_files": source_files,
        "implementation_files": implementation_files,
        "implementation_hash": hash_json(implementation_files),
        "activation_ceiling": "SIM_ONLY",
        "promotion_eligible": False,
        "hardware_command_sent": False,
    }


def _probe_row(path: Path, *, require_success: bool) -> dict[str, Any]:
    raw = _read_json(path)
    if raw.get("schema_version") == "rosclaw_soccer.dynamic_strike_coordination_probe.v1":
        report = validate_dynamic_strike_coordination_probe(path)
    elif raw.get("schema_version") == "rosclaw_soccer.contextual_strike_expert_probe.v1":
        from rosclaw_soccer.training.contextual_strike_expert_probe import (
            validate_contextual_strike_expert_probe,
        )

        report = validate_contextual_strike_expert_probe(path)
    else:
        raise ValueError(f"unsupported contextual strike probe schema: {path}")
    assessment = cast(dict[str, Any], report["assessment"])
    gates = cast(dict[str, bool], assessment["gates"])
    success = bool(
        report.get("world_safe") is True
        and assessment.get("phase_sequence") == [1, 2, 3, 4, 5, 6]
        and report.get("shot_projection", {}).get("whole_ball_inside_goal") is True
        and gates.get("physical_teammate_pass_received") is True
        and gates.get("physical_foot_strike_in_strike_phase") is True
        and gates.get("stable_recovery_completed") is True
    )
    if success is not require_success:
        expected = "success" if require_success else "failure"
        raise ValueError(f"contextual strike source is not a verified {expected}: {path}")
    artifact = cast(dict[str, Any], report["trajectory_artifact"])
    trajectory_path = path.parent / str(artifact["file"])
    with np.load(trajectory_path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    context = _context_at_first_actor_frame(report, trajectory)
    actor = _actor_from_mapping(report.get("actor")) if require_success else None
    scenario = cast(dict[str, Any], report["scenario_spec"])
    return {
        "context": context,
        "actor": actor,
        "target_y_m": cast(dict[str, Any], report["goal_spec"])["target_y_m"],
        "ball_initial_y_m": scenario["ball_initial_position_m"][1],
        "world_safe": report["world_safe"],
        "whole_ball_on_target": cast(dict[str, Any], report["shot_projection"])[
            "whole_ball_inside_goal"
        ],
        "report_hash": report["report_hash"],
        "trajectory_digest": report["trajectory_digest"],
    }


def _context_at_first_actor_frame(
    report: dict[str, Any], trajectory: dict[str, NDArray[Any]]
) -> StrikeTaskContext:
    active = np.asarray(trajectory["strike_coordination_actor_active"], dtype=np.bool_)
    frames = np.flatnonzero(active)
    if len(frames) == 0:
        raise ValueError("contextual strike source has no active actor frame")
    frame = int(frames[0])
    agent_ids = tuple(str(item) for item in cast(list[Any], report["agent_ids"]))
    owner_code = int(trajectory["strike_phase_agent_code"][frame])
    if not 1 <= owner_code <= len(agent_ids):
        raise ValueError("contextual strike source owner code is invalid")
    owner_id = agent_ids[owner_code - 1]
    owner_team = owner_id.split(".", 1)[0]
    opponents = [agent_id for agent_id in agent_ids if agent_id.split(".", 1)[0] != owner_team]
    goal = cast(dict[str, Any], report["goal_spec"])
    return build_strike_task_context(
        goal_target_m=(float(goal["plane_x_m"]), float(goal["target_y_m"])),
        ball_position_m=np.asarray(trajectory["ball_pose"][frame, :2], dtype=np.float64),
        opponent_positions_m=np.asarray(
            [
                trajectory[agent_id.replace(".", "_") + "_pelvis_pose"][frame, :2]
                for agent_id in opponents
            ],
            dtype=np.float64,
        ),
    )


def _actor_from_mapping(raw: object) -> DynamicStrikeCoordinationActor:
    if not isinstance(raw, Mapping):
        raise ValueError("contextual strike source actor is invalid")
    values = dict(raw)
    for name in ("weights", "biases", "feature_mean", "feature_std"):
        value = values.get(name)
        if not isinstance(value, list):
            raise ValueError("contextual strike actor vector is invalid")
        values[name] = tuple(float(item) for item in value)
    return DynamicStrikeCoordinationActor(**values)


def _implementation_files() -> dict[str, str]:
    source = Path(__file__).parents[1]
    files = {
        "memory": source / "growth" / "contextual_strike_experts.py",
        "s209_actor": source / "growth" / "dynamic_strike_coordination.py",
        "s209_probe": source / "training" / "dynamic_strike_coordination_probe.py",
        "contextual_probe": source / "training" / "contextual_strike_expert_probe.py",
        "world": source / "skills" / "team" / "independent_team_world.py",
        "learning": Path(__file__),
    }
    return {name: hash_bytes(path.read_bytes()) for name, path in files.items()}


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"JSON artifact is not an object: {path}")
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
    parser.add_argument("--success-probe", action="append", required=True, type=Path)
    parser.add_argument("--failure-probe", action="append", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--acceptance-radius", default=0.75, type=float)
    parser.add_argument("--negative-radius", default=0.75, type=float)
    return parser


def main() -> None:
    arguments = _parser().parse_args()
    report = train_contextual_strike_expert_memory(
        success_probe_paths=tuple(arguments.success_probe),
        failure_probe_paths=tuple(arguments.failure_probe),
        output_dir=arguments.output_dir,
        acceptance_radius=arguments.acceptance_radius,
        negative_radius=arguments.negative_radius,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()


__all__ = [
    "load_contextual_strike_expert_memory",
    "train_contextual_strike_expert_memory",
    "validate_contextual_strike_expert_memory",
]
