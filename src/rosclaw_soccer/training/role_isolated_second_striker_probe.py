"""Role-isolated stability-plasticity probe for the continuous second striker.

The first pass, strike, save and recovery chain stays frozen.  Only a
target-conditioned second-striker contact candidate and the second football's
mass/friction may vary.  Candidate abstention falls back to the frozen parent;
the report distinguishes retained team competence from actual plasticity.
"""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from rosclaw.feedback import contracts as growth_core_contracts
from rosclaw.simforge.reproducibility import (
    ReproducibilityClosure,
    build_reproducibility_closure,
)

from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import simulate_shared_world
from rosclaw_soccer.training.continuous_second_striker_save_exam import (
    ContinuousSecondStrikerSaveExamConfig,
    evaluate_continuous_second_striker_save,
    physical_second_striker_kwargs,
)
from rosclaw_soccer.training.dynamic_corner_save import expanded_dynamic_corner_lanes

_CLAIM = "ROLE_ISOLATED_SECOND_STRIKER_STABILITY_PLASTICITY_PROBE"
_G1_ARTIFACTS = (
    ("g1-policy", Path("policy/robonaldo/model/policy-obs-aic.onnx")),
    ("g1-motion", Path("policy/robonaldo/model/freekick_motion.npz")),
    ("g1-scene", Path("g1_description/scene_with_ball.xml")),
    ("g1-model", Path("g1_description/g1_liao.xml")),
    ("g1-free-kick", Path("policy/robonaldo/FreeKick.py")),
)


@dataclass(frozen=True)
class RoleIsolatedSecondStrikerProbeConfig:
    lane_id: str = "left-inner"
    simulation_duration_sec: float = 23.0
    second_ball_mass_kg: float | None = None
    second_ball_ground_friction: float | None = None
    replay_count: int = 2
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.role_isolated_second_striker_probe_config.v1"

    def __post_init__(self) -> None:
        if self.lane_id not in {"left-inner", "left-outer", "right-inner", "right-outer"}:
            raise ValueError("role-isolated probe lane is unknown")
        if not 23.0 <= self.simulation_duration_sec <= 25.0:
            raise ValueError("role-isolated probe duration is invalid")
        if self.second_ball_mass_kg is not None and (
            not math.isfinite(self.second_ball_mass_kg)
            or not 0.40 <= self.second_ball_mass_kg <= 0.46
        ):
            raise ValueError("role-isolated second-ball mass is invalid")
        if self.second_ball_ground_friction is not None and (
            not math.isfinite(self.second_ball_ground_friction)
            or not 0.03 <= self.second_ball_ground_friction <= 0.80
        ):
            raise ValueError("role-isolated second-ball friction is invalid")
        if self.replay_count != 2:
            raise ValueError("role-isolated development probe requires two exact replays")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("role-isolated probe must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _atomic_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)


def _closure_sources(
    *,
    asset_root: Path,
    assets: dict[str, Path],
    candidate_actor: Path,
) -> tuple[dict[str, Path], dict[str, Path]]:
    source_trees = {
        "dive-source": assets["dive_source"],
        "rosclaw-core-reproducibility": Path(inspect.getfile(ReproducibilityClosure))
        .resolve()
        .parent,
        "rosclaw-core-runtime": Path(inspect.getfile(growth_core_contracts)).resolve().parents[2],
        "soccer": Path(__file__).resolve().parents[1],
    }
    artifacts = {
        key.replace("_", "-"): value for key, value in assets.items() if key != "dive_source"
    }
    artifacts["plastic-contact-candidate"] = candidate_actor
    artifacts.update({label: asset_root / relative for label, relative in _G1_ARTIFACTS})
    return source_trees, artifacts


def build_role_isolated_probe_closure(
    *,
    asset_root: Path,
    assets: dict[str, Path],
    candidate_actor: Path,
    expected_replays: int = 2,
) -> ReproducibilityClosure:
    source_trees, artifacts = _closure_sources(
        asset_root=asset_root,
        assets=assets,
        candidate_actor=candidate_actor,
    )
    return build_reproducibility_closure(
        source_trees=source_trees,
        dependency_packages=("mujoco", "numpy", "onnxruntime"),
        artifacts=artifacts,
        expected_replays=expected_replays,
    )


def _candidate_diagnostics(trajectory: dict[str, NDArray[Any]]) -> dict[str, Any]:
    conditioned = np.asarray(
        trajectory["second_striker_ballistic_actor_target_conditioned"], dtype=np.bool_
    )
    supported = np.asarray(
        trajectory["second_striker_ballistic_actor_launch_envelope_supported"],
        dtype=np.bool_,
    )
    selected = np.asarray(
        trajectory["second_striker_ballistic_actor_candidate_selected"], dtype=np.bool_
    )
    parent_or_candidate = np.asarray(
        trajectory["second_striker_ballistic_actor_active"], dtype=np.bool_
    )
    desired = np.asarray(
        trajectory["second_striker_ballistic_actor_desired_launch_velocity_yz_mps"],
        dtype=np.float64,
    )
    if (
        any(value.ndim != 1 for value in (conditioned, supported, selected, parent_or_candidate))
        or desired.shape != (conditioned.size, 2)
        or any(
            value.shape != conditioned.shape for value in (supported, selected, parent_or_candidate)
        )
        or np.any(selected & ~supported)
    ):
        raise ValueError("role-isolated candidate telemetry is invalid")
    observed = desired[conditioned]
    return {
        "conditioned_frame_count": int(np.count_nonzero(conditioned)),
        "supported_frame_count": int(np.count_nonzero(conditioned & supported)),
        "candidate_selected_frame_count": int(np.count_nonzero(selected)),
        "frozen_parent_selected_frame_count": int(
            np.count_nonzero(parent_or_candidate & ~selected)
        ),
        "desired_launch_velocity_yz_min_mps": (
            None if not observed.size else np.min(observed, axis=0).tolist()
        ),
        "desired_launch_velocity_yz_max_mps": (
            None if not observed.size else np.max(observed, axis=0).tolist()
        ),
    }


def _derive_probe_gates(
    replays: list[dict[str, Any]],
) -> tuple[dict[str, bool], dict[str, bool]]:
    if len(replays) != 2:
        raise ValueError("role-isolated gate derivation requires two replays")
    reference = {
        key: replays[0].get(key)
        for key in (
            "result",
            "evaluation",
            "candidate_diagnostics",
            "trajectory_digest",
            "trajectory_hash",
        )
    }
    strict_replay = all(
        {
            key: replay.get(key)
            for key in (
                "result",
                "evaluation",
                "candidate_diagnostics",
                "trajectory_digest",
                "trajectory_hash",
            )
        }
        == reference
        for replay in replays[1:]
    )
    evaluations = [cast(dict[str, Any], replay.get("evaluation", {})) for replay in replays]
    diagnostics = [
        cast(dict[str, Any], replay.get("candidate_diagnostics", {})) for replay in replays
    ]
    evidence_gates = {
        "strict_replay": strict_replay,
        "frozen_prefix_retained": all(
            cast(dict[str, Any], value.get("first_takeoff_exam", {})).get("passed") is True
            for value in evaluations
        ),
        "whole_world_safety": all(
            cast(dict[str, Any], value.get("gates", {})).get("whole_world_safety") is True
            for value in evaluations
        ),
        "candidate_attempt_observed": all(
            int(value.get("conditioned_frame_count", 0)) > 0 for value in diagnostics
        ),
    }
    plasticity_gates = {
        "candidate_envelope_supported": all(
            int(value.get("supported_frame_count", 0)) > 0 for value in diagnostics
        ),
        "candidate_selected": all(
            int(value.get("candidate_selected_frame_count", 0)) > 0 for value in diagnostics
        ),
        "complete_chain_passed": all(value.get("passed") is True for value in evaluations),
    }
    return evidence_gates, plasticity_gates


def _candidate_status(*, promoted: bool, plasticity_gates: dict[str, bool]) -> str:
    if promoted:
        return "QUALIFIED_DEVELOPMENT_CANDIDATE"
    if (
        plasticity_gates.get("candidate_envelope_supported") is True
        and plasticity_gates.get("candidate_selected") is True
        and plasticity_gates.get("complete_chain_passed") is False
    ):
        return "REJECTED_TASK_FAILURE"
    return "REJECTED_NO_SUPPORTED_PLASTICITY"


def run_role_isolated_second_striker_probe(
    *,
    asset_root: Path,
    striker_actor_path: Path,
    goalkeeper_actor_path: Path,
    gmt_model_path: Path,
    gmt_skill_path: Path,
    dive_source_checkout: Path,
    dive_athlete_checkpoint_path: Path,
    dive_athlete_exam_path: Path,
    recovery_athlete_checkpoint_path: Path,
    recovery_athlete_exam_path: Path,
    candidate_actor_path: Path,
    output_dir: Path,
    config: RoleIsolatedSecondStrikerProbeConfig | None = None,
) -> dict[str, Any]:
    active = config or RoleIsolatedSecondStrikerProbeConfig()
    root = asset_root.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    source_checkout = Path(__file__).resolve().parents[3]
    if (
        destination.exists()
        or destination == source_checkout
        or source_checkout in destination.parents
    ):
        raise ValueError("role-isolated probe requires a new external evidence directory")
    assets = {
        "striker_actor": striker_actor_path.expanduser().resolve(),
        "goalkeeper_actor": goalkeeper_actor_path.expanduser().resolve(),
        "gmt_model": gmt_model_path.expanduser().resolve(),
        "gmt_skill": gmt_skill_path.expanduser().resolve(),
        "dive_source": dive_source_checkout.expanduser().resolve(),
        "dive_athlete_checkpoint": dive_athlete_checkpoint_path.expanduser().resolve(),
        "dive_athlete_exam": dive_athlete_exam_path.expanduser().resolve(),
        "recovery_athlete_checkpoint": recovery_athlete_checkpoint_path.expanduser().resolve(),
        "recovery_athlete_exam": recovery_athlete_exam_path.expanduser().resolve(),
    }
    candidate = candidate_actor_path.expanduser().resolve()
    if not root.is_dir() or not candidate.is_file() or not assets["dive_source"].is_dir():
        raise FileNotFoundError("role-isolated probe input is missing")
    if any(not path.is_file() for key, path in assets.items() if key != "dive_source"):
        raise FileNotFoundError("role-isolated probe artifact is missing")
    closure = build_role_isolated_probe_closure(
        asset_root=root,
        assets=assets,
        candidate_actor=candidate,
        expected_replays=active.replay_count,
    )
    source_trees, artifacts = _closure_sources(
        asset_root=root,
        assets=assets,
        candidate_actor=candidate,
    )
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_second_striker_probe_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "reproducibility_closure": closure.to_dict(),
        "reproducibility_closure_hash": closure.closure_hash,
        "source_tree_locators": {key: str(value) for key, value in source_trees.items()},
        "artifact_locators": {key: str(value) for key, value in artifacts.items()},
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    destination.mkdir(parents=True)
    request_path = destination / "request.json"
    _atomic_json(request_path, request)
    lane = next(item for item in expanded_dynamic_corner_lanes() if item.lane_id == active.lane_id)
    exam = ContinuousSecondStrikerSaveExamConfig(
        lane_ids=(active.lane_id,), simulation_duration_sec=active.simulation_duration_sec
    )
    kwargs, goalkeeper, goal = physical_second_striker_kwargs(
        lane=lane,
        assets=assets,
        recovery_checkpoint=assets["recovery_athlete_checkpoint"],
        recovery_exam=assets["recovery_athlete_exam"],
        config=exam,
    )
    kwargs.update(
        second_striker_ballistic_actor_path=candidate,
        second_ball_mass_kg=active.second_ball_mass_kg,
        second_ball_ground_friction=active.second_ball_ground_friction,
    )
    replays: list[dict[str, Any]] = []
    for index in range(active.replay_count):
        result, trajectory = simulate_shared_world(root, **kwargs)
        evaluation = evaluate_continuous_second_striker_save(
            result=result,
            trajectory=trajectory,
            lane=lane,
            goal=goal,
            goalkeeper=goalkeeper,
            config=exam,
        )
        trajectory_path = destination / f"replay-{index}-trajectory.npz"
        _atomic_trajectory(trajectory_path, trajectory)
        replays.append(
            {
                "result": result.to_dict(),
                "evaluation": evaluation,
                "candidate_diagnostics": _candidate_diagnostics(trajectory),
                "trajectory_file": trajectory_path.name,
                "trajectory_hash": hash_bytes(trajectory_path.read_bytes()),
                "trajectory_digest": trajectory_digest(trajectory),
            }
        )
    evidence_gates, plasticity_gates = _derive_probe_gates(replays)
    evidence_passed = all(evidence_gates.values())
    candidate_promoted = evidence_passed and all(plasticity_gates.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_second_striker_probe_evidence.v1",
        "claim": _CLAIM,
        "evidence_passed": evidence_passed,
        "candidate_promoted": candidate_promoted,
        "candidate_status": _candidate_status(
            promoted=candidate_promoted, plasticity_gates=plasticity_gates
        ),
        "evidence_gates": evidence_gates,
        "plasticity_gates": plasticity_gates,
        "replays": replays,
        "request_hash": hash_bytes(request_path.read_bytes()),
        "reproducibility_closure_hash": closure.closure_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
        "commercial_use_allowed": False,
        "pixels_used_for_scoring": False,
    }
    report["report_hash"] = hash_json(report)
    evidence_path = destination / "evidence.json"
    _atomic_json(evidence_path, report)
    return validate_role_isolated_second_striker_probe(evidence_path)


def validate_role_isolated_second_striker_probe(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("role-isolated evidence must be an object")
    expected_hash = payload.pop("report_hash", None)
    try:
        request_path = resolved.parent / "request.json"
        request = json.loads(request_path.read_text(encoding="utf-8"))
        closure_value = request.get("reproducibility_closure")
        source_locators = request.get("source_tree_locators")
        artifact_locators = request.get("artifact_locators")
        if (
            not isinstance(closure_value, dict)
            or not isinstance(source_locators, dict)
            or not isinstance(artifact_locators, dict)
        ):
            raise ValueError("role-isolated closure request is invalid")
        recorded_closure = ReproducibilityClosure.from_dict(closure_value)
        raw_config = request.get("config")
        if not isinstance(raw_config, dict):
            raise ValueError("role-isolated request config is invalid")
        active = RoleIsolatedSecondStrikerProbeConfig(**raw_config)
        closure = build_reproducibility_closure(
            source_trees={key: Path(value) for key, value in source_locators.items()},
            dependency_packages=("mujoco", "numpy", "onnxruntime"),
            artifacts={key: Path(value) for key, value in artifact_locators.items()},
            expected_replays=2,
        )
        replays = payload.get("replays")
        if not isinstance(replays, list) or len(replays) != 2:
            raise ValueError("role-isolated replay set is incomplete")
        for replay in replays:
            if not isinstance(replay, dict):
                raise ValueError("role-isolated replay is invalid")
            trajectory = resolved.parent / str(replay.get("trajectory_file", ""))
            if not trajectory.is_file() or replay.get("trajectory_hash") != hash_bytes(
                trajectory.read_bytes()
            ):
                raise ValueError("role-isolated trajectory binding changed")
            with np.load(trajectory, allow_pickle=False) as archive:
                trajectory_value = {key: np.asarray(archive[key]) for key in archive.files}
            if replay.get("trajectory_digest") != trajectory_digest(trajectory_value) or replay.get(
                "candidate_diagnostics"
            ) != _candidate_diagnostics(trajectory_value):
                raise ValueError("role-isolated trajectory semantics changed")
        derived_evidence_gates, derived_plasticity_gates = _derive_probe_gates(replays)
        evidence_passed = all(derived_evidence_gates.values())
        candidate_promoted = evidence_passed and all(derived_plasticity_gates.values())
        if (
            request.get("schema_version")
            != "rosclaw_soccer.role_isolated_second_striker_probe_request.v1"
            or request.get("config_hash") != active.config_hash
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or recorded_closure != closure
            or request.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or payload.get("schema_version")
            != "rosclaw_soccer.role_isolated_second_striker_probe_evidence.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("evidence_gates") != derived_evidence_gates
            or payload.get("plasticity_gates") != derived_plasticity_gates
            or payload.get("evidence_passed") is not evidence_passed
            or payload.get("candidate_promoted") is not candidate_promoted
            or payload.get("candidate_status")
            != _candidate_status(
                promoted=candidate_promoted,
                plasticity_gates=derived_plasticity_gates,
            )
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or expected_hash != hash_json(payload)
        ):
            raise ValueError("role-isolated authority or integrity contract is invalid")
    finally:
        if expected_hash is not None:
            payload["report_hash"] = expected_hash
    return cast(dict[str, Any], payload)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--striker-actor", type=Path, required=True)
    parser.add_argument("--goalkeeper-actor", type=Path, required=True)
    parser.add_argument("--gmt-model", type=Path, required=True)
    parser.add_argument("--gmt-skill", type=Path, required=True)
    parser.add_argument("--dive-source", type=Path, required=True)
    parser.add_argument("--dive-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--dive-athlete-exam", type=Path, required=True)
    parser.add_argument("--recovery-athlete-checkpoint", type=Path, required=True)
    parser.add_argument("--recovery-athlete-exam", type=Path, required=True)
    parser.add_argument("--candidate-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--second-ball-mass-kg", type=float)
    parser.add_argument("--second-ball-ground-friction", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_role_isolated_second_striker_probe(
        asset_root=args.asset_root,
        striker_actor_path=args.striker_actor,
        goalkeeper_actor_path=args.goalkeeper_actor,
        gmt_model_path=args.gmt_model,
        gmt_skill_path=args.gmt_skill,
        dive_source_checkout=args.dive_source,
        dive_athlete_checkpoint_path=args.dive_athlete_checkpoint,
        dive_athlete_exam_path=args.dive_athlete_exam,
        recovery_athlete_checkpoint_path=args.recovery_athlete_checkpoint,
        recovery_athlete_exam_path=args.recovery_athlete_exam,
        candidate_actor_path=args.candidate_actor,
        output_dir=args.output_dir,
        config=RoleIsolatedSecondStrikerProbeConfig(
            second_ball_mass_kg=args.second_ball_mass_kg,
            second_ball_ground_friction=args.second_ball_ground_friction,
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("evidence_passed") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RoleIsolatedSecondStrikerProbeConfig",
    "build_role_isolated_probe_closure",
    "run_role_isolated_second_striker_probe",
    "validate_role_isolated_second_striker_probe",
]
