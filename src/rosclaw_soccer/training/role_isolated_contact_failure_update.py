"""S114 failure-memory update for the role-isolated contact actor.

This is deliberately an offline, fail-closed update.  It consumes immutable
teacher rehearsals plus a complete-chain candidate failure, contracts the
inverse-model step around the nearest successful rehearsal, and emits another
unqualified SIM-only actor.  Qualification remains the responsibility of a
fresh physics probe and its sealed holdout.
"""

from __future__ import annotations

import argparse
import inspect
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
from rosclaw.feedback import contracts as growth_core_contracts
from rosclaw.simforge.reproducibility import (
    ReproducibilityClosure,
    build_reproducibility_closure,
)

from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    load_g1_ballistic_contact_impulse_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.skills.team.shared_world import G1PhysicalSecondStrikerConfig
from rosclaw_soccer.training.role_isolated_contact_actor_growth import (
    RoleIsolatedContactActorGrowthConfig,
    _fit_candidate,
)
from rosclaw_soccer.training.role_isolated_second_striker_probe import (
    _candidate_diagnostics,
    _derive_probe_gates,
)

_CLAIM = "ROLE_ISOLATED_CONTACT_ACTOR_FAILURE_MEMORY_UPDATE"


@dataclass(frozen=True)
class RoleIsolatedContactFailureUpdateConfig:
    """Bounded trust-region response to one verified task-level failure."""

    local_plasticity_gain: float = 0.05
    proprioceptive_feedback_gain_n_per_mps: float = 6.0
    required_failed_gates: tuple[str, ...] = (
        "outward_physical_save",
        "final_goalkeeper_ready",
    )
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.role_isolated_contact_failure_update_config.v1"

    def __post_init__(self) -> None:
        if not 0.01 <= self.local_plasticity_gain <= 0.10:
            raise ValueError("failure update plasticity gain is outside the trust region")
        if not 5.0 <= self.proprioceptive_feedback_gain_n_per_mps <= 10.0:
            raise ValueError("failure update proprioceptive gain is invalid")
        if not self.required_failed_gates or len(set(self.required_failed_gates)) != len(
            self.required_failed_gates
        ):
            raise ValueError("failure update requires unique task-level failure gates")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("failure memory update must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"failure-memory input {path.name} must be an object")
    return cast(dict[str, Any], value)


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _verify_report_hash(payload: dict[str, Any], *, label: str) -> None:
    expected = payload.get("report_hash")
    body = dict(payload)
    body.pop("report_hash", None)
    if expected != hash_json(body):
        raise ValueError(f"frozen {label} report integrity changed")


def _verified_training_rows(path: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    payload = _read_object(path)
    _verify_report_hash(payload, label="teacher rehearsal")
    request_path = path.parent / "request.json"
    if payload.get("request_hash") != hash_bytes(request_path.read_bytes()):
        raise ValueError("frozen teacher request binding changed")
    rows_value = payload.get("probes")
    if not isinstance(rows_value, list) or len(rows_value) < 8:
        raise ValueError("frozen teacher rehearsal set is incomplete")
    rows: list[dict[str, Any]] = []
    for raw in rows_value:
        if not isinstance(raw, dict):
            raise ValueError("frozen teacher rehearsal row is invalid")
        row = dict(raw)
        replays = row.get("replays")
        if (
            row.get("strict_replay") is not True
            or not isinstance(replays, list)
            or len(replays) != 2
        ):
            raise ValueError("frozen teacher rehearsal has no exact replay pair")
        replay_semantics: list[dict[str, Any]] = []
        peak_foot_velocity: list[float] | None = None
        for replay in replays:
            if not isinstance(replay, dict):
                raise ValueError("frozen teacher replay is invalid")
            trajectory_path = path.parent / str(replay.get("trajectory_file", ""))
            if not trajectory_path.is_file() or replay.get("trajectory_hash") != hash_bytes(
                trajectory_path.read_bytes()
            ):
                raise ValueError("frozen teacher trajectory binding changed")
            with np.load(trajectory_path, allow_pickle=False) as archive:
                trajectory = {name: np.asarray(archive[name]) for name in archive.files}
            if replay.get("trajectory_digest") != trajectory_digest(trajectory):
                raise ValueError("frozen teacher trajectory semantics changed")
            active = np.asarray(trajectory["second_striker_loft_teacher_active"], dtype=np.bool_)
            force = np.asarray(
                trajectory["second_striker_loft_teacher_force_yz_n"], dtype=np.float64
            )[active]
            foot = np.asarray(
                trajectory["second_striker_loft_teacher_foot_velocity_yz_mps"],
                dtype=np.float64,
            )[active]
            if (
                force.ndim != 2
                or force.shape[1] != 2
                or foot.shape != force.shape
                or not force.size
            ):
                raise ValueError("frozen teacher contact telemetry is invalid")
            index = int(np.argmax(np.linalg.norm(force, axis=1)))
            current_peak = foot[index].tolist()
            if peak_foot_velocity is None:
                peak_foot_velocity = current_peak
            elif peak_foot_velocity != current_peak:
                raise ValueError("frozen teacher proprioception is not replay exact")
            replay_semantics.append(
                {
                    key: replay.get(key)
                    for key in (
                        "hard_safe",
                        "teacher_success",
                        "teacher_success_gates",
                        "launch_velocity_xyz_mps",
                        "teacher_peak_force_yz_n",
                        "result",
                        "evaluation",
                        "trajectory_hash",
                        "trajectory_digest",
                    )
                }
            )
        if replay_semantics[0] != replay_semantics[1] or peak_foot_velocity is None:
            raise ValueError("frozen teacher replay pair diverged")
        row["teacher_peak_foot_velocity_yz_mps"] = peak_foot_velocity
        rows.append(row)
    return payload, rows


def _verified_failure(path: Path) -> tuple[dict[str, Any], tuple[str, ...]]:
    payload = _read_object(path)
    _verify_report_hash(payload, label="candidate failure")
    request_path = path.parent / "request.json"
    if payload.get("request_hash") != hash_bytes(request_path.read_bytes()):
        raise ValueError("frozen failure request binding changed")
    replays = payload.get("replays")
    if not isinstance(replays, list) or len(replays) != 2:
        raise ValueError("frozen candidate failure replay set is incomplete")
    for replay in replays:
        if not isinstance(replay, dict):
            raise ValueError("frozen candidate failure replay is invalid")
        trajectory_path = path.parent / str(replay.get("trajectory_file", ""))
        if not trajectory_path.is_file() or replay.get("trajectory_hash") != hash_bytes(
            trajectory_path.read_bytes()
        ):
            raise ValueError("frozen candidate failure trajectory binding changed")
        with np.load(trajectory_path, allow_pickle=False) as archive:
            trajectory = {name: np.asarray(archive[name]) for name in archive.files}
        if (
            replay.get("trajectory_digest") != trajectory_digest(trajectory)
            or replay.get("candidate_diagnostics") != _candidate_diagnostics(trajectory)
        ):
            raise ValueError("frozen candidate failure trajectory semantics changed")
    evidence_gates, plasticity_gates = _derive_probe_gates(cast(list[dict[str, Any]], replays))
    if (
        not all(evidence_gates.values())
        or plasticity_gates.get("candidate_envelope_supported") is not True
        or plasticity_gates.get("candidate_selected") is not True
        or plasticity_gates.get("complete_chain_passed") is not False
    ):
        raise ValueError("failure memory is not a safe selected-candidate task failure")
    evaluations = [cast(dict[str, Any], replay["evaluation"]) for replay in replays]
    failed = tuple(
        sorted(
            key
            for key, passed in cast(dict[str, bool], evaluations[0]["gates"]).items()
            if not passed
        )
    )
    if any(
        tuple(
            sorted(
                key
                for key, passed in cast(dict[str, bool], value["gates"]).items()
                if not passed
            )
        )
        != failed
        for value in evaluations[1:]
    ):
        raise ValueError("failure gates are not replay exact")
    return payload, failed


def _closure_inputs(
    *, training: Path, failure: Path, parent_actor: Path
) -> tuple[dict[str, Path], dict[str, Path]]:
    source_trees = {
        "rosclaw-core-reproducibility": Path(inspect.getfile(ReproducibilityClosure))
        .resolve()
        .parent,
        "rosclaw-core-runtime": Path(inspect.getfile(growth_core_contracts)).resolve().parents[2],
        "soccer": Path(__file__).resolve().parents[1],
    }
    failure_request = _read_object(failure.parent / "request.json")
    failure_locators = failure_request.get("artifact_locators")
    if not isinstance(failure_locators, dict) or not isinstance(
        failure_locators.get("plastic-contact-candidate"), str
    ):
        raise ValueError("failed candidate artifact locator is absent")
    artifacts: dict[str, Path] = {
        "teacher-evidence": training,
        "teacher-request": training.parent / "request.json",
        "teacher-actor": training.parent / "role-isolated-contact-actor.json",
        "failed-candidate-evidence": failure,
        "failed-candidate-request": failure.parent / "request.json",
        "failed-contact-actor": Path(failure_locators["plastic-contact-candidate"]),
        "parent-contact-actor": parent_actor,
    }
    for index, path in enumerate(sorted(training.parent.glob("*-replay-*.npz"))):
        artifacts[f"teacher-trajectory-{index:02d}"] = path
    for index, path in enumerate(sorted(failure.parent.glob("replay-*-trajectory.npz"))):
        artifacts[f"failure-trajectory-{index:02d}"] = path
    return source_trees, artifacts


def _derive_actor(
    *,
    training_path: Path,
    failure_path: Path,
    parent_actor_path: Path,
    config: RoleIsolatedContactFailureUpdateConfig,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    training, rows = _verified_training_rows(training_path)
    failure, failed_gates = _verified_failure(failure_path)
    if not set(config.required_failed_gates) <= set(failed_gates):
        raise ValueError("candidate memory does not contain the required task failure")
    parent = load_g1_ballistic_contact_impulse_actor(parent_actor_path)
    context_hash = str(
        hash_json(
            {
                "teacher_report_hash": training["report_hash"],
                "failed_candidate_report_hash": failure["report_hash"],
                "parent_actor_hash": parent.actor_hash,
                "failure_update_config": asdict(config),
            }
        )
    )
    growth_config = RoleIsolatedContactActorGrowthConfig(
        local_plasticity_gain=config.local_plasticity_gain,
        proprioceptive_feedback_gain_n_per_mps=(
            config.proprioceptive_feedback_gain_n_per_mps
        ),
    )
    actor, fit = _fit_candidate(
        rows=rows,
        parent_actor=parent,
        striker=G1PhysicalSecondStrikerConfig(),
        goal_x_m=7.5,
        context_hash=context_hash,
        config=growth_config,
    )
    memory = {
        "teacher_report_hash": training["report_hash"],
        "failed_candidate_report_hash": failure["report_hash"],
        "failed_task_gates": list(failed_gates),
        "failure_response": "CONTRACT_INVERSE_STEP_AND_SLOW_PROPRIOCEPTIVE_FEEDBACK",
        "new_local_plasticity_gain": config.local_plasticity_gain,
        "new_proprioceptive_feedback_gain_n_per_mps": (
            config.proprioceptive_feedback_gain_n_per_mps
        ),
    }
    return actor, fit, memory


def run_role_isolated_contact_failure_update(
    *,
    training_evidence_path: Path,
    failed_candidate_evidence_path: Path,
    parent_actor_path: Path,
    output_dir: Path,
    config: RoleIsolatedContactFailureUpdateConfig | None = None,
) -> dict[str, Any]:
    active = config or RoleIsolatedContactFailureUpdateConfig()
    training = training_evidence_path.expanduser().resolve()
    failure = failed_candidate_evidence_path.expanduser().resolve()
    parent = parent_actor_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("failure update requires a new external evidence directory")
    if not all(path.is_file() for path in (training, failure, parent)):
        raise FileNotFoundError("failure update input artifact is missing")
    source_trees, artifacts = _closure_inputs(
        training=training, failure=failure, parent_actor=parent
    )
    closure = build_reproducibility_closure(
        source_trees=source_trees,
        dependency_packages=("numpy",),
        artifacts=artifacts,
        expected_replays=2,
    )
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_contact_failure_update_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_tree_locators": {key: str(value) for key, value in source_trees.items()},
        "artifact_locators": {key: str(value) for key, value in artifacts.items()},
        "reproducibility_closure": closure.to_dict(),
        "reproducibility_closure_hash": closure.closure_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    destination.mkdir(parents=True)
    request_path = destination / "request.json"
    _atomic_json(request_path, request)
    actor, fit, memory = _derive_actor(
        training_path=training,
        failure_path=failure,
        parent_actor_path=parent,
        config=active,
    )
    actor_path = destination / "failure-updated-contact-actor.json"
    _atomic_json(actor_path, actor.to_dict())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_contact_failure_update_evidence.v1",
        "claim": _CLAIM,
        "candidate_derived": True,
        "candidate_promoted": False,
        "candidate_status": "UNQUALIFIED_SIM_ONLY_FAILURE_UPDATED_CANDIDATE",
        "memory": memory,
        "fit": fit,
        "actor_file": actor_path.name,
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
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
    return validate_role_isolated_contact_failure_update(evidence_path)


def validate_role_isolated_contact_failure_update(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload = _read_object(resolved)
    expected_hash = payload.pop("report_hash", None)
    try:
        request_path = resolved.parent / "request.json"
        request = _read_object(request_path)
        config_value = request.get("config")
        source_value = request.get("source_tree_locators")
        artifact_value = request.get("artifact_locators")
        closure_value = request.get("reproducibility_closure")
        if not all(
            isinstance(value, dict)
            for value in (config_value, source_value, artifact_value, closure_value)
        ):
            raise ValueError("failure update request is incomplete")
        active = RoleIsolatedContactFailureUpdateConfig(
            **cast(dict[str, Any], config_value)
        )
        sources = {key: Path(value) for key, value in cast(dict[str, str], source_value).items()}
        artifacts = {
            key: Path(value) for key, value in cast(dict[str, str], artifact_value).items()
        }
        closure = build_reproducibility_closure(
            source_trees=sources,
            dependency_packages=("numpy",),
            artifacts=artifacts,
            expected_replays=2,
        )
        recorded = ReproducibilityClosure.from_dict(cast(dict[str, Any], closure_value))
        actor, fit, memory = _derive_actor(
            training_path=artifacts["teacher-evidence"],
            failure_path=artifacts["failed-candidate-evidence"],
            parent_actor_path=artifacts["parent-contact-actor"],
            config=active,
        )
        actor_path = resolved.parent / str(payload.get("actor_file", ""))
        stored_actor = load_g1_ballistic_contact_impulse_actor(actor_path)
        if (
            request.get("schema_version")
            != "rosclaw_soccer.role_isolated_contact_failure_update_request.v1"
            or request.get("config_hash") != active.config_hash
            or request.get("activation_ceiling") != "SIM_ONLY"
            or request.get("hardware_command_sent") is not False
            or recorded != closure
            or request.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("schema_version")
            != "rosclaw_soccer.role_isolated_contact_failure_update_evidence.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("candidate_derived") is not True
            or payload.get("candidate_promoted") is not False
            or payload.get("candidate_status")
            != "UNQUALIFIED_SIM_ONLY_FAILURE_UPDATED_CANDIDATE"
            or payload.get("memory") != memory
            or payload.get("fit") != fit
            or stored_actor.to_dict() != actor.to_dict()
            or payload.get("actor_hash") != actor.actor_hash
            or payload.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or payload.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or expected_hash != hash_json(payload)
        ):
            raise ValueError("failure update authority or integrity contract is invalid")
    finally:
        if expected_hash is not None:
            payload["report_hash"] = expected_hash
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-evidence", type=Path, required=True)
    parser.add_argument("--failed-candidate-evidence", type=Path, required=True)
    parser.add_argument("--parent-actor", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_role_isolated_contact_failure_update(
        training_evidence_path=args.training_evidence,
        failed_candidate_evidence_path=args.failed_candidate_evidence,
        parent_actor_path=args.parent_actor,
        output_dir=args.output_dir,
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RoleIsolatedContactFailureUpdateConfig",
    "run_role_isolated_contact_failure_update",
    "validate_role_isolated_contact_failure_update",
]
