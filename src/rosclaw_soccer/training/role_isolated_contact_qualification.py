"""Portfolio gate for a context-bound, failure-updated contact actor."""

from __future__ import annotations

import argparse
import inspect
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, cast

from rosclaw.feedback import contracts as growth_core_contracts
from rosclaw.simforge.reproducibility import (
    ReproducibilityClosure,
    build_reproducibility_closure,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_isolated_contact_failure_update import (
    validate_role_isolated_contact_failure_update,
)
from rosclaw_soccer.training.role_isolated_second_striker_probe import (
    validate_role_isolated_second_striker_probe,
)

_CLAIM = "ROLE_ISOLATED_CONTACT_CANDIDATE_CONTROL_AND_SEALED_HOLDOUT_QUALIFICATION"


@dataclass(frozen=True)
class RoleIsolatedContactQualificationConfig:
    control_second_ball_mass_kg: float | None = None
    control_second_ball_ground_friction: float | None = None
    control_second_striker_foot_pitch_offset_rad: float | None = None
    holdout_second_ball_mass_kg: float = 0.46
    holdout_second_ball_ground_friction: float = 0.16
    holdout_second_striker_foot_pitch_offset_rad: float | None = None
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.role_isolated_contact_qualification_config.v1"

    def __post_init__(self) -> None:
        optional = (
            self.control_second_ball_mass_kg,
            self.control_second_ball_ground_friction,
            self.control_second_striker_foot_pitch_offset_rad,
            self.holdout_second_striker_foot_pitch_offset_rad,
        )
        if any(value is not None and not math.isfinite(value) for value in optional):
            raise ValueError("role-isolated qualification split must be finite")
        if self.control_second_ball_mass_kg is not None and not (
            0.40 <= self.control_second_ball_mass_kg <= 0.46
        ):
            raise ValueError("role-isolated qualification control mass is invalid")
        if self.control_second_ball_ground_friction is not None and not (
            0.03 <= self.control_second_ball_ground_friction <= 0.80
        ):
            raise ValueError("role-isolated qualification control friction is invalid")
        if any(
            value is not None and not -0.18 <= value <= 0.18
            for value in (
                self.control_second_striker_foot_pitch_offset_rad,
                self.holdout_second_striker_foot_pitch_offset_rad,
            )
        ):
            raise ValueError("role-isolated qualification foot pitch is invalid")
        if (
            not math.isfinite(self.holdout_second_ball_mass_kg)
            or not 0.40 <= self.holdout_second_ball_mass_kg <= 0.46
            or not math.isfinite(self.holdout_second_ball_ground_friction)
            or not 0.03 <= self.holdout_second_ball_ground_friction <= 0.80
        ):
            raise ValueError("role-isolated qualification holdout is invalid")
        if self.activation_ceiling != "SIM_ONLY" or self.hardware_authorized:
            raise ValueError("role-isolated qualification must remain SIM_ONLY")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _request(path: Path) -> dict[str, Any]:
    value = json.loads((path.parent / "request.json").read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("role-isolated qualification request input is invalid")
    return cast(dict[str, Any], value)


def _derive_qualification(
    *, control: dict[str, Any], holdout: dict[str, Any]
) -> tuple[dict[str, bool], bool, str]:
    gates = {
        "control_evidence_reproducible": bool(
            control.get("evidence_passed") is True
            and cast(dict[str, bool], control.get("evidence_gates", {})).get("strict_replay")
            is True
        ),
        "control_candidate_selected": bool(
            cast(dict[str, bool], control.get("plasticity_gates", {})).get(
                "candidate_selected"
            )
            is True
        ),
        "control_complete_chain_passed": bool(
            cast(dict[str, bool], control.get("plasticity_gates", {})).get(
                "complete_chain_passed"
            )
            is True
        ),
        "holdout_evidence_reproducible": bool(
            holdout.get("evidence_passed") is True
            and cast(dict[str, bool], holdout.get("evidence_gates", {})).get("strict_replay")
            is True
        ),
        "holdout_candidate_selected": bool(
            cast(dict[str, bool], holdout.get("plasticity_gates", {})).get(
                "candidate_selected"
            )
            is True
        ),
        "holdout_complete_chain_passed": bool(
            cast(dict[str, bool], holdout.get("plasticity_gates", {})).get(
                "complete_chain_passed"
            )
            is True
        ),
    }
    promoted = all(gates.values())
    if promoted:
        status = "QUALIFIED_CONTROL_AND_SEALED_HOLDOUT_SIM_ONLY_CANDIDATE"
    elif all(value for key, value in gates.items() if key != "holdout_complete_chain_passed"):
        status = "REJECTED_SEALED_HOLDOUT_TASK_FAILURE"
    else:
        status = "REJECTED_INCOMPLETE_QUALIFICATION_PORTFOLIO"
    return gates, promoted, status


def _closure_inputs(
    *, update: Path, control: Path, holdout: Path
) -> tuple[dict[str, Path], dict[str, Path]]:
    sources = {
        "rosclaw-core-reproducibility": Path(inspect.getfile(ReproducibilityClosure))
        .resolve()
        .parent,
        "rosclaw-core-runtime": Path(inspect.getfile(growth_core_contracts)).resolve().parents[2],
        "soccer": Path(__file__).resolve().parents[1],
    }
    artifacts: dict[str, Path] = {
        "failure-update-evidence": update,
        "failure-update-request": update.parent / "request.json",
        "failure-updated-actor": update.parent / "failure-updated-contact-actor.json",
        "control-evidence": control,
        "control-request": control.parent / "request.json",
        "holdout-evidence": holdout,
        "holdout-request": holdout.parent / "request.json",
    }
    for prefix, evidence in (("control", control), ("holdout", holdout)):
        for index, trajectory in enumerate(sorted(evidence.parent.glob("replay-*-trajectory.npz"))):
            artifacts[f"{prefix}-trajectory-{index}"] = trajectory
    return sources, artifacts


def _verified_inputs(
    *,
    update_path: Path,
    control_path: Path,
    holdout_path: Path,
    config: RoleIsolatedContactQualificationConfig,
) -> tuple[dict[str, Any], dict[str, bool], bool, str]:
    update = validate_role_isolated_contact_failure_update(update_path)
    control = validate_role_isolated_second_striker_probe(control_path)
    holdout = validate_role_isolated_second_striker_probe(holdout_path)
    control_request = _request(control_path)
    holdout_request = _request(holdout_path)
    control_config = cast(dict[str, Any], control_request.get("config", {}))
    holdout_config = cast(dict[str, Any], holdout_request.get("config", {}))
    control_actor = cast(dict[str, str], control_request.get("artifact_locators", {})).get(
        "plastic-contact-candidate"
    )
    holdout_actor = cast(dict[str, str], holdout_request.get("artifact_locators", {})).get(
        "plastic-contact-candidate"
    )
    actor_path = update_path.parent / str(update.get("actor_file", ""))
    if (
        control_actor is None
        or holdout_actor is None
        or not actor_path.is_file()
        or not Path(control_actor).is_file()
        or not Path(holdout_actor).is_file()
        or hash_bytes(Path(control_actor).read_bytes()) != update.get("actor_file_hash")
        or hash_bytes(Path(holdout_actor).read_bytes()) != update.get("actor_file_hash")
        or control_config.get("second_ball_mass_kg")
        != config.control_second_ball_mass_kg
        or control_config.get("second_ball_ground_friction")
        != config.control_second_ball_ground_friction
        or control_config.get("second_striker_foot_pitch_offset_rad")
        != config.control_second_striker_foot_pitch_offset_rad
        or holdout_config.get("second_ball_mass_kg") != config.holdout_second_ball_mass_kg
        or holdout_config.get("second_ball_ground_friction")
        != config.holdout_second_ball_ground_friction
        or holdout_config.get("second_striker_foot_pitch_offset_rad")
        != config.holdout_second_striker_foot_pitch_offset_rad
    ):
        raise ValueError("qualification probes do not share the sealed candidate and split")
    gates, promoted, status = _derive_qualification(control=control, holdout=holdout)
    inputs = {
        "failure_update_report_hash": update["report_hash"],
        "candidate_actor_hash": update["actor_hash"],
        "control_report_hash": control["report_hash"],
        "holdout_report_hash": holdout["report_hash"],
        "control_status": control["candidate_status"],
        "holdout_status": holdout["candidate_status"],
    }
    return inputs, gates, promoted, status


def run_role_isolated_contact_qualification(
    *,
    failure_update_evidence_path: Path,
    control_evidence_path: Path,
    holdout_evidence_path: Path,
    output_dir: Path,
    config: RoleIsolatedContactQualificationConfig | None = None,
) -> dict[str, Any]:
    active = config or RoleIsolatedContactQualificationConfig()
    update = failure_update_evidence_path.expanduser().resolve()
    control = control_evidence_path.expanduser().resolve()
    holdout = holdout_evidence_path.expanduser().resolve()
    destination = output_dir.expanduser().resolve()
    checkout = Path(__file__).resolve().parents[3]
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("qualification requires a new external evidence directory")
    if not all(path.is_file() for path in (update, control, holdout)):
        raise FileNotFoundError("qualification input evidence is missing")
    sources, artifacts = _closure_inputs(update=update, control=control, holdout=holdout)
    closure = build_reproducibility_closure(
        source_trees=sources,
        dependency_packages=("numpy",),
        artifacts=artifacts,
        expected_replays=2,
    )
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_contact_qualification_request.v1",
        "config": asdict(active),
        "config_hash": active.config_hash,
        "source_tree_locators": {key: str(value) for key, value in sources.items()},
        "artifact_locators": {key: str(value) for key, value in artifacts.items()},
        "reproducibility_closure": closure.to_dict(),
        "reproducibility_closure_hash": closure.closure_hash,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    destination.mkdir(parents=True)
    request_path = destination / "request.json"
    _atomic_json(request_path, request)
    inputs, gates, promoted, status = _verified_inputs(
        update_path=update,
        control_path=control,
        holdout_path=holdout,
        config=active,
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.role_isolated_contact_qualification_evidence.v1",
        "claim": _CLAIM,
        "candidate_promoted": promoted,
        "candidate_status": status,
        "gates": gates,
        "inputs": inputs,
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
    return validate_role_isolated_contact_qualification(evidence_path)


def validate_role_isolated_contact_qualification(path: Path) -> dict[str, Any]:
    resolved = path.expanduser().resolve()
    payload_value = json.loads(resolved.read_text(encoding="utf-8"))
    if not isinstance(payload_value, dict):
        raise ValueError("qualification evidence must be an object")
    payload = cast(dict[str, Any], payload_value)
    expected_hash = payload.pop("report_hash", None)
    try:
        request_path = resolved.parent / "request.json"
        request = _request(resolved)
        config_value = request.get("config")
        source_value = request.get("source_tree_locators")
        artifact_value = request.get("artifact_locators")
        closure_value = request.get("reproducibility_closure")
        if not all(
            isinstance(value, dict)
            for value in (config_value, source_value, artifact_value, closure_value)
        ):
            raise ValueError("qualification request is incomplete")
        active = RoleIsolatedContactQualificationConfig(**cast(dict[str, Any], config_value))
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
        inputs, gates, promoted, status = _verified_inputs(
            update_path=artifacts["failure-update-evidence"],
            control_path=artifacts["control-evidence"],
            holdout_path=artifacts["holdout-evidence"],
            config=active,
        )
        if (
            request.get("schema_version")
            != "rosclaw_soccer.role_isolated_contact_qualification_request.v1"
            or request.get("config_hash") != active.config_hash
            or recorded != closure
            or request.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("schema_version")
            != "rosclaw_soccer.role_isolated_contact_qualification_evidence.v1"
            or payload.get("claim") != _CLAIM
            or payload.get("inputs") != inputs
            or payload.get("gates") != gates
            or payload.get("candidate_promoted") is not promoted
            or payload.get("candidate_status") != status
            or payload.get("request_hash") != hash_bytes(request_path.read_bytes())
            or payload.get("reproducibility_closure_hash") != closure.closure_hash
            or payload.get("activation_ceiling") != "SIM_ONLY"
            or payload.get("hardware_command_sent") is not False
            or payload.get("commercial_use_allowed") is not False
            or payload.get("pixels_used_for_scoring") is not False
            or expected_hash != hash_json(payload)
        ):
            raise ValueError("qualification authority or integrity contract is invalid")
    finally:
        if expected_hash is not None:
            payload["report_hash"] = expected_hash
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--failure-update-evidence", type=Path, required=True)
    parser.add_argument("--control-evidence", type=Path, required=True)
    parser.add_argument("--holdout-evidence", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--control-second-ball-mass-kg", type=float)
    parser.add_argument("--control-second-ball-ground-friction", type=float)
    parser.add_argument("--control-second-striker-foot-pitch-offset-rad", type=float)
    parser.add_argument("--holdout-second-ball-mass-kg", type=float, default=0.46)
    parser.add_argument("--holdout-second-ball-ground-friction", type=float, default=0.16)
    parser.add_argument("--holdout-second-striker-foot-pitch-offset-rad", type=float)
    return parser


def main() -> int:
    args = _parser().parse_args()
    report = run_role_isolated_contact_qualification(
        failure_update_evidence_path=args.failure_update_evidence,
        control_evidence_path=args.control_evidence,
        holdout_evidence_path=args.holdout_evidence,
        output_dir=args.output_dir,
        config=RoleIsolatedContactQualificationConfig(
            control_second_ball_mass_kg=args.control_second_ball_mass_kg,
            control_second_ball_ground_friction=args.control_second_ball_ground_friction,
            control_second_striker_foot_pitch_offset_rad=(
                args.control_second_striker_foot_pitch_offset_rad
            ),
            holdout_second_ball_mass_kg=args.holdout_second_ball_mass_kg,
            holdout_second_ball_ground_friction=args.holdout_second_ball_ground_friction,
            holdout_second_striker_foot_pitch_offset_rad=(
                args.holdout_second_striker_foot_pitch_offset_rad
            ),
        ),
    )
    print(json.dumps(report, indent=2, sort_keys=True, ensure_ascii=False))
    return 0 if report.get("candidate_promoted") is True else 1


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "RoleIsolatedContactQualificationConfig",
    "run_role_isolated_contact_qualification",
    "validate_role_isolated_contact_qualification",
]
