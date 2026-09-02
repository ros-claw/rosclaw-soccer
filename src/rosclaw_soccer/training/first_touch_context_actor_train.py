"""Distill a bounded First Touch context actor from CPU MuJoCo teachers."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.first_touch_context_actor import (
    FirstTouchContextResidualActor,
    save_first_touch_context_actor,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.first_touch_physics import (
    FirstTouchCandidate,
    FirstTouchPhysicsScenario,
)


@dataclass(frozen=True)
class FirstTouchTeacherSample:
    """One verified teacher report reduced to learner inputs and labels."""

    report_hash: str
    scenario_hash: str
    body_hash: str
    kick_prior_hash: str
    source_implementation_hash: str
    scenario: FirstTouchPhysicsScenario
    candidate: FirstTouchCandidate
    passed: bool
    safety_passed: bool

    @property
    def features(self) -> NDArray[np.float64]:
        return np.asarray(
            (
                self.scenario.incoming_speed_mps,
                self.scenario.incoming_lateral_m,
                self.scenario.target_direction_deg,
                self.scenario.target_outgoing_speed_mps,
            ),
            dtype=np.float64,
        )

    @property
    def targets(self) -> NDArray[np.float64]:
        value = self.candidate
        return np.asarray(
            (
                value.receiver_start_delay_sec,
                value.stance_offset_x,
                value.stance_offset_y,
                value.swing_amplitude,
                value.swing_speed_scale,
                value.com_shift_y,
                value.pelvis_yaw_offset,
                value.foot_yaw_offset,
                value.foot_pitch_offset,
                value.loft_synergy,
            ),
            dtype=np.float64,
        )


def load_first_touch_teacher_sample(path: Path) -> FirstTouchTeacherSample:
    """Load one report and verify its physics and content commitments."""

    report_path = path.expanduser().resolve()
    payload = json.loads(report_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("First Touch teacher report must be a mapping")
    claimed_hash = payload.get("report_hash")
    committed = dict(payload)
    committed.pop("report_hash", None)
    if claimed_hash != hash_json(committed):
        raise ValueError("First Touch teacher report hash is invalid")
    if payload.get("schema_version") != "rosclaw_soccer.first_touch_physics_evidence.v1":
        raise ValueError("First Touch teacher report schema is unsupported")
    physics = payload.get("physics")
    provenance = payload.get("provenance")
    evaluation = payload.get("evaluation")
    evidence_ceiling = payload.get("evidence_ceiling")
    if not all(
        isinstance(value, dict) for value in (physics, provenance, evaluation, evidence_ceiling)
    ):
        raise ValueError("First Touch teacher report is incomplete")
    assert isinstance(physics, dict)
    assert isinstance(provenance, dict)
    assert isinstance(evaluation, dict)
    assert isinstance(evidence_ceiling, dict)
    trajectory_name = physics.get("trajectory_artifact")
    if not isinstance(trajectory_name, str) or Path(trajectory_name).name != trajectory_name:
        raise ValueError("First Touch teacher trajectory path is invalid")
    trajectory_path = report_path.parent / trajectory_name
    if physics.get("trajectory_artifact_hash") != hash_bytes(trajectory_path.read_bytes()):
        raise ValueError("First Touch teacher trajectory hash is invalid")
    if (
        physics.get("authority") != "CPU_MUJOCO"
        or physics.get("strict_replay") is not True
        or physics.get("pixels_used_for_scoring") is not False
        or payload.get("hardware_command_sent") is not False
        or evidence_ceiling.get("activation_ceiling") != "SIM_ONLY"
    ):
        raise ValueError("First Touch teacher report exceeds the SIM_ONLY boundary")
    scenario_value = payload.get("scenario")
    candidate_value = payload.get("candidate")
    if not isinstance(scenario_value, dict) or not isinstance(candidate_value, dict):
        raise ValueError("First Touch teacher scenario or candidate is missing")
    scenario = FirstTouchPhysicsScenario(**scenario_value)
    candidate = FirstTouchCandidate(**candidate_value)
    if payload.get("scenario_hash") != scenario.scenario_hash:
        raise ValueError("First Touch teacher scenario commitment is invalid")
    if payload.get("candidate_hash") != candidate.candidate_hash:
        raise ValueError("First Touch teacher candidate commitment is invalid")
    values = {
        "body_hash": provenance.get("body_hash"),
        "kick_prior_hash": provenance.get("kick_prior_hash"),
        "source_implementation_hash": provenance.get("implementation_hash"),
    }
    if any(
        not isinstance(value, str) or not value.startswith("sha256:") for value in values.values()
    ):
        raise ValueError("First Touch teacher provenance is invalid")
    return FirstTouchTeacherSample(
        report_hash=str(claimed_hash),
        scenario_hash=scenario.scenario_hash,
        body_hash=str(values["body_hash"]),
        kick_prior_hash=str(values["kick_prior_hash"]),
        source_implementation_hash=str(values["source_implementation_hash"]),
        scenario=scenario,
        candidate=candidate,
        passed=evaluation.get("passed") is True,
        safety_passed=evaluation.get("safety_passed") is True,
    )


def fit_first_touch_context_actor(
    samples: tuple[FirstTouchTeacherSample, ...],
    *,
    kick_foot: str,
    sealed_retention_scenario_hashes: tuple[str, ...] = (),
    ridge_regularization: float = 0.05,
    maximum_support_distance: float = 2.25,
) -> FirstTouchContextResidualActor:
    """Fit one foot-local actor while keeping the retention set sealed."""

    if kick_foot not in {"left", "right"}:
        raise ValueError("First Touch learner kick foot is invalid")
    if not math.isfinite(ridge_regularization) or not 0.0 < ridge_regularization <= 10.0:
        raise ValueError("First Touch learner ridge regularization is invalid")
    local = tuple(sample for sample in samples if sample.candidate.kick_foot == kick_foot)
    if len(local) < 6 or len({sample.report_hash for sample in local}) != len(local):
        raise ValueError("First Touch learner needs six unique foot-local reports")
    body_hashes = {sample.body_hash for sample in local}
    prior_hashes = {sample.kick_prior_hash for sample in local}
    implementation_hashes = {sample.source_implementation_hash for sample in local}
    if len(body_hashes) != 1 or len(prior_hashes) != 1 or len(implementation_hashes) != 1:
        raise ValueError("First Touch learner teacher provenance is heterogeneous")
    retention = set(sealed_retention_scenario_hashes)
    training_scenarios = {sample.scenario_hash for sample in local}
    if training_scenarios & retention:
        raise ValueError("First Touch retention scenario leaked into actor training")
    successful = tuple(sample for sample in local if sample.passed and sample.safety_passed)
    rejected = tuple(sample for sample in local if not sample.passed)
    if len(successful) < 4 or len(rejected) < 2:
        raise ValueError("First Touch learner lacks successful and rejected support")
    features = np.stack([sample.features for sample in successful])
    targets = np.stack([sample.targets for sample in successful])
    center = np.mean(features, axis=0)
    scale = np.maximum(
        np.std(features, axis=0),
        np.asarray((0.05, 0.02, 5.0, 0.10), dtype=np.float64),
    )
    normalized = (features - center) / scale
    design = np.column_stack((np.ones(features.shape[0]), normalized))
    penalty = np.eye(design.shape[1], dtype=np.float64) * ridge_regularization
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ targets).T
    predicted = design @ coefficients.T
    fit_rmse = float(np.sqrt(np.mean(np.square(predicted - targets))))
    source_hashes = tuple(sorted(sample.report_hash for sample in local))
    scenario_hashes = tuple(sorted(training_scenarios))
    snapshot = hash_json(
        {
            "source_report_hashes": source_hashes,
            "training_scenario_hashes": scenario_hashes,
            "kick_foot": kick_foot,
            "ridge_regularization": ridge_regularization,
            "sealed_retention_scenario_hashes": sorted(retention),
        }
    )
    implementation_hash = hash_json(
        {
            "actor_module": hash_bytes(
                Path(__file__)
                .resolve()
                .parents[1]
                .joinpath("growth", "first_touch_context_actor.py")
                .read_bytes()
            ),
            "trainer_module": hash_bytes(Path(__file__).read_bytes()),
            "source_physics_implementation_hash": next(iter(implementation_hashes)),
        }
    )
    return FirstTouchContextResidualActor(
        body_hash=next(iter(body_hashes)),
        kick_prior_hash=next(iter(prior_hashes)),
        implementation_hash=implementation_hash,
        training_snapshot_hash=str(snapshot),
        source_report_hashes=source_hashes,
        training_scenario_hashes=scenario_hashes,
        kick_foot=kick_foot,
        feature_center=tuple(float(value) for value in center),
        feature_scale=tuple(float(value) for value in scale),
        support_minimum=tuple(float(value) for value in features.min(axis=0)),
        support_maximum=tuple(float(value) for value in features.max(axis=0)),
        normalized_support_points=tuple(tuple(float(value) for value in row) for row in normalized),
        coefficient_matrix=tuple(tuple(float(value) for value in row) for row in coefficients),
        successful_teacher_count=len(successful),
        rejected_teacher_count=len(rejected),
        fit_rmse=fit_rmse,
        ridge_regularization=ridge_regularization,
        maximum_support_distance=maximum_support_distance,
    )


def train_first_touch_context_actor(
    *,
    report_paths: tuple[Path, ...],
    output_path: Path,
    kick_foot: str,
    sealed_retention_scenario_hashes: tuple[str, ...] = (),
    ridge_regularization: float = 0.05,
    maximum_support_distance: float = 2.25,
) -> FirstTouchContextResidualActor:
    samples = tuple(load_first_touch_teacher_sample(path) for path in report_paths)
    actor = fit_first_touch_context_actor(
        samples,
        kick_foot=kick_foot,
        sealed_retention_scenario_hashes=sealed_retention_scenario_hashes,
        ridge_regularization=ridge_regularization,
        maximum_support_distance=maximum_support_distance,
    )
    save_first_touch_context_actor(actor, output_path)
    return actor


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--kick-foot", choices=("left", "right"), required=True)
    parser.add_argument("--retention-scenario-hash", action="append", default=[])
    parser.add_argument("--ridge-regularization", type=float, default=0.05)
    parser.add_argument("--maximum-support-distance", type=float, default=2.25)
    args = parser.parse_args()
    actor = train_first_touch_context_actor(
        report_paths=tuple(args.report),
        output_path=args.output,
        kick_foot=args.kick_foot,
        sealed_retention_scenario_hashes=tuple(args.retention_scenario_hash),
        ridge_regularization=args.ridge_regularization,
        maximum_support_distance=args.maximum_support_distance,
    )
    print(json.dumps(actor.to_dict(), indent=2, ensure_ascii=False, allow_nan=False))


if __name__ == "__main__":
    _main()


__all__ = [
    "FirstTouchTeacherSample",
    "fit_first_touch_context_actor",
    "load_first_touch_teacher_sample",
    "train_first_touch_context_actor",
]
