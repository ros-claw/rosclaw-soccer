"""Matched CPU MuJoCo exam for a frozen-memory recurrent residual candidate."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal

import numpy as np
from rosclaw.continual.residual_adaptation import (
    ParameterIsolationEvidence,
    load_residual_adaptation_contract,
)

from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.training.opentrack_recovery_bridge_exam import (
    _atomic_json,
    _file_hash,
)
from rosclaw_soccer.training.opentrack_recovery_bridge_holdout import (
    _verified_development_report,
    _wilson_lower_bound,
)
from rosclaw_soccer.training.opentrack_recovery_residual_ppo import (
    _base_memories,
    _build_actor_critic,
    _create_environment,
    _memory_hash,
    _RecoveryResidualPhysics,
    _TrainingState,
)
from rosclaw_soccer.training.opentrack_recovery_student_collect import (
    _teacher_body_hash,
    _verified_holdout_report,
)
from rosclaw_soccer.training.recovery_residual_ppo import (
    RecoveryResidualObservationSpec,
    RecoveryResidualPPOConfig,
    RecoveryRewardConfig,
)
from rosclaw_soccer.training.recovery_snapshot import load_recovery_snapshot_corpus
from rosclaw_soccer.training.recovery_student import (
    load_recovery_distillation_corpus,
)
from rosclaw_soccer.training.recovery_teacher_bridge import (
    RecoveryPerturbationConfig,
    build_recovery_perturbation_holdout,
)


@dataclass(frozen=True)
class RecoveryResidualPhysicsTrial:
    controller: Literal["FROZEN_MEMORY_PARENT", "RECURRENT_RESIDUAL_CANDIDATE"]
    suite: Literal["RETENTION_BASE", "SEALED_ACQUISITION"]
    initial_snapshot_hash: str
    base_snapshot_hash: str
    succeeded: bool
    finite_state: bool
    executed_steps: int
    final_stable_frames: int
    final_pelvis_height_m: float
    final_upright_projection: float
    final_root_linear_speed_mps: float
    final_root_angular_speed_rad_s: float
    peak_root_angular_speed_rad_s: float
    mean_torque_saturation_fraction: float
    residual_output_rms_rad: float
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.recovery_residual_physics_trial.v1"

    def __post_init__(self) -> None:
        values = (
            self.final_pelvis_height_m,
            self.final_upright_projection,
            self.final_root_linear_speed_mps,
            self.final_root_angular_speed_rad_s,
            self.peak_root_angular_speed_rad_s,
            self.mean_torque_saturation_fraction,
            self.residual_output_rms_rad,
        )
        if (
            self.controller not in {"FROZEN_MEMORY_PARENT", "RECURRENT_RESIDUAL_CANDIDATE"}
            or self.suite not in {"RETENTION_BASE", "SEALED_ACQUISITION"}
            or any(
                not value.startswith("sha256:") or len(value) != 71
                for value in (self.initial_snapshot_hash, self.base_snapshot_hash)
            )
            or self.executed_steps <= 0
            or self.final_stable_frames < 0
            or any(not math.isfinite(value) for value in values)
            or min(
                self.final_pelvis_height_m,
                self.final_root_linear_speed_mps,
                self.final_root_angular_speed_rad_s,
                self.peak_root_angular_speed_rad_s,
                self.mean_torque_saturation_fraction,
                self.residual_output_rms_rad,
            )
            < 0.0
            or not 0.0 <= self.mean_torque_saturation_fraction <= 1.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("recovery residual physics trial is invalid")

    @property
    def trial_hash(self) -> str:
        return str(hash_json(asdict(self)))


def _verified_json(path: Path, *, schema: str, hash_key: str) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError("recovery residual evidence must be an object")
    declared = value.pop(hash_key, None)
    if value.get("schema_version") != schema or declared != hash_json(value):
        raise ValueError("recovery residual evidence integrity failed")
    value[hash_key] = declared
    return value


def _run_trial(
    *,
    physics: _RecoveryResidualPhysics,
    record: _TrainingState,
    suite: Literal["RETENTION_BASE", "SEALED_ACQUISITION"],
    controller: Literal["FROZEN_MEMORY_PARENT", "RECURRENT_RESIDUAL_CANDIDATE"],
    model: Any,
    config: RecoveryResidualPPOConfig,
) -> RecoveryResidualPhysicsTrial:
    import torch  # type: ignore[import-not-found]

    actor, critic = physics.reset(record)
    hidden = torch.zeros((1, config.hidden_size), dtype=torch.float32)
    peak_angular = 0.0
    saturation_sum = 0.0
    residual_square_sum = 0.0
    finite = True
    outcome = None
    for step in range(config.maximum_episode_steps):
        if controller == "FROZEN_MEMORY_PARENT":
            action = np.zeros(29, dtype=np.float32)
        else:
            with torch.no_grad():
                mean, _, hidden, _ = model(
                    torch.as_tensor(actor).view(1, 1, -1),
                    torch.as_tensor(critic).view(1, 1, -1),
                    hidden,
                    torch.tensor([[float(step == 0)]], dtype=torch.float32),
                )
                action = torch.tanh(mean[0, 0]).cpu().numpy()
        (actor, critic), outcome = physics.step(action)
        peak_angular = max(peak_angular, outcome.root_angular_speed_rad_s)
        saturation_sum += outcome.torque_saturation_fraction
        residual_square_sum += float(np.mean(np.square(physics.last_residual)))
        finite = finite and not (
            outcome.root_linear_speed_mps >= 100.0 or outcome.root_angular_speed_rad_s >= 100.0
        )
        if outcome.done:
            break
    if outcome is None:
        raise RuntimeError("recovery residual exam executed no physics steps")
    executed = step + 1
    return RecoveryResidualPhysicsTrial(
        controller=controller,
        suite=suite,
        initial_snapshot_hash=record.snapshot.snapshot_hash,
        base_snapshot_hash=record.base_snapshot_hash,
        succeeded=outcome.succeeded,
        finite_state=finite,
        executed_steps=executed,
        final_stable_frames=physics.stable_streak,
        final_pelvis_height_m=outcome.pelvis_height_m,
        final_upright_projection=outcome.upright_projection,
        final_root_linear_speed_mps=outcome.root_linear_speed_mps,
        final_root_angular_speed_rad_s=outcome.root_angular_speed_rad_s,
        peak_root_angular_speed_rad_s=peak_angular,
        mean_torque_saturation_fraction=saturation_sum / executed,
        residual_output_rms_rad=math.sqrt(residual_square_sum / executed),
    )


def run_opentrack_recovery_residual_exam(
    *,
    opentrack_root: Path,
    environment_config_path: Path,
    motion_dataset_id: str,
    snapshot_manifest_path: Path,
    development_report_path: Path,
    sealed_holdout_report_path: Path,
    corpus_manifest_path: Path,
    artifact_manifest_path: Path,
    training_report_path: Path,
    adaptation_contract_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Evaluate parent and candidate on identical retained and sealed states."""

    from safetensors.torch import load_file  # type: ignore[import-not-found]

    root = opentrack_root.expanduser().resolve()
    environment_path = environment_config_path.expanduser().resolve()
    snapshot_path = snapshot_manifest_path.expanduser().resolve()
    development_path = development_report_path.expanduser().resolve()
    sealed_path = sealed_holdout_report_path.expanduser().resolve()
    corpus_path = corpus_manifest_path.expanduser().resolve()
    artifact_path = artifact_manifest_path.expanduser().resolve()
    training_path = training_report_path.expanduser().resolve()
    contract_path = adaptation_contract_path.expanduser().resolve()
    output = output_path.expanduser().resolve()
    if not root.is_dir() or any(
        not path.is_file()
        for path in (
            environment_path,
            snapshot_path,
            development_path,
            sealed_path,
            corpus_path,
            artifact_path,
            training_path,
            contract_path,
        )
    ):
        raise FileNotFoundError("recovery residual exam inputs are incomplete")
    if output.exists() or output == root or root in output.parents:
        raise ValueError("recovery residual exam output must be new and external")
    development = _verified_development_report(development_path)
    sealed = _verified_holdout_report(sealed_path)
    artifact = _verified_json(
        artifact_path,
        schema="rosclaw_soccer.recovery_residual_actor_critic.v1",
        hash_key="artifact_hash",
    )
    training = _verified_json(
        training_path,
        schema="rosclaw_soccer.recovery_residual_ppo_training_report.v1",
        hash_key="report_hash",
    )
    contract = load_residual_adaptation_contract(contract_path)
    corpus = load_recovery_distillation_corpus(corpus_path)
    corpus_payload = json.loads(corpus_path.read_text(encoding="utf-8"))
    weights_path = artifact_path.parent / str(artifact["weights"])
    if (
        sealed.get("development_report_hash") != development["report_hash"]
        or development.get("snapshot_manifest_hash") != _file_hash(snapshot_path)
        or development.get("teacher_config_hash") != _file_hash(environment_path)
        or artifact.get("weights_hash") != _file_hash(weights_path)
        or training.get("artifact_hash") != artifact["artifact_hash"]
        or training.get("adaptation_contract_hash") != contract.contract_hash
        or artifact.get("body_hash") != corpus_payload.get("body_hash")
        or artifact.get("physics_scene_hash") != corpus_payload.get("physics_scene_hash")
        or artifact.get("external_reference_features") is not False
        or artifact.get("teacher_identity_input") is not False
        or artifact.get("future_reference_input") is not False
        or artifact.get("activation_ceiling") != "SIM_ONLY"
        or artifact.get("hardware_authorized") is not False
    ):
        raise ValueError("recovery residual exam evidence bindings differ")
    config = RecoveryResidualPPOConfig(**dict(artifact["config"]))
    reward = RecoveryRewardConfig(**dict(artifact["reward_config"]))
    spec_payload = dict(artifact["observation_spec"])
    spec_payload["actor_features"] = tuple(spec_payload["actor_features"])
    spec_payload["forbidden_actor_features"] = tuple(spec_payload["forbidden_actor_features"])
    spec = RecoveryResidualObservationSpec(**spec_payload)
    if (
        config.config_hash != artifact["config_hash"]
        or reward.config_hash != artifact["reward_config_hash"]
        or spec.spec_hash != artifact["observation_spec_hash"]
    ):
        raise ValueError("recovery residual exam config integrity failed")
    memories = _base_memories(corpus)
    memory_hash = _memory_hash(memories, corpus_hash=corpus.manifest_hash)
    if memory_hash != artifact["frozen_skill_memory_hash"]:
        raise ValueError("recovery residual frozen memory differs")
    model = _build_actor_critic(config, spec).cpu()
    model.load_state_dict(load_file(str(weights_path), device="cpu"), strict=True)
    model.eval()
    base_snapshots = load_recovery_snapshot_corpus(snapshot_path)
    holdout_config = RecoveryPerturbationConfig(**dict(sealed["perturbation_config"]))
    sealed_states = build_recovery_perturbation_holdout(base_snapshots, config=holdout_config)
    if len(sealed_states) != len(sealed["perturbations"]):
        raise ValueError("recovery residual sealed state count differs")
    for (snapshot, identity), expected in zip(sealed_states, sealed["perturbations"], strict=True):
        if (
            snapshot.snapshot_hash != expected["perturbed_snapshot_hash"]
            or identity.perturbation_hash != expected["perturbation_hash"]
        ):
            raise ValueError("recovery residual sealed state identity differs")
    retention = tuple(
        _TrainingState(
            snapshot=snapshot,
            base_snapshot_hash=snapshot.snapshot_hash,
            source="HISTORICAL_ANCHOR",
            difficulty=0.0,
            perturbation_hash=None,
        )
        for snapshot in base_snapshots
    )
    acquisition = tuple(
        _TrainingState(
            snapshot=snapshot,
            base_snapshot_hash=identity.base_snapshot_hash,
            source="RECENT_FAILURE",
            difficulty=0.60,
            perturbation_hash=identity.perturbation_hash,
        )
        for snapshot, identity in sealed_states
    )
    environment, constants, mujoco = _create_environment(
        opentrack_root=root,
        environment_config_path=environment_path,
        motion_dataset_id=motion_dataset_id,
        development=development,
        rank=0,
    )
    try:
        if (
            _teacher_body_hash(environment, mujoco) != corpus_payload["body_hash"]
            or _file_hash(Path(constants.task_to_xml("flat_terrain")).resolve())
            != corpus_payload["physics_scene_hash"]
        ):
            raise ValueError("recovery residual exam body or scene differs")
        physics = _RecoveryResidualPhysics(
            environment=environment,
            constants=constants,
            mujoco=mujoco,
            corpus=corpus,
            memories=memories,
            residual_config=config,
            reward_config=reward,
            observation_spec=spec,
        )
        trials: list[RecoveryResidualPhysicsTrial] = []
        for controller in (
            "FROZEN_MEMORY_PARENT",
            "RECURRENT_RESIDUAL_CANDIDATE",
        ):
            for state in retention:
                trials.append(
                    _run_trial(
                        physics=physics,
                        record=state,
                        suite="RETENTION_BASE",
                        controller=controller,
                        model=model,
                        config=config,
                    )
                )
            for state in acquisition:
                trials.append(
                    _run_trial(
                        physics=physics,
                        record=state,
                        suite="SEALED_ACQUISITION",
                        controller=controller,
                        model=model,
                        config=config,
                    )
                )
    finally:
        environment.close()
    grouped: dict[str, list[RecoveryResidualPhysicsTrial]] = {}
    for trial in trials:
        grouped.setdefault(f"{trial.controller}:{trial.suite}", []).append(trial)

    def _rate(controller: str, suite: str) -> tuple[int, int, float]:
        rows = grouped[f"{controller}:{suite}"]
        passed = sum(item.succeeded for item in rows)
        return passed, len(rows), passed / len(rows)

    parent_retained, retained_count, parent_retention_rate = _rate(
        "FROZEN_MEMORY_PARENT", "RETENTION_BASE"
    )
    candidate_retained, _, candidate_retention_rate = _rate(
        "RECURRENT_RESIDUAL_CANDIDATE", "RETENTION_BASE"
    )
    parent_acquired, acquisition_count, parent_acquisition_rate = _rate(
        "FROZEN_MEMORY_PARENT", "SEALED_ACQUISITION"
    )
    candidate_acquired, _, candidate_acquisition_rate = _rate(
        "RECURRENT_RESIDUAL_CANDIDATE", "SEALED_ACQUISITION"
    )
    candidate_trials = [
        item for item in trials if item.controller == "RECURRENT_RESIDUAL_CANDIDATE"
    ]
    critical_regressions = sum(not item.finite_state for item in candidate_trials)
    retention_passed = bool(
        candidate_retained == retained_count and candidate_retention_rate >= parent_retention_rate
    )
    acquisition_passed = bool(
        candidate_acquisition_rate >= 0.80 and candidate_acquisition_rate > parent_acquisition_rate
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_residual_matched_exam.v1",
        "artifact_hash": artifact["artifact_hash"],
        "training_report_hash": training["report_hash"],
        "adaptation_contract_hash": contract.contract_hash,
        "frozen_skill_memory_hash": memory_hash,
        "development_report_hash": development["report_hash"],
        "sealed_holdout_report_hash": sealed["report_hash"],
        "physics_backend": "opentrack_mujoco_cpu_direct_pd",
        "policy_inference_device": "cpu",
        "matched_parent_candidate_states": True,
        "retention_trial_count": retained_count,
        "parent_retention_passed_count": parent_retained,
        "parent_retention_pass_rate": parent_retention_rate,
        "candidate_retention_passed_count": candidate_retained,
        "candidate_retention_pass_rate": candidate_retention_rate,
        "sealed_acquisition_trial_count": acquisition_count,
        "parent_sealed_acquisition_passed_count": parent_acquired,
        "parent_sealed_acquisition_pass_rate": parent_acquisition_rate,
        "candidate_sealed_acquisition_passed_count": candidate_acquired,
        "candidate_sealed_acquisition_pass_rate": candidate_acquisition_rate,
        "candidate_sealed_wilson_95_lower_bound": _wilson_lower_bound(
            passed=candidate_acquired, count=acquisition_count
        ),
        "retention_passed": retention_passed,
        "acquisition_passed": acquisition_passed,
        "critical_safety_regressions": critical_regressions,
        "maximum_candidate_residual_output_rms_rad": max(
            item.residual_output_rms_rad for item in candidate_trials
        ),
        "mean_candidate_torque_saturation_fraction": float(
            np.mean([item.mean_torque_saturation_fraction for item in candidate_trials])
        ),
        "trials": [asdict(item) | {"trial_hash": item.trial_hash} for item in trials],
        "actor_reference_phase_reads": 0,
        "actor_teacher_identity_reads": 0,
        "environment_step_calls_during_control": 0,
        "physical_truth": True,
        "promotion_eligible": bool(
            retention_passed and acquisition_passed and critical_regressions == 0
        ),
        "promotion_blockers": [
            *([] if retention_passed else ["RETENTION_GATE_FAILED"]),
            *([] if acquisition_passed else ["SEALED_ACQUISITION_GATE_FAILED"]),
            *([] if critical_regressions == 0 else ["CRITICAL_SAFETY_REGRESSION"]),
            "NO_NEW_SOURCE_SCENE_FULL_CHAIN_EPISODES",
        ],
        "claim_boundary": "SIM_ONLY_RECOVERY_RESIDUAL_MATCHED_CPU_PHYSICS",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(output, report)
    parameter_evidence = ParameterIsolationEvidence(
        adaptation_contract_hash=contract.contract_hash,
        parent_artifact_hash=memory_hash,
        candidate_artifact_hash=str(artifact["artifact_hash"]),
        frozen_base_hash_before=str(training["frozen_skill_memory_hash_before"]),
        frozen_base_hash_after=str(training["frozen_skill_memory_hash_after"]),
        matched_exam_hash=str(report["report_hash"]),
        examined_frozen_parameter_count=sum(sequence.size for sequence in memories.values()),
        examined_trainable_parameter_count=sum(
            parameter.numel() for parameter in model.parameters()
        ),
        candidate_world_steps=int(training["world_steps"]),
        maximum_frozen_parameter_drift=float(training["maximum_frozen_parameter_drift"]),
        residual_output_rms=float(report["maximum_candidate_residual_output_rms_rad"]),
        retention_passed=retention_passed,
        acquisition_passed=acquisition_passed,
        critical_safety_regressions=critical_regressions,
    )
    evidence_payload = (
        parameter_evidence.to_dict()
        if hasattr(parameter_evidence, "to_dict")
        else asdict(parameter_evidence)
    )
    evidence_payload["evidence_hash"] = parameter_evidence.evidence_hash
    _atomic_json(output.with_name("parameter-isolation-evidence.json"), evidence_payload)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--opentrack-root", required=True, type=Path)
    parser.add_argument("--environment-config", required=True, type=Path)
    parser.add_argument("--motion-dataset-id", required=True)
    parser.add_argument("--snapshot-manifest", required=True, type=Path)
    parser.add_argument("--development-report", required=True, type=Path)
    parser.add_argument("--sealed-holdout-report", required=True, type=Path)
    parser.add_argument("--corpus-manifest", required=True, type=Path)
    parser.add_argument("--artifact-manifest", required=True, type=Path)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--adaptation-contract", required=True, type=Path)
    parser.add_argument("--output-path", required=True, type=Path)
    args = parser.parse_args()
    report = run_opentrack_recovery_residual_exam(
        opentrack_root=args.opentrack_root,
        environment_config_path=args.environment_config,
        motion_dataset_id=args.motion_dataset_id,
        snapshot_manifest_path=args.snapshot_manifest,
        development_report_path=args.development_report,
        sealed_holdout_report_path=args.sealed_holdout_report,
        corpus_manifest_path=args.corpus_manifest,
        artifact_manifest_path=args.artifact_manifest,
        training_report_path=args.training_report,
        adaptation_contract_path=args.adaptation_contract,
        output_path=args.output_path,
    )
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "parent_retention": report["parent_retention_pass_rate"],
                "candidate_retention": report["candidate_retention_pass_rate"],
                "parent_sealed": report["parent_sealed_acquisition_pass_rate"],
                "candidate_sealed": report["candidate_sealed_acquisition_pass_rate"],
                "promotion_eligible": report["promotion_eligible"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["run_opentrack_recovery_residual_exam"]
