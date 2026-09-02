"""Train and examine a bounded 2v1 tactical actor on sealed physics states."""

from __future__ import annotations

import json
import math
import os
import subprocess
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.tactical_2v1 import TacticalAction, TwoVsOneDecisionEvidence
from rosclaw_soccer.growth.tactical_2v1_actor import (
    TwoVsOneTacticalActor,
    fit_two_vs_one_tactical_actor,
    load_two_vs_one_tactical_actor,
    save_two_vs_one_tactical_actor,
)
from rosclaw_soccer.providers.g1.asset_qualification import trajectory_digest
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.tactical_2v1_physics import (
    FrozenTacticalSkillBundle,
    TwoVsOnePhysicsConfig,
    TwoVsOnePhysicsResult,
    TwoVsOnePhysicsScenario,
    matched_two_vs_one_decision,
    simulate_two_vs_one_physics,
)

_ACTIONS = (TacticalAction.PASS, TacticalAction.SHOOT)


@dataclass(frozen=True)
class TwoVsOneRetentionManifest:
    scenarios: tuple[TwoVsOnePhysicsScenario, ...]
    suite_id: str = "s119.two-vs-one.sealed-retention"
    training_access_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.two_vs_one_retention_manifest.v1"

    def __post_init__(self) -> None:
        hashes = tuple(scenario.scenario_hash for scenario in self.scenarios)
        if (
            len(self.scenarios) < 16
            or len(set(hashes)) != len(hashes)
            or self.training_access_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("2v1 retention manifest must be unique, sealed and SIM_ONLY")

    def to_dict(self, *, include_hash: bool = True) -> dict[str, Any]:
        value: dict[str, Any] = {
            "schema_version": self.schema_version,
            "suite_id": self.suite_id,
            "scenarios": [asdict(scenario) for scenario in self.scenarios],
            "scenario_hashes": [scenario.scenario_hash for scenario in self.scenarios],
            "training_access_allowed": self.training_access_allowed,
            "activation_ceiling": self.activation_ceiling,
        }
        if include_hash:
            value["manifest_hash"] = hash_json(value)
        return value

    @property
    def manifest_hash(self) -> str:
        return str(hash_json(self.to_dict(include_hash=False)))


@dataclass(frozen=True)
class TwoVsOneGrowthThresholds:
    minimum_task_success_rate: float = 1.0
    minimum_action_agreement_rate: float = 0.80
    minimum_exact_replay_rate: float = 1.0
    maximum_mean_regret: float = 0.15
    maximum_case_regret: float = 0.55
    minimum_action_count: int = 4
    schema_version: str = "rosclaw_soccer.two_vs_one_growth_thresholds.v1"

    def __post_init__(self) -> None:
        values = (
            self.minimum_task_success_rate,
            self.minimum_action_agreement_rate,
            self.minimum_exact_replay_rate,
            self.maximum_mean_regret,
            self.maximum_case_regret,
        )
        if any(not math.isfinite(value) for value in values) or not (
            0.5 <= self.minimum_task_success_rate <= 1.0
            and 0.5 <= self.minimum_action_agreement_rate <= 1.0
            and 0.5 <= self.minimum_exact_replay_rate <= 1.0
            and 0.0 <= self.maximum_mean_regret <= self.maximum_case_regret <= 1.0
            and 1 <= self.minimum_action_count <= 128
        ):
            raise ValueError("2v1 growth thresholds are invalid")


def default_two_vs_one_acquisition_scenarios() -> tuple[TwoVsOnePhysicsScenario, ...]:
    """Balanced acquisition states; no retention state is returned here."""

    commitments = (0.0, 0.30, 0.48, 0.56, 0.62, 0.66, 0.72, 1.0)
    rows: list[TwoVsOnePhysicsScenario] = []
    index = 0
    for side in (-1.0, 1.0):
        for lateral in (1.40, 1.65):
            for commitment in commitments:
                rows.append(
                    TwoVsOnePhysicsScenario(
                        scenario_id=f"s119.acquisition.{index:03d}",
                        seed=119_000 + index,
                        defender_commitment=commitment,
                        finisher_lateral_m=side * lateral,
                    )
                )
                index += 1
    return tuple(rows)


def default_two_vs_one_retention_manifest() -> TwoVsOneRetentionManifest:
    """Return a disjoint suite with unseen layouts and physical nuisance values."""

    commitments = (0.15, 0.42, 0.52, 0.60, 0.68, 0.84)
    rows: list[TwoVsOnePhysicsScenario] = []
    index = 0
    for side in (-1.0, 1.0):
        for lateral in (1.50, 1.75):
            for commitment in commitments:
                rows.append(
                    TwoVsOnePhysicsScenario(
                        scenario_id=f"s119.retention.{index:03d}",
                        seed=119_500 + index,
                        defender_commitment=commitment,
                        finisher_lateral_m=side * lateral,
                        defender_reaction_delay_sec=0.33 + 0.04 * (index % 2),
                        defender_speed_scale=0.90 + 0.10 * (index % 3),
                        ball_ground_friction=0.30 + 0.10 * (index % 2),
                    )
                )
                index += 1
    return TwoVsOneRetentionManifest(scenarios=tuple(rows))


def _policy_hash(label: str, skill_bundle: FrozenTacticalSkillBundle) -> str:
    return str(hash_json({"policy": label, "skill_bundle_hash": skill_bundle.bundle_hash}))


def collect_two_vs_one_acquisition(
    *,
    scenarios: Iterable[TwoVsOnePhysicsScenario],
    skill_bundle: FrozenTacticalSkillBundle,
    config: TwoVsOnePhysicsConfig | None = None,
) -> tuple[tuple[TwoVsOneDecisionEvidence, ...], tuple[dict[str, Any], ...]]:
    """Collect exhaustive physical action labels without reading retention."""

    active = config or TwoVsOnePhysicsConfig()
    evidence: list[TwoVsOneDecisionEvidence] = []
    ledger: list[dict[str, Any]] = []
    policy_hash = _policy_hash("exhaustive_physics_teacher_v1", skill_bundle)
    for scenario in scenarios:
        actions: dict[str, Any] = {}
        for action in _ACTIONS:
            item, primary, ablated, _, _ = matched_two_vs_one_decision(
                scenario=scenario,
                action=action,
                policy_hash=policy_hash,
                skill_bundle=skill_bundle,
                config=active,
            )
            evidence.append(item)
            actions[action.value] = {
                "evidence_hash": item.evidence_hash,
                "weighted_score": item.weighted_score,
                "difference_reward": item.rollout.difference_reward,
                "primary": primary.to_dict(),
                "ablated": ablated.to_dict(),
            }
        ledger.append(
            {
                "scenario": asdict(scenario),
                "scenario_hash": scenario.scenario_hash,
                "state": scenario.state(skill_bundle=skill_bundle, config=active).__dict__,
                "actions": actions,
            }
        )
    return tuple(evidence), tuple(ledger)


def _save_trajectory(path: Path, trajectory: dict[str, NDArray[Any]]) -> dict[str, str]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **trajectory)  # type: ignore[arg-type]
    os.replace(temporary, path)
    return {
        "file": path.name,
        "file_hash": hash_bytes(path.read_bytes()),
        "trajectory_digest": trajectory_digest(trajectory),
    }


def _task_succeeded(result: TwoVsOnePhysicsResult) -> bool:
    return bool(result.goal_scored or result.pass_completed)


def evaluate_two_vs_one_retention(
    *,
    actor: TwoVsOneTacticalActor,
    manifest: TwoVsOneRetentionManifest,
    skill_bundle: FrozenTacticalSkillBundle,
    output_dir: Path,
    config: TwoVsOnePhysicsConfig | None = None,
    thresholds: TwoVsOneGrowthThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate unseen states, alternatives, ablations and exact replays."""

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    active = config or TwoVsOnePhysicsConfig()
    gates = thresholds or TwoVsOneGrowthThresholds()
    rows: list[dict[str, Any]] = []
    policy_hash = actor.actor_hash
    selected_counts = {action.value: 0 for action in _ACTIONS}
    for index, scenario in enumerate(manifest.scenarios):
        state = scenario.state(skill_bundle=skill_bundle, config=active)
        decision = actor.decide(state)
        action_runs: dict[TacticalAction, tuple[Any, ...]] = {}
        for action in _ACTIONS:
            action_runs[action] = matched_two_vs_one_decision(
                scenario=scenario,
                action=action,
                policy_hash=policy_hash,
                skill_bundle=skill_bundle,
                config=active,
            )
        if decision.action not in action_runs:
            raise ValueError("sealed 2v1 state fell outside actor support")
        selected_counts[decision.action.value] += 1
        selected = action_runs[decision.action]
        selected_evidence = selected[0]
        primary = selected[1]
        primary_trace = selected[3]
        replay, replay_trace = simulate_two_vs_one_physics(
            scenario=scenario,
            action=decision.action,
            skill_bundle=skill_bundle,
            config=active,
            focal_agent_present=True,
        )
        exact_replay = bool(
            replay.to_dict() == primary.to_dict()
            and trajectory_digest(replay_trace) == trajectory_digest(primary_trace)
        )
        scores = {action: float(action_runs[action][0].weighted_score) for action in _ACTIONS}
        oracle = max(_ACTIONS, key=lambda action: (scores[action], action.value))
        regret = max(0.0, scores[oracle] - scores[decision.action])
        case_dir = destination / f"case-{index:03d}"
        case_dir.mkdir()
        primary_artifact = _save_trajectory(case_dir / "selected-primary.npz", primary_trace)
        replay_artifact = _save_trajectory(case_dir / "selected-replay.npz", replay_trace)
        rows.append(
            {
                "scenario_id": scenario.scenario_id,
                "scenario_hash": scenario.scenario_hash,
                "state_hash": state.state_hash,
                "decision": {
                    **asdict(decision),
                    "action": decision.action.value,
                },
                "selected_action": decision.action.value,
                "oracle_action": oracle.value,
                "action_agreement": decision.action is oracle,
                "regret": regret,
                "task_succeeded": _task_succeeded(primary),
                "safe": primary.safe,
                "exact_replay": exact_replay,
                "selected_evidence_hash": selected_evidence.evidence_hash,
                "selected_weighted_score": selected_evidence.weighted_score,
                "selected_difference_reward": selected_evidence.rollout.difference_reward,
                "pass_score": scores[TacticalAction.PASS],
                "shoot_score": scores[TacticalAction.SHOOT],
                "pass_task_succeeded": _task_succeeded(action_runs[TacticalAction.PASS][1]),
                "shoot_task_succeeded": _task_succeeded(action_runs[TacticalAction.SHOOT][1]),
                "primary_result": primary.to_dict(),
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
            }
        )

    count = len(rows)
    success_rate = sum(bool(row["task_succeeded"]) for row in rows) / count
    action_agreement = sum(bool(row["action_agreement"]) for row in rows) / count
    exact_replay_rate = sum(bool(row["exact_replay"]) for row in rows) / count
    safe_rate = sum(bool(row["safe"]) for row in rows) / count
    regrets = [float(row["regret"]) for row in rows]
    pass_counterfactual = [
        float(row["selected_difference_reward"])
        for row in rows
        if row["selected_action"] == TacticalAction.PASS.value
    ]
    baseline_shoot_rate = sum(bool(row["shoot_task_succeeded"]) for row in rows) / count
    baseline_pass_rate = sum(bool(row["pass_task_succeeded"]) for row in rows) / count
    minimum_pass_difference = min(pass_counterfactual) if pass_counterfactual else None
    gate_values = {
        "task_success_rate": success_rate >= gates.minimum_task_success_rate,
        "action_agreement_rate": action_agreement >= gates.minimum_action_agreement_rate,
        "exact_replay_rate": exact_replay_rate >= gates.minimum_exact_replay_rate,
        "safe_rate": safe_rate == 1.0,
        "mean_regret": float(np.mean(regrets)) <= gates.maximum_mean_regret,
        "maximum_case_regret": max(regrets) <= gates.maximum_case_regret,
        "both_actions_selected": all(
            selected_counts[action.value] >= gates.minimum_action_count for action in _ACTIONS
        ),
        "pass_counterfactual_positive": minimum_pass_difference is not None
        and minimum_pass_difference > 0.0,
        "beats_fixed_shoot": success_rate > baseline_shoot_rate,
        "beats_fixed_pass": success_rate > baseline_pass_rate,
    }
    passed = all(gate_values.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.two_vs_one_retention_exam.v1",
        "status": "PASS_BOUNDED_TACTICAL_RETENTION"
        if passed
        else "REJECTED_BOUNDED_TACTICAL_RETENTION",
        "actor_hash": actor.actor_hash,
        "training_snapshot_hash": actor.training_snapshot_hash,
        "manifest_hash": manifest.manifest_hash,
        "skill_bundle_hash": skill_bundle.bundle_hash,
        "physics_config_hash": active.config_hash,
        "thresholds": asdict(gates),
        "metrics": {
            "case_count": count,
            "task_success_rate": success_rate,
            "action_agreement_rate": action_agreement,
            "exact_replay_rate": exact_replay_rate,
            "safe_rate": safe_rate,
            "mean_regret": float(np.mean(regrets)),
            "maximum_case_regret": max(regrets),
            "selected_action_counts": selected_counts,
            "fixed_shoot_success_rate": baseline_shoot_rate,
            "fixed_pass_success_rate": baseline_pass_rate,
            "gain_over_fixed_shoot": success_rate - baseline_shoot_rate,
            "gain_over_fixed_pass": success_rate - baseline_pass_rate,
            "minimum_selected_pass_difference_reward": minimum_pass_difference,
        },
        "gates": gate_values,
        "rows": rows,
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "tactical_plane_only": True,
            "g1_whole_body_rollout_claimed": False,
            "team_champion_promoted": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "retention-exam.json", report)
    return report


def run_two_vs_one_growth_round(
    *,
    output_dir: Path,
    source_checkout: Path,
    skill_bundle: FrozenTacticalSkillBundle,
    acquisition_scenarios: Iterable[TwoVsOnePhysicsScenario] | None = None,
    retention_manifest: TwoVsOneRetentionManifest | None = None,
    config: TwoVsOnePhysicsConfig | None = None,
    thresholds: TwoVsOneGrowthThresholds | None = None,
) -> dict[str, Any]:
    """Persist a complete train/seal/exam round outside the checkout."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("2v1 growth evidence must be new and outside the checkout")
    destination.mkdir(parents=True)
    active = config or TwoVsOnePhysicsConfig()
    acquisition = tuple(acquisition_scenarios or default_two_vs_one_acquisition_scenarios())
    manifest = retention_manifest or default_two_vs_one_retention_manifest()
    acquisition_hashes = {scenario.scenario_hash for scenario in acquisition}
    retention_hashes = {scenario.scenario_hash for scenario in manifest.scenarios}
    if len(acquisition) < 8 or acquisition_hashes & retention_hashes:
        raise ValueError("2v1 acquisition and retention must be sufficient and disjoint")

    # Seal and persist the private suite before any actor fit.
    _write_json(destination / "sealed-retention.json", manifest.to_dict())
    evidence, ledger = collect_two_vs_one_acquisition(
        scenarios=acquisition,
        skill_bundle=skill_bundle,
        config=active,
    )
    _write_json(
        destination / "acquisition-ledger.json",
        {
            "schema_version": "rosclaw_soccer.two_vs_one_acquisition_ledger.v1",
            "skill_bundle_hash": skill_bundle.bundle_hash,
            "physics_config_hash": active.config_hash,
            "retention_manifest_hash_visible_to_training": False,
            "rows": list(ledger),
            "ledger_hash": hash_json(list(ledger)),
        },
    )
    actor = fit_two_vs_one_tactical_actor(evidence)
    actor_path = destination / "two-vs-one-tactical-actor.json"
    save_two_vs_one_tactical_actor(actor, actor_path)
    retention = evaluate_two_vs_one_retention(
        actor=actor,
        manifest=manifest,
        skill_bundle=skill_bundle,
        output_dir=destination / "retention",
        config=active,
        thresholds=thresholds,
    )
    source_commit = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=checkout,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    stage: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.two_vs_one_growth_stage.v1",
        "status": retention["status"],
        "source_commit": source_commit,
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "training_snapshot_hash": actor.training_snapshot_hash,
        "acquisition_state_count": len(acquisition),
        "acquisition_action_evidence_count": len(evidence),
        "sealed_retention_manifest_hash": manifest.manifest_hash,
        "retention_report_hash": retention["report_hash"],
        "retention_metrics": retention["metrics"],
        "implementation_hashes": {
            "growth": hash_bytes(Path(__file__).read_bytes()),
            "actor": actor.implementation_hash,
        },
        "evidence_boundary": retention["evidence_boundary"],
    }
    stage["stage_hash"] = hash_json(stage)
    _write_json(destination / "stage-summary.json", stage)
    return stage


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_trajectory(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    required = {
        "time",
        "ball_pose",
        "ball_velocity",
        "carrier_position",
        "finisher_position",
        "defender_position",
        "ball_contact_role",
        "control_force",
        "focal_agent_present",
    }
    if required - trajectory.keys() or trajectory["time"].size < 2:
        raise ValueError("2v1 trajectory artifact is incomplete")
    length = len(trajectory["time"])
    if any(len(value) != length for value in trajectory.values()) or any(
        not np.all(np.isfinite(value)) for value in trajectory.values()
    ):
        raise ValueError("2v1 trajectory artifact is invalid")
    return trajectory


def validate_two_vs_one_retention_report(path: Path) -> dict[str, Any]:
    """Validate report integrity, authority and every selected replay artifact."""

    source = path.expanduser().resolve()
    report = _load_json_object(source, "2v1 retention report")
    claimed = report.pop("report_hash", None)
    try:
        rows = report.get("rows")
        gates = report.get("gates")
        boundary = report.get("evidence_boundary")
        if (
            claimed != hash_json(report)
            or report.get("schema_version") != "rosclaw_soccer.two_vs_one_retention_exam.v1"
            or report.get("status") != "PASS_BOUNDED_TACTICAL_RETENTION"
            or not isinstance(rows, list)
            or len(rows) < 16
            or not isinstance(gates, dict)
            or not gates
            or not all(gates.values())
            or not isinstance(boundary, dict)
            or boundary.get("activation_ceiling") != "SIM_ONLY"
            or boundary.get("physics_authority") != "CPU_MUJOCO"
            or boundary.get("tactical_plane_only") is not True
            or boundary.get("g1_whole_body_rollout_claimed") is not False
            or boundary.get("team_champion_promoted") is not False
            or boundary.get("pixels_used_for_scoring") is not False
            or boundary.get("hardware_command_sent") is not False
        ):
            raise ValueError("2v1 retention authority or integrity contract is invalid")
        for index, row_value in enumerate(rows):
            if not isinstance(row_value, dict):
                raise ValueError("2v1 retention row is invalid")
            case_dir = source.parent / f"case-{index:03d}"
            digests: list[str] = []
            for key in ("primary_artifact", "replay_artifact"):
                artifact = row_value.get(key)
                if not isinstance(artifact, dict) or not isinstance(artifact.get("file"), str):
                    raise ValueError("2v1 retention artifact binding is invalid")
                artifact_path = case_dir / artifact["file"]
                if not artifact_path.is_file() or artifact.get("file_hash") != hash_bytes(
                    artifact_path.read_bytes()
                ):
                    raise ValueError("2v1 retention artifact changed")
                digest = trajectory_digest(_load_trajectory(artifact_path))
                if artifact.get("trajectory_digest") != digest:
                    raise ValueError("2v1 retention trajectory semantics changed")
                digests.append(digest)
            if (
                row_value.get("exact_replay") is not True
                or row_value.get("safe") is not True
                or row_value.get("task_succeeded") is not True
                or digests[0] != digests[1]
            ):
                raise ValueError("2v1 retention row did not strictly replay")
    finally:
        report["report_hash"] = claimed
    return report


def validate_two_vs_one_growth_stage(
    path: Path,
    *,
    source_checkout: Path | None = None,
) -> dict[str, Any]:
    """Validate the complete seal → train → retention evidence chain."""

    source = path.expanduser().resolve()
    stage = _load_json_object(source, "2v1 growth stage")
    claimed = stage.pop("stage_hash", None)
    try:
        root = source.parent
        actor_path = root / "two-vs-one-tactical-actor.json"
        actor = load_two_vs_one_tactical_actor(actor_path)
        manifest = _load_json_object(root / "sealed-retention.json", "2v1 retention manifest")
        manifest_claimed = manifest.pop("manifest_hash", None)
        manifest_valid = manifest_claimed == hash_json(manifest)
        manifest["manifest_hash"] = manifest_claimed
        ledger = _load_json_object(root / "acquisition-ledger.json", "2v1 acquisition ledger")
        retention = validate_two_vs_one_retention_report(root / "retention/retention-exam.json")
        implementation_hashes = stage.get("implementation_hashes")
        if (
            claimed != hash_json(stage)
            or stage.get("schema_version") != "rosclaw_soccer.two_vs_one_growth_stage.v1"
            or stage.get("status") != "PASS_BOUNDED_TACTICAL_RETENTION"
            or actor.actor_hash != stage.get("actor_hash")
            or hash_bytes(actor_path.read_bytes()) != stage.get("actor_file_hash")
            or actor.training_snapshot_hash != stage.get("training_snapshot_hash")
            or not manifest_valid
            or manifest_claimed != stage.get("sealed_retention_manifest_hash")
            or manifest.get("training_access_allowed") is not False
            or ledger.get("retention_manifest_hash_visible_to_training") is not False
            or ledger.get("ledger_hash") != hash_json(ledger.get("rows"))
            or retention.get("report_hash") != stage.get("retention_report_hash")
            or retention.get("actor_hash") != actor.actor_hash
            or retention.get("manifest_hash") != manifest_claimed
            or not isinstance(implementation_hashes, dict)
            or implementation_hashes.get("growth") != hash_bytes(Path(__file__).read_bytes())
            or implementation_hashes.get("actor") != actor.implementation_hash
        ):
            raise ValueError("2v1 growth stage authority or integrity contract is invalid")
        if source_checkout is not None:
            checkout = source_checkout.expanduser().resolve()
            current = subprocess.run(
                ("git", "rev-parse", "HEAD"),
                cwd=checkout,
                check=True,
                capture_output=True,
                text=True,
            ).stdout.strip()
            if current != stage.get("source_commit"):
                raise ValueError("2v1 growth stage belongs to another source commit")
    finally:
        stage["stage_hash"] = claimed
    return stage


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "TwoVsOneGrowthThresholds",
    "TwoVsOneRetentionManifest",
    "collect_two_vs_one_acquisition",
    "default_two_vs_one_acquisition_scenarios",
    "default_two_vs_one_retention_manifest",
    "evaluate_two_vs_one_retention",
    "run_two_vs_one_growth_round",
    "validate_two_vs_one_growth_stage",
    "validate_two_vs_one_retention_report",
]
