"""Seal, learn, and examine the full-body G1 2v1 tactical bridge."""

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
from rosclaw_soccer.training.full_body_tactical_2v1 import (
    FullBodyTwoVsOneConfig,
    FullBodyTwoVsOneResult,
    FullBodyTwoVsOneScenario,
    matched_full_body_two_vs_one_decision,
    simulate_full_body_two_vs_one,
)
from rosclaw_soccer.training.tactical_2v1_physics import FrozenTacticalSkillBundle

_ACTIONS = (TacticalAction.PASS, TacticalAction.SHOOT)


@dataclass(frozen=True)
class FullBodyTwoVsOneRetentionManifest:
    scenarios: tuple[FullBodyTwoVsOneScenario, ...]
    suite_id: str = "s120.full-body-two-vs-one.sealed-retention"
    training_access_allowed: bool = False
    activation_ceiling: str = "SIM_ONLY"
    schema_version: str = "rosclaw_soccer.full_body_2v1_retention_manifest.v1"

    def __post_init__(self) -> None:
        hashes = tuple(scenario.scenario_hash for scenario in self.scenarios)
        if (
            len(self.scenarios) < 8
            or len(set(hashes)) != len(hashes)
            or self.training_access_allowed
            or self.activation_ceiling != "SIM_ONLY"
        ):
            raise ValueError("full-body retention must be unique, sealed and SIM_ONLY")

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
class FullBodyTwoVsOneThresholds:
    minimum_task_success_rate: float = 1.0
    minimum_action_agreement_rate: float = 1.0
    minimum_exact_replay_rate: float = 1.0
    maximum_mean_regret: float = 0.05
    maximum_case_regret: float = 0.20
    minimum_action_count: int = 4
    schema_version: str = "rosclaw_soccer.full_body_2v1_thresholds.v1"

    def __post_init__(self) -> None:
        values = (
            self.minimum_task_success_rate,
            self.minimum_action_agreement_rate,
            self.minimum_exact_replay_rate,
            self.maximum_mean_regret,
            self.maximum_case_regret,
        )
        if any(not math.isfinite(value) for value in values) or not (
            0.80 <= self.minimum_task_success_rate <= 1.0
            and 0.80 <= self.minimum_action_agreement_rate <= 1.0
            and 0.80 <= self.minimum_exact_replay_rate <= 1.0
            and 0.0 <= self.maximum_mean_regret <= self.maximum_case_regret <= 0.50
            and 1 <= self.minimum_action_count <= 32
        ):
            raise ValueError("full-body 2v1 thresholds are invalid")


def default_full_body_acquisition_scenarios() -> tuple[FullBodyTwoVsOneScenario, ...]:
    """Eight balanced states inside the measured low-level initiation sets."""

    layouts = (
        ((5.42, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.46, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.54, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.58, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.30, 0.30, 0.0), (4.05, 0.30, 0.0)),
        ((5.40, 0.40, 0.0), (4.25, 0.40, 0.0)),
        ((5.60, 0.50, 0.0), (4.65, 0.50, 0.0)),
        ((5.70, 0.60, 0.0), (4.85, 0.60, 0.0)),
    )
    return tuple(
        FullBodyTwoVsOneScenario(
            scenario_id=f"s120.acquisition.{index:03d}",
            seed=120_000 + index,
            teammate_origin_m=teammate,
            defender_origin_m=defender,
        )
        for index, (teammate, defender) in enumerate(layouts)
    )


def default_full_body_retention_manifest() -> FullBodyTwoVsOneRetentionManifest:
    """Return eight unseen states sealed before actor fitting."""

    layouts = (
        ((5.44, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.48, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.52, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.56, -0.40, 0.0), (2.00, 0.40, 0.0)),
        ((5.35, 0.35, 0.0), (4.15, 0.35, 0.0)),
        ((5.45, 0.45, 0.0), (4.35, 0.45, 0.0)),
        ((5.55, 0.55, 0.0), (4.55, 0.55, 0.0)),
        ((5.65, 0.55, 0.0), (4.75, 0.45, 0.0)),
    )
    scenarios = tuple(
        FullBodyTwoVsOneScenario(
            scenario_id=f"s120.retention.{index:03d}",
            seed=120_500 + index,
            teammate_origin_m=teammate,
            defender_origin_m=defender,
        )
        for index, (teammate, defender) in enumerate(layouts)
    )
    return FullBodyTwoVsOneRetentionManifest(scenarios=scenarios)


def _policy_hash(label: str, bundle: FrozenTacticalSkillBundle) -> str:
    return str(hash_json({"policy": label, "skill_bundle_hash": bundle.bundle_hash}))


def collect_full_body_acquisition(
    *,
    asset_root: Path,
    scenarios: Iterable[FullBodyTwoVsOneScenario],
    skill_bundle: FrozenTacticalSkillBundle,
    config: FullBodyTwoVsOneConfig | None = None,
) -> tuple[tuple[TwoVsOneDecisionEvidence, ...], tuple[dict[str, Any], ...]]:
    """Physically execute both options without exposing retention layouts."""

    active = config or FullBodyTwoVsOneConfig()
    policy_hash = _policy_hash("exhaustive_full_body_teacher_v1", skill_bundle)
    evidence: list[TwoVsOneDecisionEvidence] = []
    ledger: list[dict[str, Any]] = []
    for scenario in scenarios:
        actions: dict[str, Any] = {}
        for action in _ACTIONS:
            item, primary, ablated, _, _ = matched_full_body_two_vs_one_decision(
                asset_root=asset_root,
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
                "state": asdict(scenario.state(skill_bundle=skill_bundle, config=active)),
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


def _qualified_success(result: FullBodyTwoVsOneResult) -> bool:
    return bool(result.task_succeeded and result.safe)


def evaluate_full_body_retention(
    *,
    asset_root: Path,
    actor: TwoVsOneTacticalActor,
    manifest: FullBodyTwoVsOneRetentionManifest,
    skill_bundle: FrozenTacticalSkillBundle,
    output_dir: Path,
    config: FullBodyTwoVsOneConfig | None = None,
    thresholds: FullBodyTwoVsOneThresholds | None = None,
) -> dict[str, Any]:
    """Run selected, alternative, focal ablation, and exact replay per state."""

    destination = output_dir.expanduser().resolve()
    destination.mkdir(parents=True, exist_ok=False)
    active = config or FullBodyTwoVsOneConfig()
    gates = thresholds or FullBodyTwoVsOneThresholds()
    rows: list[dict[str, Any]] = []
    selected_counts = {action.value: 0 for action in _ACTIONS}
    for index, scenario in enumerate(manifest.scenarios):
        state = scenario.state(skill_bundle=skill_bundle, config=active)
        decision = actor.decide(state)
        action_runs = {
            action: matched_full_body_two_vs_one_decision(
                asset_root=asset_root,
                scenario=scenario,
                action=action,
                policy_hash=actor.actor_hash,
                skill_bundle=skill_bundle,
                config=active,
            )
            for action in _ACTIONS
        }
        if decision.action not in action_runs:
            raise ValueError("sealed full-body state fell outside actor support")
        selected_counts[decision.action.value] += 1
        selected = action_runs[decision.action]
        primary = selected[1]
        primary_trace = selected[3]
        replay, replay_trace = simulate_full_body_two_vs_one(
            asset_root=asset_root,
            scenario=scenario,
            action=decision.action,
            skill_bundle=skill_bundle,
            config=active,
            focal_teammate_present=True,
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
                "decision": {**asdict(decision), "action": decision.action.value},
                "selected_action": decision.action.value,
                "oracle_action": oracle.value,
                "action_agreement": decision.action is oracle,
                "regret": regret,
                "task_succeeded": _qualified_success(primary),
                "safe": primary.safe,
                "exact_replay": exact_replay,
                "selected_evidence_hash": selected[0].evidence_hash,
                "selected_weighted_score": selected[0].weighted_score,
                "selected_difference_reward": selected[0].rollout.difference_reward,
                "pass_score": scores[TacticalAction.PASS],
                "shoot_score": scores[TacticalAction.SHOOT],
                "pass_task_succeeded": _qualified_success(action_runs[TacticalAction.PASS][1]),
                "shoot_task_succeeded": _qualified_success(action_runs[TacticalAction.SHOOT][1]),
                "primary_result": primary.to_dict(),
                "primary_artifact": primary_artifact,
                "replay_artifact": replay_artifact,
            }
        )

    count = len(rows)
    success_rate = sum(bool(row["task_succeeded"]) for row in rows) / count
    agreement_rate = sum(bool(row["action_agreement"]) for row in rows) / count
    replay_rate = sum(bool(row["exact_replay"]) for row in rows) / count
    safe_rate = sum(bool(row["safe"]) for row in rows) / count
    regrets = [float(row["regret"]) for row in rows]
    pass_difference = [
        float(row["selected_difference_reward"])
        for row in rows
        if row["selected_action"] == TacticalAction.PASS.value
    ]
    fixed_shoot = sum(bool(row["shoot_task_succeeded"]) for row in rows) / count
    fixed_pass = sum(bool(row["pass_task_succeeded"]) for row in rows) / count
    minimum_pass_difference = min(pass_difference) if pass_difference else None
    gate_values = {
        "task_success_rate": success_rate >= gates.minimum_task_success_rate,
        "action_agreement_rate": agreement_rate >= gates.minimum_action_agreement_rate,
        "exact_replay_rate": replay_rate >= gates.minimum_exact_replay_rate,
        "safe_rate": safe_rate == 1.0,
        "mean_regret": float(np.mean(regrets)) <= gates.maximum_mean_regret,
        "maximum_case_regret": max(regrets) <= gates.maximum_case_regret,
        "both_actions_selected": all(
            selected_counts[action.value] >= gates.minimum_action_count for action in _ACTIONS
        ),
        "pass_counterfactual_positive": minimum_pass_difference is not None
        and minimum_pass_difference > 0.0,
        "beats_fixed_shoot": success_rate > fixed_shoot,
        "beats_fixed_pass": success_rate > fixed_pass,
    }
    passed = all(gate_values.values())
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.full_body_2v1_retention_exam.v1",
        "status": (
            "PASS_FULL_BODY_TACTICAL_BRIDGE" if passed else "REJECTED_FULL_BODY_TACTICAL_BRIDGE"
        ),
        "actor_hash": actor.actor_hash,
        "training_snapshot_hash": actor.training_snapshot_hash,
        "manifest_hash": manifest.manifest_hash,
        "skill_bundle_hash": skill_bundle.bundle_hash,
        "physics_config_hash": active.config_hash,
        "thresholds": asdict(gates),
        "metrics": {
            "case_count": count,
            "task_success_rate": success_rate,
            "action_agreement_rate": agreement_rate,
            "exact_replay_rate": replay_rate,
            "safe_rate": safe_rate,
            "mean_regret": float(np.mean(regrets)),
            "maximum_case_regret": max(regrets),
            "selected_action_counts": selected_counts,
            "fixed_shoot_success_rate": fixed_shoot,
            "fixed_pass_success_rate": fixed_pass,
            "gain_over_fixed_shoot": success_rate - fixed_shoot,
            "gain_over_fixed_pass": success_rate - fixed_pass,
            "minimum_selected_pass_difference_reward": minimum_pass_difference,
        },
        "gates": gate_values,
        "rows": rows,
        "evidence_boundary": {
            "activation_ceiling": "SIM_ONLY",
            "physics_authority": "CPU_MUJOCO",
            "whole_body_g1_count": 3,
            "shared_solver_and_ball": True,
            "pass_termination": "CONTROLLED_FOOT_RECEPTION",
            "one_touch_finish_claimed": False,
            "continuous_match_claimed": False,
            "team_champion_promoted": False,
            "pixels_used_for_scoring": False,
            "hardware_command_sent": False,
        },
    }
    report["report_hash"] = hash_json(report)
    _write_json(destination / "retention-exam.json", report)
    return report


def run_full_body_tactical_growth_round(
    *,
    output_dir: Path,
    source_checkout: Path,
    asset_root: Path,
    skill_bundle: FrozenTacticalSkillBundle,
    acquisition_scenarios: Iterable[FullBodyTwoVsOneScenario] | None = None,
    retention_manifest: FullBodyTwoVsOneRetentionManifest | None = None,
    config: FullBodyTwoVsOneConfig | None = None,
    thresholds: FullBodyTwoVsOneThresholds | None = None,
) -> dict[str, Any]:
    """Persist one sealed SIM-only full-body bridge round outside the checkout."""

    destination = output_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    assets = asset_root.expanduser().resolve()
    if destination.exists() or destination == checkout or checkout in destination.parents:
        raise ValueError("full-body evidence must be new and outside the checkout")
    destination.mkdir(parents=True)
    active = config or FullBodyTwoVsOneConfig()
    acquisition = tuple(acquisition_scenarios or default_full_body_acquisition_scenarios())
    manifest = retention_manifest or default_full_body_retention_manifest()
    acquisition_hashes = {scenario.scenario_hash for scenario in acquisition}
    retention_hashes = {scenario.scenario_hash for scenario in manifest.scenarios}
    if len(acquisition) < 8 or acquisition_hashes & retention_hashes:
        raise ValueError("full-body acquisition and retention must be sufficient and disjoint")

    # The private exam is committed before any actor fit.  Training gets only
    # a boolean saying that a manifest exists, never its scenarios or hash.
    _write_json(destination / "sealed-retention.json", manifest.to_dict())
    evidence, ledger = collect_full_body_acquisition(
        asset_root=assets,
        scenarios=acquisition,
        skill_bundle=skill_bundle,
        config=active,
    )
    _write_json(
        destination / "acquisition-ledger.json",
        {
            "schema_version": "rosclaw_soccer.full_body_2v1_acquisition_ledger.v1",
            "skill_bundle": asdict(skill_bundle),
            "skill_bundle_hash": skill_bundle.bundle_hash,
            "physics_config_hash": active.config_hash,
            "retention_manifest_visible_to_training": False,
            "rows": list(ledger),
            "ledger_hash": hash_json(list(ledger)),
        },
    )
    actor = fit_two_vs_one_tactical_actor(evidence)
    actor_path = destination / "full-body-tactical-actor.json"
    save_two_vs_one_tactical_actor(actor, actor_path)
    retention = evaluate_full_body_retention(
        asset_root=assets,
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
        "schema_version": "rosclaw_soccer.full_body_2v1_growth_stage.v1",
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
            "bridge": hash_bytes(
                Path(__file__).with_name("full_body_tactical_2v1.py").read_bytes()
            ),
            "growth": hash_bytes(Path(__file__).read_bytes()),
            "actor": actor.implementation_hash,
        },
        "evidence_boundary": retention["evidence_boundary"],
    }
    stage["stage_hash"] = hash_json(stage)
    _write_json(destination / "stage-summary.json", stage)
    return stage


def validate_full_body_tactical_growth_stage(
    evidence_dir: Path,
    *,
    source_checkout: Path,
) -> dict[str, Any]:
    """Recompute the durable commitments and reject edited S120 evidence."""

    root = evidence_dir.expanduser().resolve()
    checkout = source_checkout.expanduser().resolve()
    errors: list[str] = []
    try:
        manifest = _load_json_object(root / "sealed-retention.json", "manifest")
        ledger = _load_json_object(root / "acquisition-ledger.json", "ledger")
        retention = _load_json_object(root / "retention/retention-exam.json", "retention")
        stage = _load_json_object(root / "stage-summary.json", "stage")
        actor_path = root / "full-body-tactical-actor.json"
        actor = load_two_vs_one_tactical_actor(actor_path)
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        return {
            "status": "REJECTED_FULL_BODY_TACTICAL_STAGE",
            "errors": [f"load_error:{type(exc).__name__}:{exc}"],
        }

    def committed(payload: dict[str, Any], key: str, label: str) -> None:
        claimed = payload.get(key)
        body = dict(payload)
        body.pop(key, None)
        if claimed != hash_json(body):
            errors.append(f"{label}_hash_mismatch")

    committed(manifest, "manifest_hash", "manifest")
    committed(retention, "report_hash", "retention")
    committed(stage, "stage_hash", "stage")
    rows = ledger.get("rows")
    if not isinstance(rows, list) or ledger.get("ledger_hash") != hash_json(rows):
        errors.append("acquisition_ledger_hash_mismatch")
    if (
        manifest.get("training_access_allowed") is not False
        or ledger.get("retention_manifest_visible_to_training") is not False
        or manifest.get("activation_ceiling") != "SIM_ONLY"
    ):
        errors.append("sealed_exam_boundary_invalid")
    bundle_payload = ledger.get("skill_bundle")
    try:
        if not isinstance(bundle_payload, dict):
            raise ValueError("missing frozen bundle")
        bundle = FrozenTacticalSkillBundle(**bundle_payload)
        if bundle.bundle_hash != ledger.get(
            "skill_bundle_hash"
        ) or bundle.bundle_hash != retention.get("skill_bundle_hash"):
            errors.append("skill_bundle_hash_mismatch")
    except (TypeError, ValueError):
        errors.append("skill_bundle_invalid")
    if (
        stage.get("actor_hash") != actor.actor_hash
        or stage.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or retention.get("actor_hash") != actor.actor_hash
        or retention.get("training_snapshot_hash") != actor.training_snapshot_hash
    ):
        errors.append("actor_commitment_mismatch")
    implementation = stage.get("implementation_hashes")
    bridge_path = Path(__file__).with_name("full_body_tactical_2v1.py")
    if not isinstance(implementation, dict) or (
        implementation.get("bridge") != hash_bytes(bridge_path.read_bytes())
        or implementation.get("growth") != hash_bytes(Path(__file__).read_bytes())
        or implementation.get("actor") != actor.implementation_hash
    ):
        errors.append("implementation_hash_mismatch")
    retention_rows = retention.get("rows")
    if not isinstance(retention_rows, list) or len(retention_rows) < 8:
        errors.append("retention_rows_invalid")
    else:
        for index, row in enumerate(retention_rows):
            if not isinstance(row, dict):
                errors.append(f"retention_row_{index}_invalid")
                continue
            for artifact_key in ("primary_artifact", "replay_artifact"):
                artifact = row.get(artifact_key)
                if not isinstance(artifact, dict):
                    errors.append(f"retention_row_{index}_{artifact_key}_invalid")
                    continue
                path = root / "retention" / f"case-{index:03d}" / str(artifact.get("file"))
                try:
                    trajectory = _load_trajectory(path)
                    if artifact.get("file_hash") != hash_bytes(path.read_bytes()) or artifact.get(
                        "trajectory_digest"
                    ) != trajectory_digest(trajectory):
                        errors.append(f"retention_row_{index}_{artifact_key}_hash_mismatch")
                except (OSError, ValueError):
                    errors.append(f"retention_row_{index}_{artifact_key}_unreadable")
            if (
                row.get("primary_artifact", {}).get("trajectory_digest")
                != row.get("replay_artifact", {}).get("trajectory_digest")
                or row.get("exact_replay") is not True
            ):
                errors.append(f"retention_row_{index}_replay_mismatch")
    boundary = retention.get("evidence_boundary")
    if not isinstance(boundary, dict) or (
        boundary.get("whole_body_g1_count") != 3
        or boundary.get("shared_solver_and_ball") is not True
        or boundary.get("pass_termination") != "CONTROLLED_FOOT_RECEPTION"
        or boundary.get("continuous_match_claimed") is not False
        or boundary.get("hardware_command_sent") is not False
    ):
        errors.append("evidence_boundary_invalid")
    if retention.get("status") != "PASS_FULL_BODY_TACTICAL_BRIDGE" or not all(
        retention.get("gates", {}).values()
    ):
        errors.append("retention_not_passed")
    try:
        ancestor = subprocess.run(
            ("git", "merge-base", "--is-ancestor", str(stage.get("source_commit")), "HEAD"),
            cwd=checkout,
            check=False,
        )
        if ancestor.returncode != 0:
            errors.append("source_commit_not_in_checkout_history")
    except OSError:
        errors.append("source_commit_unverifiable")
    return {
        "status": (
            "VALIDATED_FULL_BODY_TACTICAL_STAGE"
            if not errors
            else "REJECTED_FULL_BODY_TACTICAL_STAGE"
        ),
        "errors": errors,
        "stage_hash": stage.get("stage_hash"),
        "actor_hash": actor.actor_hash,
        "retention_report_hash": retention.get("report_hash"),
    }


def _load_json_object(path: Path, label: str) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _load_trajectory(path: Path) -> dict[str, NDArray[Any]]:
    with np.load(path, allow_pickle=False) as archive:
        trajectory = {name: np.asarray(archive[name]) for name in archive.files}
    if "time" not in trajectory or trajectory["time"].size < 2:
        raise ValueError("full-body trajectory is incomplete")
    return trajectory


def _write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "FullBodyTwoVsOneRetentionManifest",
    "FullBodyTwoVsOneThresholds",
    "collect_full_body_acquisition",
    "default_full_body_acquisition_scenarios",
    "default_full_body_retention_manifest",
    "evaluate_full_body_retention",
    "run_full_body_tactical_growth_round",
    "validate_full_body_tactical_growth_stage",
]
