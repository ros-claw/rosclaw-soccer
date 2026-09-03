"""Fresh role-local holdout for the learned playmaker pass actor."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.playmaker_pass_actor import (
    load_playmaker_pass_actor,
    playmaker_pass_features,
)
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.playmaker_pass_discovery import (
    _run_probe,
    validate_playmaker_pass_discovery,
)


def fresh_playmaker_pass_holdouts() -> tuple[CausalTransitionContext, ...]:
    """Three preregistered, unseen ball-placement/friction combinations."""

    def context(
        case_id: str,
        lane: float,
        target_x: float,
        ball_xy: tuple[float, float],
        friction: float,
    ) -> CausalTransitionContext:
        return CausalTransitionContext(
            case_id,
            (5.10, -0.16406006503921598, 0.0),
            lane,
            target_x,
            ball_xy,
            0.80,
            friction,
        )

    return (
        context("s132.fresh.00", -0.0860, 1.2990, (1.2097, -0.1657), 0.1010),
        context("s132.fresh.01", -0.0890, 1.3020, (1.2107, -0.1667), 0.1017),
        context("s132.fresh.02", -0.0865, 1.2985, (1.2128, -0.1688), 0.1032),
    )


def run_playmaker_pass_holdout_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    discovery_report_path: Path,
    actor_training_report_path: Path,
    actor_path: Path,
    finisher_actor_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    """Execute actor-selected passes with both other role policies frozen."""

    if not 1 <= workers <= 4:
        raise ValueError("playmaker holdout workers must be in [1, 4]")
    discovery_path = discovery_report_path.expanduser().resolve()
    training_path = actor_training_report_path.expanduser().resolve()
    actor_source = actor_path.expanduser().resolve()
    finisher_source = finisher_actor_path.expanduser().resolve()
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    handoff_source = handoff_actor_path.expanduser().resolve()
    discovery = validate_playmaker_pass_discovery(discovery_path)
    training = _bound_report(training_path)
    actor = load_playmaker_pass_actor(actor_source)
    finisher = load_g1_neural_contact_actor(finisher_source)
    teacher = _bound_json(teacher_path)
    handoff = load_contact_handoff_actor(handoff_source)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if (
        discovery.get("status") != "PASS_PLAYMAKER_PASS_DISCOVERY"
        or training.get("status") != "PASS_PLAYMAKER_PASS_DISTILLATION"
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_source.read_bytes())
        or actor.source_discovery_hash != discovery.get("report_hash")
        or actor.frozen_finisher_actor_hash != finisher.actor_hash
        or finisher.body_hash != qualification.body_hash
        or actor.body_hash != qualification.body_hash
        or handoff.body_hash != qualification.body_hash
    ):
        raise ValueError("playmaker holdout lineage changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("playmaker holdout needs one frozen finisher action")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    finisher_action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=finisher_action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("playmaker holdout finisher handoff is invalid")

    contexts = fresh_playmaker_pass_holdouts()
    decisions = tuple(actor.decide(playmaker_pass_features(context)) for context in contexts)
    if not all(decision.accepted and decision.action is not None for decision in decisions):
        raise ValueError("playmaker actor rejected a preregistered fresh holdout")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_holdout_request.v1",
        "partition": "FRESH_S132_ROLE_LOCAL_HOLDOUT",
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "decisions": [asdict(decision) for decision in decisions],
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_source.read_bytes()),
        "actor_training_report_hash": training["report_hash"],
        "source_discovery_report_hash": discovery["report_hash"],
        "frozen_finisher_actor_hash": finisher.actor_hash,
        "frozen_finisher_actor_file_hash": hash_bytes(finisher_source.read_bytes()),
        "frozen_goalkeeper_policy_hash": actor.frozen_goalkeeper_policy_hash,
        "frozen_handoff_actor_hash": handoff.actor_hash,
        "maximum_delivery_error_m": actor.maximum_delivery_error_m,
        "minimum_strict_chain_count": 2,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            finisher_source,
            output,
            2 * index + replay_index,
            context,
            cast(Any, decision.action),
            finisher_action,
            handoff_decision.handoff_policy_frame,
            quality,
            actor.maximum_delivery_error_m,
        )
        for index, (context, decision) in enumerate(zip(contexts, decisions, strict=True))
        for replay_index in range(2)
    )
    if workers == 1:
        runs = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = list(executor.map(_run_probe, jobs))
    rows: list[dict[str, Any]] = []
    for index, (context, decision) in enumerate(zip(contexts, decisions, strict=True)):
        primary = runs[2 * index]
        replay = runs[2 * index + 1]
        rows.append(
            {
                "case_id": context.case_id,
                "context_hash": context.context_hash,
                "decision": asdict(decision),
                "primary": primary,
                "replay": replay,
                "exact_replay": bool(
                    primary["result"] == replay["result"]
                    and primary["trajectory"]["trajectory_digest"]
                    == replay["trajectory"]["trajectory_digest"]
                ),
                "safe": bool(primary["quality"]["safe"]),
                "ordered": bool(primary["quality"]["ordered_contacts"]),
                "delivery_passed": bool(primary["playmaker_pass_success"]),
                "strict_chain": bool(primary["quality"]["strict_chain_passed"]),
                "clear_outcome": bool(primary["quality"]["clear_outcome"]),
            }
        )
    gates = _derive_gates(
        rows,
        minimum_strict_chain_count=2,
        maximum_delivery_error_m=actor.maximum_delivery_error_m,
    )
    observed_delivery_errors = [
        float(value)
        for row in rows
        if isinstance((value := row["primary"]["result"]["pass_delivery_error_m"]), int | float)
        and not isinstance(value, bool)
    ]
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.playmaker_pass_holdout_exam.v1",
        "status": (
            "PASS_PLAYMAKER_PASS_FRESH_HOLDOUT"
            if all(gates.values())
            else "REJECTED_PLAYMAKER_PASS_FRESH_HOLDOUT"
        ),
        "promotion_eligible": False,
        "claim": "FRESH_ROLE_LOCAL_PLAYMAKER_GENERALIZATION_WITH_FROZEN_TEAMMATE_AND_OPPONENT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "actor_hash": actor.actor_hash,
        "metrics": {
            "case_count": len(rows),
            "safe_count": sum(int(row["safe"]) for row in rows),
            "delivery_pass_count": sum(int(row["delivery_passed"]) for row in rows),
            "strict_chain_count": sum(int(row["strict_chain"]) for row in rows),
            "goal_count": sum(int(row["primary"]["result"]["goal_crossed"]) for row in rows),
            "save_count": sum(
                int(row["primary"]["result"]["goalkeeper_save_observed"]) for row in rows
            ),
            "delivery_observed_count": len(observed_delivery_errors),
            "mean_observed_delivery_error_m": (
                sum(observed_delivery_errors) / len(observed_delivery_errors)
                if observed_delivery_errors
                else None
            ),
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "holdout-report.json", report)
    return report


def validate_playmaker_pass_holdout_exam(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = _bound_report(source)
    request_path = source.parent / "request.json"
    request = _read_object(request_path)
    rows = report.get("rows")
    if not isinstance(rows, list) or len(rows) != 3:
        raise ValueError("playmaker holdout rows are invalid")
    minimum = request.get("minimum_strict_chain_count")
    if not isinstance(minimum, int) or isinstance(minimum, bool):
        raise ValueError("playmaker holdout strict-chain gate is invalid")
    threshold = request.get("maximum_delivery_error_m")
    if not isinstance(threshold, int | float) or isinstance(threshold, bool):
        raise ValueError("playmaker holdout delivery gate is invalid")
    gates = _derive_gates(
        rows,
        minimum_strict_chain_count=minimum,
        maximum_delivery_error_m=float(threshold),
    )
    expected = (
        "PASS_PLAYMAKER_PASS_FRESH_HOLDOUT"
        if all(gates.values())
        else "REJECTED_PLAYMAKER_PASS_FRESH_HOLDOUT"
    )
    if (
        hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or report.get("implementation_hash") != _implementation_hash()
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("gates") != gates
        or report.get("status") != expected
        or report.get("promotion_eligible") is not False
        or report.get("physics_authority") != "CPU_MUJOCO"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
        or report.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("playmaker holdout authority is invalid")
    for row in rows:
        for name in ("primary", "replay"):
            artifact = row[name]["trajectory"]
            if hash_bytes((source.parent / artifact["file"]).read_bytes()) != artifact["file_hash"]:
                raise ValueError("playmaker holdout trajectory changed")
    return report


def _derive_gates(
    rows: list[dict[str, Any]],
    *,
    minimum_strict_chain_count: int,
    maximum_delivery_error_m: float,
) -> dict[str, bool]:
    def summary(row: dict[str, Any]) -> dict[str, bool]:
        primary = row["primary"]
        replay = row["replay"]
        result = primary["result"]
        quality = primary["quality"]
        error = result.get("pass_delivery_error_m")
        safe = quality.get("safe") is True
        ordered = quality.get("ordered_contacts") is True
        return {
            "exact_replay": bool(
                primary["result"] == replay["result"]
                and primary["trajectory"]["trajectory_digest"]
                == replay["trajectory"]["trajectory_digest"]
            ),
            "safe": safe,
            "ordered": ordered,
            "delivery_passed": bool(
                safe
                and ordered
                and isinstance(error, int | float)
                and not isinstance(error, bool)
                and float(error) <= maximum_delivery_error_m
            ),
            "strict_chain": quality.get("strict_chain_passed") is True,
            "clear_outcome": quality.get("clear_outcome") is True,
        }

    summaries = [summary(row) for row in rows]
    return {
        "all_actor_decisions_accepted": all(row["decision"]["accepted"] for row in rows),
        "all_exact_replay": all(value["exact_replay"] for value in summaries),
        "all_safe": all(value["safe"] for value in summaries),
        "all_ordered": all(value["ordered"] for value in summaries),
        "all_delivery_passed": all(value["delivery_passed"] for value in summaries),
        "minimum_two_strict_complete_chains": sum(int(value["strict_chain"]) for value in summaries)
        >= minimum_strict_chain_count,
        "all_clear_team_outcomes": all(value["clear_outcome"] for value in summaries),
        "row_summaries_match_raw": all(
            all(row[name] is expected for name, expected in value.items())
            for row, value in zip(rows, summaries, strict=True)
        ),
        "finisher_actor_executed_all": all(
            row["primary"]["finisher_actor_active"] and row["replay"]["finisher_actor_active"]
            for row in rows
        ),
        "no_teacher_or_scripted_contact": all(
            not row[name]["teacher_active"] and not row[name]["scripted_contact_active"]
            for row in rows
            for name in ("primary", "replay")
        ),
    }


def _bound_report(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    claimed = value.pop("report_hash", None)
    if claimed != hash_json(value):
        raise ValueError(f"{path.name} integrity changed")
    value["report_hash"] = claimed
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "playmaker_pass_actor.py",
        Path(__file__).parent / "playmaker_pass_discovery.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("playmaker holdout output must use a new external directory")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = [
    "fresh_playmaker_pass_holdouts",
    "run_playmaker_pass_holdout_exam",
    "validate_playmaker_pass_holdout_exam",
]
