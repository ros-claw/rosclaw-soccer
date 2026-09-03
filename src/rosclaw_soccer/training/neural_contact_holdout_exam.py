"""Fresh local holdout exam for the S130 neural contact muscle memory."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionContext,
    CausalTransitionGrowthConfig,
    _load_lead_policy,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json, _run


def fresh_neural_contact_holdouts() -> tuple[CausalTransitionContext, ...]:
    def context(
        case_id: str,
        receiver_lane_m: float,
        reception_target_x_m: float,
        passer_ball_local_xy_m: tuple[float, float],
        ball_ground_friction: float,
    ) -> CausalTransitionContext:
        return CausalTransitionContext(
            case_id,
            (5.10, -0.16406006503921598, 0.0),
            receiver_lane_m,
            reception_target_x_m,
            passer_ball_local_xy_m,
            0.80,
            ball_ground_friction,
        )

    return (
        context(
            "s131.sealed.00",
            -0.084,
            1.3005,
            (1.2115, -0.1675),
            0.1022,
        ),
        context(
            "s131.sealed.01",
            -0.091,
            1.3005,
            (1.2115, -0.1675),
            0.1022,
        ),
        context(
            "s131.sealed.02",
            -0.0875,
            1.296,
            (1.2115, -0.1675),
            0.1022,
        ),
        context(
            "s131.sealed.03",
            -0.0875,
            1.305,
            (1.2115, -0.1675),
            0.1022,
        ),
        context(
            "s131.sealed.04",
            -0.0875,
            1.3005,
            (1.2090, -0.1650),
            0.1005,
        ),
        context(
            "s131.sealed.05",
            -0.0875,
            1.3005,
            (1.2140, -0.1700),
            0.1040,
        ),
    )


def run_neural_contact_holdout_exam(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    source_canary_report_path: Path,
    teacher_discovery_report_path: Path,
    actor_training_report_path: Path,
    actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 4,
) -> dict[str, Any]:
    if not 1 <= workers <= 4:
        raise ValueError("neural contact holdout workers must be in [1, 4]")
    canary_path = source_canary_report_path.expanduser().resolve()
    canary = _bound_json(canary_path)
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    teacher = _bound_json(teacher_path)
    training_path = actor_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    actor = load_g1_neural_contact_actor(actor_path)
    handoff = load_contact_handoff_actor(handoff_actor_path)
    if (
        canary.get("status") != "PASS_NEURAL_CONTACT_CANARY"
        or not all(canary.get("gates", {}).values())
        or canary.get("actor_hash") != actor.actor_hash
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or training.get("source_teacher_report_hash") != teacher["report_hash"]
        or handoff.body_hash != actor.body_hash
    ):
        raise ValueError("neural contact holdout lineage changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    if len(source_rows) != 1:
        raise ValueError("neural contact holdout requires one source action")
    payload = dict(source_rows[0]["action"])
    payload["target_foot_velocity_xyz_mps"] = tuple(payload["target_foot_velocity_xyz_mps"])
    action = TargetContactPlanAction(**payload)
    handoff_decision = handoff.decide(contact_policy_frame=action.contact_policy_frame)
    if not handoff_decision.accepted or handoff_decision.handoff_policy_frame is None:
        raise ValueError("neural contact holdout handoff was rejected")
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if qualification.body_hash != actor.body_hash:
        raise ValueError("neural contact holdout Body identity changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    _, source = _load_lead_policy(source_s95_dir)
    contexts = fresh_neural_contact_holdouts()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_holdout_request.v1",
        "partition": "FRESH_S131_LOCAL_HOLDOUT",
        "contexts": [asdict(context) for context in contexts],
        "context_hashes": [context.context_hash for context in contexts],
        "action": asdict(action),
        "source_canary_report_hash": canary["report_hash"],
        "source_canary_file_hash": hash_bytes(canary_path.read_bytes()),
        "teacher_discovery_report_hash": teacher["report_hash"],
        "actor_training_report_hash": training["report_hash"],
        "actor_hash": actor.actor_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "contact_handoff_actor_hash": handoff.actor_hash,
        "source_s95_evidence_hash": source["evidence_hash"],
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "body_hash": qualification.body_hash,
        "implementation_hash": _implementation_hash(),
        "minimum_goal_count": 4,
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            actor_path.expanduser().resolve(),
            output,
            f"case-{index:02d}-{suffix}",
            context,
            action,
            handoff_decision.handoff_policy_frame,
            quality,
            True,
        )
        for index, context in enumerate(contexts)
        for suffix in ("primary", "replay")
    )
    if workers == 1:
        runs = [_run(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            runs = list(executor.map(_run, jobs))
    rows: list[dict[str, Any]] = []
    for index, context in enumerate(contexts):
        primary = runs[2 * index]
        replay = runs[2 * index + 1]
        exact = bool(
            primary["result"] == replay["result"]
            and primary["trajectory"]["trajectory_digest"]
            == replay["trajectory"]["trajectory_digest"]
        )
        rows.append(
            {
                "case_id": context.case_id,
                "context_hash": context.context_hash,
                "primary": primary,
                "replay": replay,
                "exact_replay": exact,
                "safe": bool(primary["quality"]["safe"]),
                "strict_right_foot_chain": bool(primary["quality"]["strict_chain_passed"]),
                "goal": bool(primary["result"]["goal_crossed"]),
            }
        )
    goal_count = sum(int(row["goal"]) for row in rows)
    gates = {
        "all_exact_replay": all(row["exact_replay"] for row in rows),
        "all_safe": all(row["safe"] for row in rows),
        "all_strict_right_foot_chain": all(row["strict_right_foot_chain"] for row in rows),
        "minimum_four_goals": goal_count >= 4,
        "actor_executed_all": all(row["primary"]["actor_active"] for row in rows),
        "actor_sole_contact_residual": all(
            not row["primary"]["teacher_active"] and not row["primary"]["scripted_contact_active"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_holdout_exam.v1",
        "status": (
            "PASS_NEURAL_CONTACT_LOCAL_HOLDOUT"
            if all(gates.values())
            else "REJECTED_NEURAL_CONTACT_LOCAL_HOLDOUT"
        ),
        "promotion_eligible": False,
        "partition": "FRESH_S131_LOCAL_HOLDOUT",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "actor_hash": actor.actor_hash,
        "metrics": {
            "case_count": len(rows),
            "goal_count": goal_count,
            "safe_count": sum(int(row["safe"]) for row in rows),
            "strict_right_foot_chain_count": sum(
                int(row["strict_right_foot_chain"]) for row in rows
            ),
            "minimum_shot_speed_mps": min(
                row["primary"]["result"]["shot_peak_ball_speed_mps"] for row in rows
            ),
            "maximum_tail_wobble_index": max(
                row["primary"]["result"]["shooter_tail_wobble_index"] for row in rows
            ),
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "holdout-report.json", report)
    return report


def validate_neural_contact_holdout_exam(path: Path) -> dict[str, Any]:
    source = path.expanduser().resolve()
    report = _bound_json(source)
    request_path = source.parent / "request.json"
    request = _bound_unhashed_json(request_path)
    rows = report.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError("neural contact holdout rows are invalid")
    minimum_goal_count = request.get("minimum_goal_count")
    if not isinstance(minimum_goal_count, int) or isinstance(minimum_goal_count, bool):
        raise ValueError("neural contact holdout goal gate is invalid")
    goal_count = sum(int(isinstance(row, dict) and row.get("goal") is True) for row in rows)
    derived_gates = {
        "all_exact_replay": all(
            isinstance(row, dict) and row.get("exact_replay") is True for row in rows
        ),
        "all_safe": all(isinstance(row, dict) and row.get("safe") is True for row in rows),
        "all_strict_right_foot_chain": all(
            isinstance(row, dict) and row.get("strict_right_foot_chain") is True for row in rows
        ),
        "minimum_four_goals": goal_count >= minimum_goal_count,
        "actor_executed_all": all(
            isinstance(row, dict)
            and isinstance(row.get("primary"), dict)
            and row["primary"].get("actor_active") is True
            for row in rows
        ),
        "actor_sole_contact_residual": all(
            isinstance(row, dict)
            and isinstance(row.get("primary"), dict)
            and row["primary"].get("teacher_active") is False
            and row["primary"].get("scripted_contact_active") is False
            for row in rows
        ),
    }
    expected_status = (
        "PASS_NEURAL_CONTACT_LOCAL_HOLDOUT"
        if all(derived_gates.values())
        else "REJECTED_NEURAL_CONTACT_LOCAL_HOLDOUT"
    )
    if (
        request.get("schema_version") != "rosclaw.growth.neural_contact_holdout_request.v1"
        or request.get("partition") != "FRESH_S131_LOCAL_HOLDOUT"
        or request.get("activation_ceiling") != "SIM_ONLY"
        or request.get("hardware_command_sent") is not False
        or request.get("teacher_enabled") is not False
        or request.get("scripted_contact_torque_enabled") is not False
        or request.get("pixels_used_for_scoring") is not False
        or report.get("schema_version") != "rosclaw.growth.neural_contact_holdout_exam.v1"
        or report.get("status") != expected_status
        or report.get("gates") != derived_gates
        or report.get("partition") != "FRESH_S131_LOCAL_HOLDOUT"
        or report.get("promotion_eligible") is not False
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
        or report.get("implementation_hash") != _implementation_hash()
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("actor_hash") != request.get("actor_hash")
    ):
        raise ValueError("neural contact holdout authority is invalid")
    if hash_bytes(request_path.read_bytes()) != report["request_hash"]:
        raise ValueError("neural contact holdout request changed")
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("neural contact holdout row is invalid")
        for run_name in ("primary", "replay"):
            run = row.get(run_name)
            if not isinstance(run, dict) or not isinstance(run.get("trajectory"), dict):
                raise ValueError("neural contact holdout run is invalid")
            artifact = run["trajectory"]
            trajectory_path = source.parent / artifact["file"]
            if hash_bytes(trajectory_path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("neural contact holdout trajectory changed")
    return report


def _bound_unhashed_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object: {path}")
    return payload


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parents[1] / "growth" / "neural_contact_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("neural contact holdout output must be new and external")
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
    "fresh_neural_contact_holdouts",
    "run_neural_contact_holdout_exam",
    "validate_neural_contact_holdout_exam",
]
