"""Precision-aware repair of consumed prepared-finisher holdout failures."""

from __future__ import annotations

import hashlib
import json
import math
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.neural_contact_actor import load_g1_neural_contact_actor
from rosclaw_soccer.growth.runtime_contact_target_actor import (
    RuntimeContactTargetAction,
    load_runtime_contact_target_actor,
)
from rosclaw_soccer.growth.runtime_finish_plan_actor import (
    RuntimeFinishPlanAction,
    load_runtime_finish_plan_actor,
)
from rosclaw_soccer.growth.runtime_receive_actor import RuntimeReceiveAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.causal_transition_growth import (
    CausalTransitionGrowthConfig,
    _load_lead_policy,
)
from rosclaw_soccer.training.neural_contact_canary import _bound_json
from rosclaw_soccer.training.prepared_finish_plan_repair import _run_probe
from rosclaw_soccer.training.runtime_finish_plan_exam import (
    validate_runtime_finish_plan_exam,
)


@dataclass(frozen=True)
class PreparedFinishPrecisionRepairConfig:
    """Development-only gates; this stage can never promote an actor."""

    precision_radius_m: float = 0.10
    expected_failed_contexts: int = 3
    minimum_actions: int = 8
    maximum_actions: int = 16
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.precision_radius_m)
            or not 0.05 <= self.precision_radius_m <= 0.20
            or not 1 <= self.expected_failed_contexts <= 6
            or not 4 <= self.minimum_actions <= self.maximum_actions <= 24
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("prepared finish precision repair config is invalid")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


def default_prepared_finish_precision_actions() -> tuple[RuntimeFinishPlanAction, ...]:
    """S168-v2 screen derived from v1 direction and contact diagnostics."""

    return tuple(
        RuntimeFinishPlanAction(
            receive=RuntimeReceiveAction(
                maximum_arrival_advance_frames=18,
                arrival_alignment_tolerance_sec=0.02,
                stance_offset_x_m=stance_x,
                stance_offset_y_m=stance_y,
                contact_policy_frame=frame,
                foot_yaw_offset_rad=-0.12,
                foot_pitch_offset_rad=0.01,
            ),
            target=RuntimeContactTargetAction((9.0, target_y, -1.0)),
        )
        for stance_x, stance_y, frame in (
            (0.12, 0.12, 248),
            (0.12, 0.12, 250),
            (0.12, 0.12, 252),
            (0.12, 0.06, 248),
            (0.12, 0.06, 250),
        )
        for target_y in (4.0, 5.0, 6.0)
    )


def run_prepared_finish_precision_repair(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    consumed_exam_path: Path,
    finish_plan_actor_path: Path,
    finish_plan_training_report_path: Path,
    base_target_actor_path: Path,
    neural_actor_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    actions: tuple[RuntimeFinishPlanAction, ...] | None = None,
    config: PreparedFinishPrecisionRepairConfig | None = None,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 1,
) -> dict[str, Any]:
    """Search consumed failures without reusing them as a sealed exam."""

    if not 1 <= workers <= 4:
        raise ValueError("prepared finish precision repair workers must be in [1, 4]")
    active = config or PreparedFinishPrecisionRepairConfig()
    active_actions = actions or default_prepared_finish_precision_actions()
    if not active.minimum_actions <= len(active_actions) <= active.maximum_actions or len(
        {action.action_hash for action in active_actions}
    ) != len(active_actions):
        raise ValueError("prepared finish precision actions are incomplete or duplicated")

    exam_path = consumed_exam_path.expanduser().resolve()
    exam = validate_runtime_finish_plan_exam(exam_path)
    failed_rows = [
        row for row in exam["rows"] if not row["candidate"]["quality"]["strict_chain_passed"]
    ]
    actor_path = finish_plan_actor_path.expanduser().resolve()
    actor = load_runtime_finish_plan_actor(actor_path)
    training_path = finish_plan_training_report_path.expanduser().resolve()
    training = _bound_json(training_path)
    base_path = base_target_actor_path.expanduser().resolve()
    base = load_runtime_contact_target_actor(base_path)
    neural_path = neural_actor_path.expanduser().resolve()
    neural = load_g1_neural_contact_actor(neural_path)
    handoff_path = handoff_actor_path.expanduser().resolve()
    handoff = load_contact_handoff_actor(handoff_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    _, lead_source = _load_lead_policy(source_s95_dir)
    if (
        exam.get("status") != "PASS_RUNTIME_FINISH_PLAN_FRESH_HOLDOUT"
        or exam.get("sealed") is not True
        or exam.get("promotion_eligible") is not True
        or len(failed_rows) != active.expected_failed_contexts
        or exam.get("finish_plan_actor_hash") != actor.actor_hash
        or training.get("status") != "PASS_RUNTIME_FINISH_PLAN_TRAINING"
        or training.get("actor_hash") != actor.actor_hash
        or training.get("actor_file_hash") != hash_bytes(actor_path.read_bytes())
        or actor.neural_contact_actor_hash != neural.actor_hash
        or actor.contact_handoff_actor_hash != handoff.actor_hash
        or actor.body_hash != qualification.body_hash
        or actor.kick_prior_hash != qualification.kick_prior_hash
        or actor.roster_hash != base.roster_hash
        or actor.finisher_self_model_hash != base.finisher_self_model_hash
        or base.body_hash != qualification.body_hash
        or neural.body_hash != qualification.body_hash
        or any(
            not neural.target_supported(action.target.target_foot_velocity_xyz_mps)
            for action in active_actions
        )
    ):
        raise ValueError("prepared finish precision repair lineage changed")

    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.prepared_finish_precision_repair_request.v1",
        "partition": "CONSUMED_S167_FAILURES",
        "source_exam_hash": exam["report_hash"],
        "source_exam_file_hash": hash_bytes(exam_path.read_bytes()),
        "failed_context_hashes": [row["context_hash"] for row in failed_rows],
        "actions": [asdict(action) for action in active_actions],
        "action_hashes": [action.action_hash for action in active_actions],
        "finish_plan_actor_hash": actor.actor_hash,
        "finish_plan_actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "finish_plan_training_report_hash": training["report_hash"],
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "handoff_actor_hash": handoff.actor_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": actor.roster_hash,
        "finisher_self_model_hash": actor.finisher_self_model_hash,
        "config": asdict(active),
        "config_hash": active.config_hash,
        "quality_config": asdict(quality),
        "quality_config_hash": quality.config_hash,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            base_path,
            neural_path,
            output,
            index,
            row,
            action,
            handoff.selected_offset_frames,
            quality,
        )
        for index, (row, action) in enumerate(
            (row, action) for row in failed_rows for action in active_actions
        )
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    selected = [_select(rows, str(source["context_hash"]), active) for source in failed_rows]
    recovered = {row["context_hash"] for row in rows if row["quality"]["strict_chain_passed"]}
    precise = [row for row in rows if _precision_passed(row, active)]
    failure_route_counts: dict[str, int] = {}
    for row in rows:
        route = str(row["failure_route"])
        failure_route_counts[route] = failure_route_counts.get(route, 0) + 1
    gates = {
        "all_consumed_failures_recovered": len(recovered) == len(failed_rows),
        "selected_safe_all": all(row["quality"]["safe"] for row in selected),
        "selected_strict_all": all(row["quality"]["strict_chain_passed"] for row in selected),
        "at_least_one_precise_goal": bool(precise),
        "neural_actor_realized_all": all(row["neural_actor_active"] for row in rows),
        "teacher_and_scripted_contact_absent": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.prepared_finish_precision_repair.v1",
        "status": (
            "PASS_PREPARED_FINISH_PRECISION_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_PREPARED_FINISH_PRECISION_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "partition": request["partition"],
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "source_exam_hash": exam["report_hash"],
        "finish_plan_actor_hash": actor.actor_hash,
        "base_target_actor_hash": base.actor_hash,
        "neural_actor_hash": neural.actor_hash,
        "handoff_actor_hash": handoff.actor_hash,
        "source_s95_evidence_hash": lead_source["evidence_hash"],
        "body_hash": qualification.body_hash,
        "kick_prior_hash": qualification.kick_prior_hash,
        "roster_hash": actor.roster_hash,
        "finisher_self_model_hash": actor.finisher_self_model_hash,
        "metrics": {
            "probe_count": len(rows),
            "context_count": len(failed_rows),
            "action_count": len(active_actions),
            "safe_count": sum(bool(row["quality"]["safe"]) for row in rows),
            "strict_success_count": sum(
                bool(row["quality"]["strict_chain_passed"]) for row in rows
            ),
            "recovered_context_count": len(recovered),
            "precise_goal_count": len(precise),
            "best_goal_target_error_m": min(
                (float(row["result"]["target_error_m"]) for row in precise),
                default=None,
            ),
            "failure_route_counts": failure_route_counts,
        },
        "selected": selected,
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "physics_authority": "CPU_MUJOCO",
        "activation_ceiling": "SIM_ONLY",
        "pixels_used_for_scoring": False,
        "teacher_enabled": False,
        "scripted_contact_torque_enabled": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "repair-report.json", report)
    return report


def _precision_passed(row: dict[str, Any], config: PreparedFinishPrecisionRepairConfig) -> bool:
    error = row["result"].get("target_error_m")
    return bool(
        row["quality"]["strict_chain_passed"]
        and row["result"]["goal_crossed"]
        and isinstance(error, int | float)
        and math.isfinite(float(error))
        and float(error) <= config.precision_radius_m
    )


def _select(
    rows: list[dict[str, Any]],
    context_hash: str,
    config: PreparedFinishPrecisionRepairConfig,
) -> dict[str, Any]:
    return min(
        (row for row in rows if row["context_hash"] == context_hash),
        key=lambda row: (
            not row["quality"]["strict_chain_passed"],
            not row["quality"]["safe"],
            not _precision_passed(row, config),
            not row["quality"]["intended_foot_contact"],
            not row["quality"]["clear_outcome"],
            _target_error(row),
            -float(row["result"]["shot_peak_ball_speed_mps"]),
            row["action_hash"],
        ),
    )


def _target_error(row: dict[str, Any]) -> float:
    value = row["result"].get("target_error_m")
    return float(value) if isinstance(value, int | float) and math.isfinite(float(value)) else 1e6


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parent / "prepared_finish_plan_repair.py",
        Path(__file__).parents[1] / "growth" / "runtime_finish_plan_actor.py",
        Path(__file__).parents[1] / "skills" / "team" / "shared_world.py",
    ):
        digest.update(path.name.encode())
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("prepared finish precision repair output must be new and external")
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
    "PreparedFinishPrecisionRepairConfig",
    "default_prepared_finish_precision_actions",
    "run_prepared_finish_precision_repair",
]
