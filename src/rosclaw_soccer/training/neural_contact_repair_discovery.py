"""Failure-driven SIM teacher collection for rejected neural contact holdouts."""

from __future__ import annotations

import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.contact_handoff_actor import load_contact_handoff_actor
from rosclaw_soccer.growth.target_contact_plan_actor import TargetContactPlanAction
from rosclaw_soccer.providers.g1.asset_qualification import qualify_g1_assets
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.adaptive_target_teacher_discovery import (
    AdaptiveTargetTeacherProbe,
    _run_probe,
)
from rosclaw_soccer.training.causal_transition_growth import CausalTransitionGrowthConfig
from rosclaw_soccer.training.contact_handoff_discovery import _context_from_dict
from rosclaw_soccer.training.neural_contact_canary import _bound_json


def run_neural_contact_repair_discovery(
    *,
    asset_root: Path,
    source_s95_dir: Path,
    rejected_holdout_report_path: Path,
    teacher_discovery_report_path: Path,
    handoff_actor_path: Path,
    output_dir: Path,
    quality_config: CausalTransitionGrowthConfig | None = None,
    workers: int = 3,
) -> dict[str, Any]:
    if not 1 <= workers <= 3:
        raise ValueError("neural contact repair workers must be in [1, 3]")
    rejected_path = rejected_holdout_report_path.expanduser().resolve()
    rejected = _bound_json(rejected_path)
    teacher_path = teacher_discovery_report_path.expanduser().resolve()
    teacher = _bound_json(teacher_path)
    if (
        rejected.get("status") != "REJECTED_NEURAL_CONTACT_LOCAL_HOLDOUT"
        or rejected.get("partition") != "FRESH_S131_LOCAL_HOLDOUT"
        or teacher.get("status") != "REJECTED_ADAPTIVE_TARGET_TEACHER_DISCOVERY"
    ):
        raise ValueError("neural contact repair requires intact rejected evidence")
    for row in rejected["rows"]:
        for run_name in ("primary", "replay"):
            artifact = row[run_name]["trajectory"]
            path = rejected_path.parent / artifact["file"]
            if hash_bytes(path.read_bytes()) != artifact["file_hash"]:
                raise ValueError("neural contact rejected trajectory changed")
    source_rows = [row for row in teacher["rows"] if row["quality"]["chain_passed"]]
    failed_rows = [row for row in rejected["rows"] if not row["strict_right_foot_chain"]]
    if len(source_rows) != 1 or len(failed_rows) != 3:
        raise ValueError("neural contact repair expects one source and three failures")
    action_payload = dict(source_rows[0]["action"])
    action_payload["target_foot_velocity_xyz_mps"] = tuple(
        action_payload["target_foot_velocity_xyz_mps"]
    )
    action = TargetContactPlanAction(**action_payload)
    # The canary run payload intentionally carries only hashes, so recover the
    # predeclared contexts from the bound request rather than trusting results.
    rejected_request_path = rejected_path.parent / "request.json"
    if hash_bytes(rejected_request_path.read_bytes()) != rejected["request_hash"]:
        raise ValueError("neural contact rejected request changed")
    request_source = json.loads(rejected_request_path.read_text("utf-8"))
    context_by_hash = {
        str(hash_json(value)): _context_from_dict(value) for value in request_source["contexts"]
    }
    action_variants = (
        action,
        replace(action, contact_policy_frame=248),
        replace(action, contact_policy_frame=256),
        replace(action, stance_offset_x_m=-0.12),
        replace(action, stance_offset_x_m=-0.04),
        replace(action, stance_offset_y_m=-0.10),
        replace(action, stance_offset_y_m=-0.02),
        replace(action, maximum_arrival_advance_frames=12),
    )
    probes = tuple(
        AdaptiveTargetTeacherProbe(context_by_hash[row["context_hash"]], candidate)
        for row in failed_rows
        for candidate in action_variants
    )
    handoff = load_contact_handoff_actor(handoff_actor_path)
    qualification = qualify_g1_assets(asset_root)
    qualification.require_eligible()
    if handoff.body_hash != qualification.body_hash:
        raise ValueError("neural contact repair Body lineage changed")
    quality = quality_config or CausalTransitionGrowthConfig()
    output = _new_external_output(output_dir)
    request: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_repair_request.v2",
        "partition": "CONSUMED_S131_FAILURES_TEACHER_REPAIR",
        "rejected_holdout_report_hash": rejected["report_hash"],
        "rejected_holdout_file_hash": hash_bytes(rejected_path.read_bytes()),
        "teacher_discovery_report_hash": teacher["report_hash"],
        "probe_hashes": [probe.probe_hash for probe in probes],
        "contexts": [asdict(probe.context) for probe in probes],
        "action": asdict(action),
        "action_variants": [asdict(candidate) for candidate in action_variants],
        "contact_handoff_actor_hash": handoff.actor_hash,
        "body_hash": qualification.body_hash,
        "quality_config": asdict(quality),
        "implementation_hash": _implementation_hash(),
        "teacher_role": "SIM_ONLY_FAILURE_CORRECTION_GENERATOR",
        "pixels_used_for_scoring": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    _write_json(output / "request.json", request)
    jobs = tuple(
        (
            asset_root.expanduser().resolve(),
            source_s95_dir.expanduser().resolve(),
            handoff_actor_path.expanduser().resolve(),
            output,
            index,
            probe,
            quality,
        )
        for index, probe in enumerate(probes)
    )
    if workers == 1:
        rows = [_run_probe(job) for job in jobs]
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            rows = list(executor.map(_run_probe, jobs))
    for row in rows:
        path = output / row["trajectory"]["file"]
        with np.load(path, allow_pickle=False) as trajectory:
            contacts = np.asarray(trajectory["shooter_ball_contact_foot"], dtype=np.int64)
        nonzero = contacts[contacts != 0]
        row["first_shooter_contact_foot"] = int(nonzero[0]) if len(nonzero) else None
        row["strict_right_foot_chain"] = bool(
            row["quality"]["chain_passed"] and len(nonzero) and nonzero[0] == 1
        )
    successful_contexts = {row["context_hash"] for row in rows if row["strict_right_foot_chain"]}
    gates = {
        "all_teacher_active": all(row["teacher_active"] for row in rows),
        "minimum_safe_teacher_support": sum(int(row["quality"]["safe"]) for row in rows)
        >= len(rows) - 6,
        "minimum_strict_successes": sum(int(row["strict_right_foot_chain"]) for row in rows) >= 3,
        "all_failure_contexts_recovered": len(successful_contexts) == len(failed_rows),
        "all_trajectories_bound": all(
            hash_bytes((output / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw.growth.neural_contact_repair_discovery.v2",
        "status": (
            "PASS_NEURAL_CONTACT_REPAIR_DATA"
            if all(gates.values())
            else "REJECTED_NEURAL_CONTACT_REPAIR_DATA"
        ),
        "promotion_eligible": False,
        "partition": "CONSUMED_S131_FAILURES_TEACHER_REPAIR",
        "request_hash": hash_bytes((output / "request.json").read_bytes()),
        "rejected_holdout_report_hash": rejected["report_hash"],
        "teacher_discovery_report_hash": teacher["report_hash"],
        "body_hash": qualification.body_hash,
        "metrics": {
            "probe_count": len(rows),
            "unique_failure_context_count": len(failed_rows),
            "safe_teacher_response_count": sum(int(row["quality"]["safe"]) for row in rows),
            "strict_right_foot_chain_count": sum(
                int(row["strict_right_foot_chain"]) for row in rows
            ),
            "goal_count": sum(int(row["result"]["goal_crossed"]) for row in rows),
            "save_count": sum(int(row["result"]["goalkeeper_save_observed"]) for row in rows),
            "recovered_context_count": len(successful_contexts),
        },
        "gates": gates,
        "rows": rows,
        "implementation_hash": _implementation_hash(),
        "activation_ceiling": "SIM_ONLY",
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _write_json(output / "repair-report.json", report)
    return report


def _implementation_hash() -> str:
    digest = hashlib.sha256()
    for path in (
        Path(__file__),
        Path(__file__).parent / "adaptive_target_teacher_discovery.py",
    ):
        digest.update(path.read_bytes())
    return "sha256:" + digest.hexdigest()


def _new_external_output(path: Path) -> Path:
    output = path.expanduser().resolve()
    checkout = Path(__file__).parents[3].resolve()
    if output.exists() or output == checkout or checkout in output.parents:
        raise ValueError("neural contact repair output must be new and external")
    output.mkdir(parents=True)
    return output


def _write_json(path: Path, value: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, path)


__all__ = ["run_neural_contact_repair_discovery"]
