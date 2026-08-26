"""Strict aggregation of multi-device recovery baseline physics evidence."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json


def aggregate_recovery_physics_reports(
    *,
    report_paths: Sequence[Path],
    output_path: Path,
    expected_devices: tuple[str, ...] = ("cuda:0", "cuda:1", "cuda:2", "cuda:3"),
) -> dict[str, Any]:
    """Aggregate component reports while preserving their narrow claim boundary."""

    if not report_paths or len(report_paths) != len(expected_devices):
        raise ValueError("recovery baseline evidence requires one report per expected device")
    reports = []
    for raw_path in report_paths:
        path = raw_path.expanduser().resolve()
        if not path.is_file() or path.stat().st_size > 2 * 1024 * 1024:
            raise ValueError("recovery baseline report is missing or too large")
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("recovery baseline report must be a JSON object")
        reports.append(payload)
    observed_devices = tuple(str(report.get("physics_device")) for report in reports)
    if set(observed_devices) != set(expected_devices) or len(set(observed_devices)) != len(reports):
        raise ValueError("recovery baseline reports do not bind every expected device")
    bindings = {
        (
            report.get("contract_hash"),
            report.get("checkpoint_hash"),
            report.get("source_hash"),
            report.get("body_hash"),
            report.get("physics_scene_hash"),
            report.get("handoff_config_hash"),
        )
        for report in reports
    }
    if len(bindings) != 1:
        raise ValueError("recovery baseline reports use different artifact contracts")
    for report in reports:
        if (
            report.get("schema_version") != "rosclaw_soccer.mjlab_getup_physics_probe.v3"
            or report.get("activation_ceiling") != "SIM_ONLY"
            or report.get("hardware_command_sent") is not False
            or report.get("environment_count", 0) <= 0
            or report.get("final_stable_recovery_count", -1) < 0
            or report.get("final_stable_recovery_count", 0) > report.get("environment_count", 0)
            or not math.isfinite(float(report.get("initial_perturbation_scale", math.nan)))
        ):
            raise ValueError("recovery baseline report violates its evidence boundary")
    environment_count = sum(int(report["environment_count"]) for report in reports)
    successful = sum(int(report["final_stable_recovery_count"]) for report in reports)
    stable_seconds = [
        float(value) for report in reports for value in report["final_continuous_stable_sec"]
    ]
    binding = next(iter(bindings))
    aggregate: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_baseline_aggregate.v1",
        "baseline_id": "R0_MJLAB_NEURAL_GETUP_WITH_WARM_HANDOFF",
        "contract_hash": binding[0],
        "checkpoint_hash": binding[1],
        "source_hash": binding[2],
        "body_hash": binding[3],
        "physics_scene_hash": binding[4],
        "handoff_config_hash": binding[5],
        "devices": sorted(observed_devices),
        "report_hashes": sorted(str(report["report_hash"]) for report in reports),
        "environment_count": environment_count,
        "final_stable_recovery_count": successful,
        "final_stable_recovery_rate": successful / environment_count,
        "minimum_final_continuous_stable_sec": min(stable_seconds),
        "maximum_final_continuous_stable_sec": max(stable_seconds),
        "initial_perturbation_scales": sorted(
            {float(report["initial_perturbation_scale"]) for report in reports}
        ),
        "claim_boundary": "COMPONENT_LOCAL_PERTURBATION_NOT_TRUE_POST_SAVE_RECOVERY",
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
    }
    aggregate["report_hash"] = hash_json(aggregate)
    destination = output_path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    temporary.write_text(
        json.dumps(aggregate, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    temporary.replace(destination)
    return aggregate


__all__ = ["aggregate_recovery_physics_reports"]
