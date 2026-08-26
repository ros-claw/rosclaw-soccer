"""Failure-driven calibration for an Athlete/Capture/Get-up router."""

from __future__ import annotations

import json
import math
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.recovery_contact_enrichment import (
    validate_recovery_contact_enrichment_report,
)

_DEFAULT_PELVIS_HEIGHT_GRID_M = tuple(
    round(0.680 + 0.005 * grid_index, 3) for grid_index in range(15)
)
_DEFAULT_ROOT_ANGULAR_SPEED_GRID_RAD_S = tuple(
    round(0.5 + 0.1 * grid_index, 1) for grid_index in range(21)
)


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _load_aggregate(path: Path) -> dict[str, Any]:
    payload = json.loads(path.expanduser().resolve().read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("recovery router aggregate is invalid")
    declared = payload.pop("report_hash", None)
    if (
        payload.get("schema_version") != "rosclaw_soccer.recovery_moe_reachability_aggregate.v1"
        or declared != hash_json(payload)
        or payload.get("promotion_eligible") is not False
        or payload.get("activation_ceiling") != "SIM_ONLY"
    ):
        raise ValueError("recovery router aggregate is invalid")
    payload["report_hash"] = declared
    return payload


def calibrate_recovery_capture_router(
    *,
    baseline_aggregate_path: Path,
    all_capture_oracle_aggregate_path: Path,
    contact_enrichment_path: Path,
    output_path: Path,
    pelvis_height_grid_m: Sequence[float] = _DEFAULT_PELVIS_HEIGHT_GRID_M,
    root_angular_speed_grid_rad_s: Sequence[float] = (_DEFAULT_ROOT_ANGULAR_SPEED_GRID_RAD_S),
) -> dict[str, Any]:
    """Choose the smallest quantized risk envelope covering every failure.

    The baseline supplies observed Athlete failures.  The all-Capture run is
    a parent-free counterfactual oracle for exactly the same states.  Contact
    enrichment supplies causal entry features.  Quantized grids deliberately
    expand the boundary beyond exact sample extrema instead of memorizing a
    floating-point state value.
    """

    baseline_path = baseline_aggregate_path.expanduser().resolve()
    oracle_path = all_capture_oracle_aggregate_path.expanduser().resolve()
    contact_path = contact_enrichment_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if target.exists() or len(set(pelvis_height_grid_m)) != len(pelvis_height_grid_m):
        raise ValueError("recovery capture-router calibration paths or grid are invalid")
    heights = tuple(float(value) for value in pelvis_height_grid_m)
    angular_speeds = tuple(float(value) for value in root_angular_speed_grid_rad_s)
    if (
        not heights
        or not angular_speeds
        or any(not math.isfinite(value) or not 0.65 <= value <= 0.80 for value in heights)
        or any(not math.isfinite(value) or not 0.25 <= value <= 4.0 for value in angular_speeds)
        or list(heights) != sorted(heights)
        or list(angular_speeds) != sorted(set(angular_speeds))
    ):
        raise ValueError("recovery capture-router calibration grid is invalid")

    baseline = _load_aggregate(baseline_path)
    oracle = _load_aggregate(oracle_path)
    contacts = validate_recovery_contact_enrichment_report(contact_path)
    invariant = (
        "failure_state_manifest_hash",
        "state_count",
        "physics_devices",
        "random_seeds",
    )
    if (
        any(baseline.get(name) != oracle.get(name) for name in invariant)
        or contacts.get("failure_state_manifest_hash")
        != baseline.get("failure_state_manifest_hash")
        or contacts.get("state_count") != baseline.get("state_count")
    ):
        raise ValueError("recovery capture-router inputs are not state/seed/device paired")
    baseline_contract = dict(baseline.get("expert_contract", {}))
    oracle_contract = dict(oracle.get("expert_contract", {}))
    for contract in (baseline_contract, oracle_contract):
        contract.pop("routing", None)
        contract.pop("capture_router_config", None)
    if (
        baseline_contract != oracle_contract
        or baseline_contract.get("getup_reference_phase_alignment") is not True
        or oracle.get("overall_final_stable_recovery_rate") != 1.0
    ):
        raise ValueError("recovery capture oracle is not a paired successful expert")

    baseline_rows = baseline.get("state_results")
    oracle_rows = oracle.get("state_results")
    contact_rows = contacts.get("state_rows")
    if not all(isinstance(rows, list) for rows in (baseline_rows, oracle_rows, contact_rows)):
        raise ValueError("recovery capture-router inputs lack per-state evidence")
    assert isinstance(baseline_rows, list)
    assert isinstance(oracle_rows, list)
    assert isinstance(contact_rows, list)
    states: list[dict[str, Any]] = []
    for baseline_row, oracle_row, contact_row in zip(
        baseline_rows, oracle_rows, contact_rows, strict=True
    ):
        identity = baseline_row.get("state_identity")
        if (
            baseline_row.get("failure_state_index") != oracle_row.get("failure_state_index")
            or baseline_row.get("failure_state_index") != contact_row.get("state_index")
            or identity != oracle_row.get("state_identity")
            or identity != contact_row.get("state_identity")
        ):
            raise ValueError("recovery capture-router state identity changed")
        states.append(
            {
                "state_index": int(baseline_row["failure_state_index"]),
                "state_identity": identity,
                "baseline_route": baseline_row["route"],
                "baseline_success": baseline_row["final_stable_recovery"],
                "oracle_success": oracle_row["final_stable_recovery"],
                "pelvis_height_m": float(contact_row["pelvis_height_m"]),
                "root_angular_speed_rad_s": float(contact_row["root_angular_speed_rad_s"]),
                "grounded_foot_sides": contact_row["grounded_foot_sides"],
                "support_aabb_signed_margin_m": contact_row["support_aabb"].get(
                    "com_aabb_signed_margin_m"
                ),
            }
        )
    athlete_failures = {
        row["state_index"]
        for row in states
        if row["baseline_route"] == "ATHLETE" and row["baseline_success"] is not True
    }
    if not athlete_failures:
        raise ValueError("recovery capture-router calibration has no Athlete failures")

    candidates: list[dict[str, Any]] = []
    for maximum_height in heights:
        for minimum_angular_speed in angular_speeds:
            selected_states = {
                row["state_index"]
                for row in states
                if row["baseline_route"] == "ATHLETE"
                and row["pelvis_height_m"] <= maximum_height
                and row["root_angular_speed_rad_s"] >= minimum_angular_speed
            }
            oracle_passed = all(
                row["oracle_success"] is True
                for row in states
                if row["state_index"] in selected_states
            )
            if athlete_failures.issubset(selected_states) and oracle_passed:
                candidates.append(
                    {
                        "maximum_pelvis_height_m": maximum_height,
                        "minimum_root_angular_speed_rad_s": minimum_angular_speed,
                        "capture_state_indices": sorted(selected_states),
                        "capture_state_count": len(selected_states),
                        "athlete_failure_coverage": len(athlete_failures),
                    }
                )
    if not candidates:
        raise ValueError("no quantized capture-router boundary covers every failure")
    selected_candidate = min(
        candidates,
        key=lambda row: (
            row["capture_state_count"],
            -float(row["maximum_pelvis_height_m"]),
            float(row["minimum_root_angular_speed_rad_s"]),
        ),
    )
    capture_indices = set(selected_candidate["capture_state_indices"])
    composed_rows = []
    for row in states:
        if row["baseline_route"] == "ATHLETE" and row["state_index"] in capture_indices:
            route = "CAPTURE"
            success = row["oracle_success"]
        else:
            route = row["baseline_route"]
            success = row["baseline_success"]
        composed_rows.append(
            {
                "state_index": row["state_index"],
                "state_identity": row["state_identity"],
                "route": route,
                "counterfactual_success": success,
            }
        )
    expected_success_count = sum(row["counterfactual_success"] is True for row in composed_rows)
    route_counts = {
        route: sum(row["route"] == route for row in composed_rows)
        for route in ("ATHLETE", "CAPTURE", "GET_UP")
    }
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_capture_router_calibration.v1",
        "baseline_aggregate_file_hash": hash_bytes(baseline_path.read_bytes()),
        "baseline_aggregate_report_hash": baseline["report_hash"],
        "all_capture_oracle_file_hash": hash_bytes(oracle_path.read_bytes()),
        "all_capture_oracle_report_hash": oracle["report_hash"],
        "contact_enrichment_file_hash": hash_bytes(contact_path.read_bytes()),
        "contact_enrichment_report_hash": contacts["report_hash"],
        "failure_state_manifest_hash": baseline["failure_state_manifest_hash"],
        "physics_devices": baseline["physics_devices"],
        "random_seeds": baseline["random_seeds"],
        "calibration_features": ["pelvis_height_m", "root_angular_speed_rad_s"],
        "feature_contract": "ENTRY_PROPRIOCEPTION_ONLY",
        "pelvis_height_grid_m": list(heights),
        "root_angular_speed_grid_rad_s": list(angular_speeds),
        "athlete_failure_state_indices": sorted(athlete_failures),
        "selected_router": selected_candidate,
        "counterfactual_route_counts": route_counts,
        "counterfactual_success_count": expected_success_count,
        "counterfactual_success_rate": expected_success_count / len(states),
        "requires_closed_loop_confirmation": True,
        "composed_state_rows": composed_rows,
        "claim_boundary": "FAILURE_BANK_CALIBRATION_NOT_BLIND_ROUTER_GENERALIZATION",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


def confirm_recovery_capture_router_closed_loop(
    *,
    calibration_report_path: Path,
    mixed_aggregate_path: Path,
    output_path: Path,
) -> dict[str, Any]:
    """Confirm that the calibrated composed routes ran and passed in physics."""

    calibration_path = calibration_report_path.expanduser().resolve()
    aggregate_path = mixed_aggregate_path.expanduser().resolve()
    target = output_path.expanduser().resolve()
    if target.exists():
        raise ValueError("recovery capture-router confirmation refuses to overwrite evidence")
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if not isinstance(calibration, dict):
        raise ValueError("recovery capture-router calibration is invalid")
    declared = calibration.pop("report_hash", None)
    if (
        calibration.get("schema_version") != "rosclaw_soccer.recovery_capture_router_calibration.v1"
        or declared != hash_json(calibration)
        or calibration.get("requires_closed_loop_confirmation") is not True
    ):
        raise ValueError("recovery capture-router calibration is invalid")
    calibration["report_hash"] = declared
    actual = _load_aggregate(aggregate_path)
    if (
        calibration.get("failure_state_manifest_hash") != actual.get("failure_state_manifest_hash")
        or calibration.get("physics_devices") != actual.get("physics_devices")
        or calibration.get("random_seeds") != actual.get("random_seeds")
    ):
        raise ValueError("recovery capture-router confirmation is not paired")
    selected = calibration.get("selected_router")
    actual_config = actual.get("expert_contract", {}).get("capture_router_config")
    if (
        not isinstance(selected, dict)
        or not isinstance(actual_config, dict)
        or actual_config.get("maximum_pelvis_height_m") != selected.get("maximum_pelvis_height_m")
        or actual_config.get("minimum_root_angular_speed_rad_s")
        != selected.get("minimum_root_angular_speed_rad_s")
        or actual_config.get("feature_contract") != calibration.get("feature_contract")
    ):
        raise ValueError("recovery capture-router runtime config differs from calibration")
    expected_rows = calibration.get("composed_state_rows")
    actual_rows = actual.get("state_results")
    if not isinstance(expected_rows, list) or not isinstance(actual_rows, list):
        raise ValueError("recovery capture-router confirmation lacks per-state evidence")
    for expected, observed in zip(expected_rows, actual_rows, strict=True):
        if (
            expected.get("state_index") != observed.get("failure_state_index")
            or expected.get("state_identity") != observed.get("state_identity")
            or expected.get("route") != observed.get("route")
            or observed.get("final_stable_recovery") is not True
            or observed.get("handoff_completed") is not True
        ):
            raise ValueError("recovery capture-router closed-loop state differs from calibration")
    expected_route_counts = calibration.get("counterfactual_route_counts")
    actual_route_counts = {
        route: actual["route_metrics"][route]["state_count"]
        for route in ("ATHLETE", "CAPTURE", "GET_UP")
    }
    confirmed = bool(
        expected_route_counts == actual_route_counts
        and actual.get("overall_final_stable_recovery_rate") == 1.0
        and all(
            actual["route_metrics"][route]["final_stable_recovery_rate"] == 1.0
            for route in ("ATHLETE", "CAPTURE", "GET_UP")
        )
    )
    report: dict[str, Any] = {
        "schema_version": "rosclaw_soccer.recovery_capture_router_confirmation.v1",
        "calibration_file_hash": hash_bytes(calibration_path.read_bytes()),
        "calibration_report_hash": calibration["report_hash"],
        "mixed_aggregate_file_hash": hash_bytes(aggregate_path.read_bytes()),
        "mixed_aggregate_report_hash": actual["report_hash"],
        "failure_state_manifest_hash": actual["failure_state_manifest_hash"],
        "physics_devices": actual["physics_devices"],
        "random_seeds": actual["random_seeds"],
        "selected_router": selected,
        "closed_loop_route_counts": actual_route_counts,
        "closed_loop_route_metrics": actual["route_metrics"],
        "closed_loop_success_rate": actual["overall_final_stable_recovery_rate"],
        "decision": (
            "CAPTURE_ROUTER_CLOSED_LOOP_CONFIRMED"
            if confirmed
            else "CAPTURE_ROUTER_CLOSED_LOOP_NOT_CONFIRMED"
        ),
        "claim_boundary": "BOUND_FAILURE_BANK_REACHABILITY_NOT_BLIND_POST_DIVE_PROMOTION",
        "deployment_candidate": False,
        "promotion_eligible": False,
        "promotion_authority": "NONE",
        "activation_ceiling": "SIM_ONLY",
        "hardware_authorized": False,
        "hardware_command_sent": False,
    }
    report["report_hash"] = hash_json(report)
    _atomic_json(target, report)
    return report


__all__ = [
    "calibrate_recovery_capture_router",
    "confirm_recovery_capture_router_closed_loop",
]
