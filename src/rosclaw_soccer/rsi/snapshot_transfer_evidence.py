"""Prospective SIM_ONLY check that a closed-loop snapshot predicts full Isaac contact."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.snapshot_replay_evidence import audit_snapshot_replay
from rosclaw_soccer.rsi.temporal_first_touch_policy import knee_extension_probe_weights
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_temporal_first_touch_execution
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _first(force: np.ndarray) -> tuple[int | None, list[int]]:
    active = np.flatnonzero(np.max(force, axis=1) > 1.0)
    if len(active) == 0:
        return None, []
    frame = int(active[0])
    return frame, np.flatnonzero(force[frame] > 1.0).tolist()


def _clean(force: np.ndarray) -> bool:
    bodies = np.flatnonzero(np.max(force, axis=0) > 1.0)
    return bool(len(bodies) and set(bodies.tolist()) <= {0, 1})


def audit_snapshot_transfer(
    *,
    protocol_path: Path,
    snapshot_bank: Path,
    baseline: Path,
    intervention: Path,
    full_candidate: Path,
    full_parent: Path,
    candidate_manifest: Path,
) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("schema") != "rsi_first_touch_closed_loop_snapshot_prospective_protocol_v1"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("learning_authorized") is not False
        or protocol.get("promotion_authorized") is not False
    ):
        raise ValueError("invalid prospective transfer protocol")
    bank = json.loads((snapshot_bank / "manifest.json").read_text(encoding="utf-8"))
    base_audit = audit_snapshot_replay(baseline, snapshot_bank=snapshot_bank)
    intervention_audit = audit_snapshot_replay(intervention, snapshot_bank=snapshot_bank)
    full_audit = audit_temporal_first_touch_execution(
        full_candidate, parent_folder=full_parent, candidate_path=candidate_manifest
    )
    base_report = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    probe_report = json.loads((intervention / "report.json").read_text(encoding="utf-8"))
    full_report = json.loads((full_candidate / "report.json").read_text(encoding="utf-8"))
    manifest = json.loads(candidate_manifest.read_text(encoding="utf-8"))
    selection = protocol["snapshot_selection"]
    start = selection["start_index"]
    count = selection["sample_count"]
    if (
        protocol["snapshot_bank_manifest_hash"] != bank["manifest_hash"]
        or base_report["start_index"] != start
        or probe_report["start_index"] != start
        or base_report["sample_count"] != count
        or probe_report["sample_count"] != count
        or not base_audit["closed_loop_sonic"]
        or not intervention_audit["closed_loop_sonic"]
        or base_audit["intervention_action_audited"]
        or not intervention_audit["intervention_action_audited"]
        or bank.get("fixed_start_frame") != selection["fixed_start_frame"]
        or bank["window_frames"] != selection["window_frames"]
        or full_report.get("training_course_seed") != protocol["source_training_seed"]
        or protocol["full_run_parent"] != full_parent.name
        or full_audit.get("execution_report_hash") != full_report["report_hash"]
        or not np.array_equal(
            np.asarray(manifest["weights_per_course"]), knee_extension_probe_weights()
        )
    ):
        raise ValueError("snapshot/full-run transfer lineage or intervention changed")
    rows = []
    with (
        np.load(baseline / "replay.npz", allow_pickle=False) as base_trace,
        np.load(intervention / "replay.npz", allow_pickle=False) as probe_trace,
        np.load(full_candidate / "trace.npz", allow_pickle=False) as full_trace,
    ):
        full_force = full_trace["ball_body_contact_force_peak_n"]
        base_force = base_trace["observed_ball_body_contact_force_peak_n"]
        probe_force = probe_trace["observed_ball_body_contact_force_peak_n"]
        if (
            base_force.shape != (bank["window_frames"], count, 6)
            or probe_force.shape != base_force.shape
            or full_force.shape != (full_report["frames"], 16, 6)
        ):
            raise ValueError("transfer force trace shape changed")
        for lane, source in enumerate(bank["snapshots"][start : start + count]):
            source_lane = source["lane"]
            if full_report["environments"][source_lane]["course"] != source["course"]:
                raise ValueError("snapshot and full-run courses differ")
            base_first, base_body = _first(base_force[:, lane])
            probe_first, probe_body = _first(probe_force[:, lane])
            full_first, full_body = _first(full_force[:, source_lane])
            if base_first is None or probe_first is None or full_first is None:
                raise ValueError("prospective transfer requires actual physical contact")
            full_offset = full_first - source["start_frame"]
            rows.append(
                {
                    "source_lane": source_lane,
                    "baseline_first_contact_offset": base_first,
                    "intervention_first_contact_offset": probe_first,
                    "full_run_first_contact_offset": full_offset,
                    "first_contact_frame_absolute_difference": abs(probe_first - full_offset),
                    "baseline_first_contact_body_indices": base_body,
                    "intervention_first_contact_body_indices": probe_body,
                    "full_run_first_contact_body_indices": full_body,
                    "first_contact_body_class_equal": probe_body == full_body,
                    "intervention_clean_foot_only": _clean(probe_force[:, lane]),
                    "full_run_clean_foot_only": _clean(full_force[:, source_lane]),
                }
            )
        minimum_root_height = float(
            min(
                np.min(base_trace["observed_root_pose_local_xyzw_m"][:, :, 2]),
                np.min(probe_trace["observed_root_pose_local_xyzw_m"][:, :, 2]),
            )
        )
    checks = protocol["acceptance"]
    equal_clean = sum(
        row["intervention_clean_foot_only"] == row["full_run_clean_foot_only"] for row in rows
    )
    false_clean = sum(
        row["intervention_clean_foot_only"] and not row["full_run_clean_foot_only"] for row in rows
    )
    first_class_equal = sum(row["first_contact_body_class_equal"] for row in rows)
    accepted = bool(
        max(base_audit["max_initial_state_error"], intervention_audit["max_initial_state_error"])
        <= checks["initial_state_error_max"]
        and max(
            base_report["warmup_max_target_error_rad"], probe_report["warmup_max_target_error_rad"]
        )
        <= checks["warmup_target_error_rad_max"]
        and max(
            base_report["max_parent_target_error_at_snapshot_rad"],
            probe_report["max_parent_target_error_at_snapshot_rad"],
        )
        <= checks["first_snapshot_target_error_rad_max"]
        and max(
            base_audit["max_precontact_ball_position_error_m"],
            intervention_audit["max_precontact_ball_position_error_m"],
        )
        <= checks["precontact_ball_error_m_max"]
        and max(
            base_audit["max_precontact_root_position_error_m"],
            intervention_audit["max_precontact_root_position_error_m"],
        )
        <= checks["precontact_root_error_m_max"]
        and all(
            row["first_contact_frame_absolute_difference"]
            <= checks["first_contact_frame_absolute_difference_max"]
            for row in rows
        )
        and first_class_equal >= checks["first_contact_body_class_equal_min_of_7"]
        and equal_clean >= checks["clean_foot_outcome_equal_min_of_7"]
        and false_clean <= checks["false_clean_foot_positive_max_of_7"]
        and minimum_root_height >= checks["minimum_robot_root_height_m"]
    )
    result = {
        "schema": "rsi_first_touch_closed_loop_snapshot_transfer_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "snapshot_bank_manifest_hash": bank["manifest_hash"],
        "baseline_audit_hash": base_audit["report_hash"],
        "intervention_audit_hash": intervention_audit["report_hash"],
        "full_candidate_audit_hash": full_audit["report_hash"],
        "sample_count": count,
        "first_contact_body_class_equal_count": first_class_equal,
        "clean_foot_outcome_equal_count": equal_clean,
        "false_clean_foot_positive_count": false_clean,
        "minimum_root_height_m": minimum_root_height,
        "rows": rows,
        "first_contact_candidate_screening_qualified": accepted,
        "learning_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "protocol",
        "snapshot_bank",
        "baseline",
        "intervention",
        "full_candidate",
        "full_parent",
        "candidate_manifest",
        "output",
    ):
        parser.add_argument(f"--{name.replace('_', '-')}", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable transfer audit already exists")
    result = audit_snapshot_transfer(
        protocol_path=args.protocol,
        snapshot_bank=args.snapshot_bank,
        baseline=args.baseline,
        intervention=args.intervention,
        full_candidate=args.full_candidate,
        full_parent=args.full_parent,
        candidate_manifest=args.candidate_manifest,
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
