"""Audit whether a batched SIM_ONLY intervention perturbs untreated lanes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def first_untreated_precontact_drift(
    baseline_root: np.ndarray[Any, Any],
    candidate_root: np.ndarray[Any, Any],
    baseline_target: np.ndarray[Any, Any],
    candidate_target: np.ndarray[Any, Any],
    untreated_mask: np.ndarray[Any, Any],
    first_contact_frames: np.ndarray[Any, Any],
    *,
    tolerance: float = 1e-5,
) -> dict[int, int]:
    """Find the first divergent frame before contact in each untreated lane."""
    root_a = np.asarray(baseline_root, dtype=np.float64)
    root_b = np.asarray(candidate_root, dtype=np.float64)
    target_a = np.asarray(baseline_target, dtype=np.float64)
    target_b = np.asarray(candidate_target, dtype=np.float64)
    mask = np.asarray(untreated_mask)
    contacts = np.asarray(first_contact_frames)
    if (
        root_a.ndim != 3
        or root_a.shape != root_b.shape
        or root_a.shape[2] != 7
        or target_a.shape != target_b.shape
        or target_a.shape != (root_a.shape[0], root_a.shape[1], 29)
        or mask.shape != (root_a.shape[1],)
        or mask.dtype != np.dtype("bool")
        or contacts.shape != (root_a.shape[1],)
        or not np.issubdtype(contacts.dtype, np.integer)
        or not np.isfinite(root_a).all()
        or not np.isfinite(root_b).all()
        or not np.isfinite(target_a).all()
        or not np.isfinite(target_b).all()
        or type(tolerance) is not float
        or not np.isfinite(tolerance)
        or tolerance <= 0
        or np.any((contacts < 0) | (contacts > root_a.shape[0]))
    ):
        raise ValueError("finite paired same-course trajectories required")
    drift: dict[int, int] = {}
    for lane in np.flatnonzero(mask):
        limit = int(contacts[lane])
        root_delta = np.max(np.abs(root_a[:limit, lane] - root_b[:limit, lane]), axis=1)
        target_delta = np.max(np.abs(target_a[:limit, lane] - target_b[:limit, lane]), axis=1)
        changed = np.flatnonzero((root_delta > tolerance) | (target_delta > tolerance))
        if len(changed):
            drift[int(lane)] = int(changed[0])
    return drift


def audit_paired_lane_isolation(baseline_folder: Path, candidate_folder: Path) -> dict[str, Any]:
    """Reaudit physical evidence, then reject lane-level causal attribution on spillover."""
    baseline_audit = audit_vector_first_touch(baseline_folder)
    candidate_audit = audit_vector_first_touch(candidate_folder)
    baseline = json.loads((baseline_folder / "report.json").read_text(encoding="utf-8"))
    candidate = json.loads((candidate_folder / "report.json").read_text(encoding="utf-8"))
    comparable = ("source_hash", "asset_hash", "course_catalog_hash", "training_course_seed")
    if (
        any(baseline.get(key) != candidate.get(key) for key in comparable)
        or baseline.get("late_swing_actor_hash") != candidate.get("late_swing_actor_hash")
        or baseline.get("support_knee_retract_m") != 0.0
        or candidate.get("support_knee_retract_m") not in (0.0, 0.04, 0.08)
        or baseline.get("selected_taskspace_mask") != candidate.get("selected_taskspace_mask")
        or baseline.get("frames") != candidate.get("frames")
        or len(baseline["environments"]) != len(candidate["environments"])
        or any(
            a["course"] != b["course"]
            for a, b in zip(baseline["environments"], candidate["environments"], strict=True)
        )
    ):
        raise ValueError("unpaired first-touch experiment")
    count = len(baseline["environments"])
    trace_path = candidate_folder / "late_swing_action_trace.npz"
    if candidate.get("late_swing_action_trace_hash") != hash_bytes(trace_path.read_bytes()):
        raise ValueError("candidate action trace hash mismatch")
    with np.load(trace_path, allow_pickle=False) as action:
        support = action["applied_support_knee_joint_delta_rad"]
    if support.shape != (candidate["frames"], count, 29) or not np.isfinite(support).all():
        raise ValueError("invalid support action trace")
    untreated = np.all(np.abs(support) <= 1e-6, axis=(0, 2))
    first_contact = np.asarray(
        [
            min(
                [
                    frame
                    for frame in (a["first_contact_frame"], b["first_contact_frame"])
                    if frame is not None
                ]
                or [candidate["frames"]]
            )
            for a, b in zip(baseline["environments"], candidate["environments"], strict=True)
        ],
        dtype=np.int64,
    )
    with (
        np.load(baseline_folder / "body_trace.npz", allow_pickle=False) as a,
        np.load(candidate_folder / "body_trace.npz", allow_pickle=False) as b,
    ):
        drift = first_untreated_precontact_drift(
            a["root_pose_xyzw_m"],
            b["root_pose_xyzw_m"],
            a["joint_target_rad"],
            b["joint_target_rad"],
            untreated,
            first_contact,
        )
    result: dict[str, Any] = {
        "schema": "rsi_paired_lane_isolation_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "baseline_report_hash": baseline_audit["source_report_hash"],
        "candidate_report_hash": candidate_audit["source_report_hash"],
        "untreated_lane_count": int(np.sum(untreated)),
        "untreated_precontact_drift_first_frame": {str(k): v for k, v in sorted(drift.items())},
        "lane_level_causal_attribution_authorized": not bool(drift),
        "body_trace_byte_identical": baseline["body_trace_hash"] == candidate["body_trace_hash"],
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result
