"""SIM_ONLY contact-order diagnostics over authenticated Parent/Candidate traces."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.vector_first_touch_evidence import (
    audit_first_touch_candidate_execution,
    audit_vector_first_touch,
)
from rosclaw_soccer.sim.contracts import hash_json

FOOT_INDICES = (0, 1)
NONFOOT_INDICES = (2, 3, 4, 5)
CONTACT_THRESHOLD_N = 1.0


def _first_frame(force: np.ndarray, body_indices: tuple[int, ...]) -> int | None:
    active = np.flatnonzero(np.max(force[:, body_indices], axis=1) > CONTACT_THRESHOLD_N)
    return int(active[0]) if len(active) else None


def _course_contacts(folder: Path) -> list[dict[str, Any]]:
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        forces = trace["ball_body_contact_force_peak_n"]
    if forces.ndim != 3 or forces.shape[2] != 6 or not np.isfinite(forces).all():
        raise ValueError("invalid contact-force trace")
    rows = []
    for index in range(forces.shape[1]):
        series = forces[:, index]
        rows.append(
            {
                "environment": index,
                "first_foot_frame": _first_frame(series, FOOT_INDICES),
                "first_nonfoot_frame": _first_frame(series, NONFOOT_INDICES),
                "peak_foot_force_n": float(np.max(series[:, FOOT_INDICES])),
                "peak_nonfoot_force_n": float(np.max(series[:, NONFOOT_INDICES])),
            }
        )
    return rows


def diagnose_contact_order(
    parent_folder: Path, candidate_folder: Path, candidate_manifest: Path
) -> dict[str, Any]:
    parent_audit = audit_vector_first_touch(parent_folder)
    candidate_audit = audit_first_touch_candidate_execution(
        candidate_folder,
        parent_folder=parent_folder,
        candidate_path=candidate_manifest,
    )
    parent = _course_contacts(parent_folder)
    candidate = _course_contacts(candidate_folder)
    if len(parent) != len(candidate):
        raise ValueError("contact-order comparison requires identical course count")
    changes = []
    for before, after in zip(parent, candidate, strict=True):
        changes.append(
            {
                "environment": before["environment"],
                "parent": before,
                "candidate": after,
                "nonfoot_removed": before["first_nonfoot_frame"] is not None
                and after["first_nonfoot_frame"] is None,
                "nonfoot_new": before["first_nonfoot_frame"] is None
                and after["first_nonfoot_frame"] is not None,
            }
        )
    report = {
        "schema": "rsi_isaac_first_touch_contact_diagnostics_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "parent_audit_hash": parent_audit["report_hash"],
        "candidate_audit_hash": candidate_audit["report_hash"],
        "parent_clean_count": parent_audit["clean_foot_only_episode_count"],
        "candidate_clean_count": candidate_audit["candidate_clean_foot_only_count"],
        "course_changes": changes,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-folder", required=True, type=Path)
    parser.add_argument("--candidate-folder", required=True, type=Path)
    parser.add_argument("--candidate-manifest", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = diagnose_contact_order(
        args.parent_folder, args.candidate_folder, args.candidate_manifest
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
