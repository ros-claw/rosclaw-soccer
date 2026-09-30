"""Fail-closed paired audit of 16-way Isaac lanes against isolated G1 courses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_json


def verify(batch_root: Path, independent_root: Path) -> dict[str, Any]:
    batch = json.loads((batch_root / "report.json").read_text(encoding="utf-8"))
    batch_audit = audit_vector_first_touch(batch_root)
    if (
        batch.get("schema") != "rsi_isaac_vector_first_touch_smoke_v1"
        or batch.get("activation_ceiling") != "SIM_ONLY"
        or batch.get("training_course_seed") != 20260953
        or batch.get("lane_spacing_m") != 24.0
        or len(batch.get("environments", [])) != 16
        or batch_audit["independent_physical_episode_count"] != 16
    ):
        raise ValueError("unauthenticated wide-lane batch")
    parent_summary = json.loads(
        (independent_root / "parent_summary.json").read_text(encoding="utf-8")
    )
    if (
        parent_summary.get("schema") != "rsi_independent_first_touch_parents_v1"
        or parent_summary.get("failures") != []
        or parent_summary.get("report_hash")
        != hash_json({k: v for k, v in parent_summary.items() if k != "report_hash"})
        or [(row["seed"], row["lane"]) for row in parent_summary.get("episodes", [])]
        != [(20260953, lane) for lane in range(0, 16, 2)]
        or parent_summary.get("source_hash") != batch["source_hash"]
        or parent_summary.get("asset_hash") != batch["asset_hash"]
    ):
        raise ValueError("unpaired independent G1 parent bank")
    rows: list[dict[str, Any]] = []
    with np.load(batch_root / "trace.npz", allow_pickle=False) as batch_trace:
        batch_ball = batch_trace["ball_position_m"]
        for lane in range(0, 16, 2):
            folder = independent_root / f"seed20260953-lane{lane}-parent"
            parent = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            physical = audit_vector_first_touch(folder)
            summary = parent_summary["episodes"][lane // 2]
            if (
                parent["report_hash"] != summary["report_hash"]
                or physical["report_hash"] != summary["physical_audit_hash"]
                or parent.get("source_hash") != batch["source_hash"]
                or parent.get("asset_hash") != batch["asset_hash"]
                or parent.get("sonic_qualification_hash") != batch["sonic_qualification_hash"]
                or parent["environments"][0]["course"] != batch["environments"][lane]["course"]
            ):
                raise ValueError(f"unpaired independent lane {lane}")
            one = parent["environments"][0]
            many = batch["environments"][lane]
            independent_first = one["first_contact_frame"]
            batch_first = many["first_contact_frame"]
            with np.load(folder / "trace.npz", allow_pickle=False) as independent_trace:
                independent_disp = post_contact_displacement(
                    independent_trace["ball_position_m"][:, 0], independent_first
                )
            batch_disp = post_contact_displacement(batch_ball[:, lane], batch_first)
            first_agrees = (
                independent_first is None
                and batch_first is None
                or independent_first is not None
                and batch_first is not None
                and abs(independent_first - batch_first) <= 1
            )
            displacement_agrees = (
                independent_first is None
                and batch_first is None
                or independent_disp["forward_60_m"] is not None
                and batch_disp["forward_60_m"] is not None
                and independent_disp["lateral_60_m"] is not None
                and batch_disp["lateral_60_m"] is not None
                and abs(independent_disp["forward_60_m"] - batch_disp["forward_60_m"]) <= 0.10
                and abs(independent_disp["lateral_60_m"] - batch_disp["lateral_60_m"]) <= 0.10
            )
            passed = bool(
                first_agrees
                and one["contact_body_indices"] == many["contact_body_indices"]
                and abs(one["minimum_pelvis_z_m"] - many["minimum_pelvis_z_m"]) <= 0.02
                and displacement_agrees
            )
            rows.append(
                {
                    "lane": lane,
                    "independent_first_contact_frame": independent_first,
                    "batch_first_contact_frame": batch_first,
                    "independent_contact_body_indices": one["contact_body_indices"],
                    "batch_contact_body_indices": many["contact_body_indices"],
                    "independent_displacement": independent_disp,
                    "batch_displacement": batch_disp,
                    "independent_minimum_pelvis_z_m": one["minimum_pelvis_z_m"],
                    "batch_minimum_pelvis_z_m": many["minimum_pelvis_z_m"],
                    "equivalent": passed,
                }
            )
    result: dict[str, Any] = {
        "schema": "rsi_wide_lane_isolation_equivalence_v1",
        "activation_ceiling": "SIM_ONLY",
        "batch_report_hash": batch["report_hash"],
        "batch_physical_audit_hash": batch_audit["report_hash"],
        "independent_parent_bank_hash": parent_summary["report_hash"],
        "lanes": rows,
        "equivalence_gate_passed": all(row["equivalent"] for row in rows),
        "batched_training_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", required=True, type=Path)
    parser.add_argument("--independent-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.batch_root == args.independent_root:
        parser.error("new output and distinct physical evidence roots required")
    result = verify(args.batch_root, args.independent_root)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"WIDE_LANE_EQUIVALENCE={result['report_hash']}", flush=True)
    print(f"EQUIVALENCE_GATE_PASSED={result['equivalence_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
