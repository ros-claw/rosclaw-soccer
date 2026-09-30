"""Audit completed bilateral motor trials and explain measured control authority."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.contact_motor_response import precontact_response
from rosclaw_soccer.sim.contracts import hash_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new analysis output required")
    commitment = json.loads((args.root / "commitment.json").read_text())
    rows = []
    for path in sorted(args.root.glob("*-g*-c*-actor/report.json")):
        report = json.loads(path.read_text())
        seed, lane = report["training_course_seed"], report["single_course_lane"]
        parent_path = args.root / f"seed{seed}-lane{lane}-gain_12-actor"
        parent = json.loads((parent_path / "report.json").read_text())
        audits = [audit_lateral_approach(folder) for folder in (parent_path, path.parent)]
        if report["source_hash"] != commitment["runner_hash"] or any(
            report[k] != parent[k]
            for k in (
                "source_hash",
                "asset_hash",
                "sonic_qualification_hash",
                "training_course_seed",
                "single_course_lane",
            )
        ):
            raise ValueError("paired motor response provenance differs")
        with (
            np.load(parent_path / "body_trace.npz", allow_pickle=False) as base_body,
            np.load(path.parent / "body_trace.npz", allow_pickle=False) as body,
        ):
            response = precontact_response(
                base_body,
                body,
                parent["environments"][0]["first_contact_frame"],
                report["environments"][0]["first_contact_frame"],
            )
        rows.append(
            {
                "folder": str(path.parent),
                "seed": seed,
                "lane": lane,
                "candidate_report_hash": report["report_hash"],
                "parent_report_hash": parent["report_hash"],
                "audit_hashes": [a["report_hash"] for a in audits],
                "contact_body_indices": report["environments"][0]["contact_body_indices"],
                "minimum_pelvis_z_m": report["environments"][0]["minimum_pelvis_z_m"],
                "maximum_lateral_excursion_m": report["single_instance_max_lateral_excursion_m"],
                "response": response,
            }
        )
    if not rows:
        raise ValueError("no completed motor trial")
    result = {
        "schema": "rsi_bilateral_motor_response_analysis_v303",
        "activation_ceiling": "SIM_ONLY",
        "commitment": commitment,
        "rows": rows,
        "completed_trial_count": len(rows),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")
    print(json.dumps({"completed": len(rows), "report_hash": result["report_hash"]}))


if __name__ == "__main__":
    main()
