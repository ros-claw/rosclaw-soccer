"""Export a hash-bound, JSON-only SIM_ONLY approach gate after development validation."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

CV_HASH = "sha256:50e98c7810bf672d21a52cd6d33703dd729aa4e62f07757da0e4187c4ff1bd15"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cv-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = json.loads(args.cv_report.read_text(encoding="utf-8"))
    if (
        args.output.exists()
        or report.get("report_hash") != CV_HASH
        or report.get("report_hash")
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report.get("development_gate_passed") is not True
        or report.get("x_margin_m") != 0.05
        or report.get("y_margin_m") != 0.02
    ):
        parser.error("committed guarded CV and new output required")
    limits = report["effective_full_fit_rectangle"]
    policy: dict[str, Any] = {
        "schema": "rsi_isaac_guarded_approach_policy_v1",
        "activation_ceiling": "SIM_ONLY",
        "cv_report_hash": CV_HASH,
        "x_max_m": limits["x_max_m"],
        "y_min_m": limits["y_min_m"],
        "navigation_lateral_ball_gain": 0.8,
        "promotion_authorized": False,
    }
    policy["policy_hash"] = hash_json(policy)
    args.output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"GUARDED_APPROACH_POLICY={policy['policy_hash']}", flush=True)


if __name__ == "__main__":
    main()
