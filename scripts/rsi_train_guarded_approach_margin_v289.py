"""Recheck the frozen learned approach gate after fixed physical guard bands."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_train_conservative_approach_rectangle_v288 import cross_validate, load_courses

PARENT_HASH = "sha256:49d880006733d8557e91b0b706aed4e7ae94ced952136f560b40fa075e40821f"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v286-root", required=True, type=Path)
    parser.add_argument("--v287-root", required=True, type=Path)
    parser.add_argument("--parent-cv", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    parent = json.loads(args.parent_cv.read_text(encoding="utf-8"))
    if (
        args.output.exists()
        or parent.get("report_hash") != PARENT_HASH
        or parent.get("report_hash")
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent.get("development_gate_passed") is not True
    ):
        parser.error("committed v288 development result and new output required")
    courses = load_courses({"v286": args.v286_root, "v287": args.v287_root})
    result = cross_validate(courses, x_margin_m=0.05, y_margin_m=0.02)
    result["parent_cv_report_hash"] = PARENT_HASH
    result["report_hash"] = hash_json(
        {key: value for key, value in result.items() if key != "report_hash"}
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"GUARDED_APPROACH_CV={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
