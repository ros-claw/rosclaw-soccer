"""Bind measured shot-option failure diagnostics to immutable match evidence."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth import shot_failure_feedback
from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_competitive_match_growth import (
    validate_continuous_competitive_match_growth,
)


def audit_shots(exams: tuple[Path, ...]) -> dict[str, Any]:
    if not exams or len(exams) > 256 or len(set(p.resolve() for p in exams)) != len(exams):
        raise ValueError("bounded distinct match evidence required")
    rows = []
    for path in exams:
        report = validate_continuous_competitive_match_growth(path)
        if not report["exact_replay"]:
            raise ValueError("shot diagnostics require reproducible physical evidence")
        policy = NearBallResidualPolicy.load(path.parent / "near-ball-policy.npz")
        if policy.policy_hash != report["near_ball_policy_hash"]:
            raise ValueError("shot feedback actor identity differs")
        raw = path.parent / "primary.npz"
        with np.load(raw, allow_pickle=False) as archive:
            trace = {key: archive[key] for key in archive.files}
        rows.append(
            {
                "exam": str(path.resolve()),
                "exam_hash": report["report_hash"],
                "trajectory_file_hash": hash_bytes(raw.read_bytes()),
                "world_safe": report["primary_assessment"]["safe"],
                "options": shot_failure_feedback.diagnose_shot_options(
                    trace,
                    policy.agent_ids,
                    goal_planes_x_m=(
                        report["world_config"]["left_goal_plane_x_m"],
                        report["fixture"]["goal"]["plane_x_m"],
                    ),
                ),
            }
        )
    result = {
        "schema": "rosclaw_soccer.shot_failure_feedback_audit.v1",
        "diagnostics_only": True,
        "promotion_eligible": False,
        "activation_ceiling": "SIM_ONLY",
        "rules_violation_adjudicated": False,
        "feedback_source_hash": hash_bytes(Path(shot_failure_feedback.__file__).read_bytes()),
        "exams": rows,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--exam", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = audit_shots(tuple(args.exam))
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(report["report_hash"])


if __name__ == "__main__":
    main()
