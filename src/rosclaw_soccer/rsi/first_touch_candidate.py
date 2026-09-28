"""Bounded SIM_ONLY first-touch action table for independent Isaac episodes."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_json

JOINT_NAMES = (
    "left_hip_pitch_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "right_hip_pitch_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
)
MAX_RESIDUAL_RAD = 0.08


def guarded_residual_target(
    baseline_rad: float, delta_rad: float, lower_rad: float, upper_rad: float
) -> float:
    """Never push a frozen SONIC target further outside its physical joint limit."""

    proposed, projected = project_residual_target(baseline_rad, delta_rad, lower_rad, upper_rad)
    if projected:
        raise ValueError("candidate residual moves outside physical joint limits")
    return proposed


def project_residual_target(
    baseline_rad: float, delta_rad: float, lower_rad: float, upper_rad: float
) -> tuple[float, bool]:
    """Explicit shield: rejected directions execute the frozen Parent target."""

    if not all(math.isfinite(v) for v in (baseline_rad, delta_rad, lower_rad, upper_rad)):
        raise ValueError("nonfinite joint target or limit")
    if lower_rad >= upper_rad or abs(delta_rad) > MAX_RESIDUAL_RAD + 1e-8:
        raise ValueError("unbounded residual or invalid physical joint limits")
    proposed = baseline_rad + delta_rad
    projected = bool(
        lower_rad <= baseline_rad <= upper_rad
        and not lower_rad <= proposed <= upper_rad
        or baseline_rad < lower_rad
        and delta_rad < 0
        or baseline_rad > upper_rad
        and delta_rad > 0
    )
    return (baseline_rad, True) if projected else (proposed, False)


@dataclass(frozen=True)
class FirstTouchCandidate:
    parent_report_hash: str
    candidate_hash: str
    actions_rad: tuple[tuple[float, ...], ...]


def load_first_touch_candidate(
    path: Path,
    *,
    expected_courses: tuple[tuple[float, float, float], ...],
    parent_report_hash: str,
) -> FirstTouchCandidate:
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    body = {key: value for key, value in data.items() if key != "candidate_hash"}
    if (
        data.get("schema") != "rsi_isaac_first_touch_candidate_v1"
        or data.get("activation_ceiling") != "SIM_ONLY"
        or data.get("partition") != "CONSUMED_DEV"
        or data.get("promotion_authorized") is not False
        or data.get("parent_report_hash") != parent_report_hash
        or data.get("joint_names") != list(JOINT_NAMES)
        or data.get("courses") != [list(course) for course in expected_courses]
    ):
        raise ValueError("first-touch candidate or frozen Parent binding invalid")
    actions = data.get("actions_rad")
    if (
        not isinstance(actions, list)
        or len(actions) != len(expected_courses)
        or any(
            not isinstance(row, list)
            or len(row) != len(JOINT_NAMES)
            or any(
                type(value) not in (int, float)
                or not math.isfinite(value)
                or abs(value) > MAX_RESIDUAL_RAD
                for value in row
            )
            for row in actions
        )
    ):
        raise ValueError("nonfinite or unbounded first-touch residual action")
    if data.get("candidate_hash") != hash_json(body):
        raise ValueError("first-touch candidate digest invalid")
    return FirstTouchCandidate(
        parent_report_hash=parent_report_hash,
        candidate_hash=data["candidate_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in actions),
    )


def candidate_manifest(
    *,
    courses: tuple[tuple[float, float, float], ...],
    parent_report_hash: str,
    actions_rad: tuple[tuple[float, ...], ...],
    seed: int,
) -> dict[str, Any]:
    if type(seed) is not int or not 0 <= seed < 2**31:
        raise ValueError("bounded deterministic candidate seed required")
    if len(courses) != len(actions_rad) or len(set(courses)) != len(courses):
        raise ValueError("one unique physical course per candidate action required")
    body: dict[str, Any] = {
        "schema": "rsi_isaac_first_touch_candidate_v1",
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "promotion_authorized": False,
        "parent_report_hash": parent_report_hash,
        "joint_names": list(JOINT_NAMES),
        "courses": [list(course) for course in courses],
        "actions_rad": [list(row) for row in actions_rad],
        "seed": seed,
    }
    body["candidate_hash"] = hash_json(body)
    return body


def main() -> None:
    import numpy as np

    from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("zero", "random"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--scale-rad", type=float, default=0.03)
    args = parser.parse_args()
    parent_audit = audit_vector_first_touch(args.parent_report.parent)
    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if parent_audit["source_report_hash"] != parent["report_hash"]:
        raise ValueError("parent report does not match authenticated physics")
    if not math.isfinite(args.scale_rad) or not 0 < args.scale_rad <= MAX_RESIDUAL_RAD:
        raise ValueError("bounded random action scale required")
    courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )
    rng = np.random.default_rng(args.seed)
    actions = (
        np.zeros((len(courses), len(JOINT_NAMES)))
        if args.mode == "zero"
        else np.clip(
            rng.normal(0.0, args.scale_rad, (len(courses), len(JOINT_NAMES))),
            -MAX_RESIDUAL_RAD,
            MAX_RESIDUAL_RAD,
        )
    )
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        actions_rad=tuple(tuple(float(value) for value in row) for row in actions),
        seed=args.seed,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"candidate_hash": manifest["candidate_hash"], "course_count": len(courses)}))


if __name__ == "__main__":
    main()
