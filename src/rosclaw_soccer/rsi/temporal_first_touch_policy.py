"""SIM_ONLY bounded, proprioceptive first-touch motor residual policy.

This is a trainable policy parameterization, not a manually timed kick. Its
features use only current ball/body observations; six-joint SONIC remains the
frozen foundation, and the existing joint-limit shield remains authoritative.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_candidate import MAX_RESIDUAL_RAD
from rosclaw_soccer.sim.contracts import hash_json

SCHEMA = "rsi_isaac_temporal_first_touch_candidate_v1"
JOINT_NAMES = ("right_hip_pitch_joint", "right_knee_joint", "right_ankle_pitch_joint")
FEATURE_NAMES = (
    "bias",
    "longitudinal_ball_gap",
    "lateral_ball_gap",
    "ball_velocity_x",
    "right_knee_position",
    "right_knee_velocity",
)
MAX_WEIGHT = 1.0
MAX_FOLLOWTHROUGH_FRAMES = 30


@dataclass(frozen=True)
class TemporalFirstTouchCandidate:
    parent_report_hash: str
    candidate_hash: str
    weights_per_course: np.ndarray


def temporal_residual(
    weights: np.ndarray,
    *,
    ball_relative_xyz_m: tuple[float, float, float],
    ball_vx_m_s: float,
    joint_position_rad: np.ndarray,
    joint_velocity_rad_s: np.ndarray,
) -> np.ndarray:
    """Compute shieldable right-leg target residual from current observations."""
    gap, lateral, _ = ball_relative_xyz_m
    if (
        weights.shape != (len(FEATURE_NAMES), len(JOINT_NAMES))
        or joint_position_rad.shape != (29,)
        or joint_velocity_rad_s.shape != (29,)
        or not np.isfinite(weights).all()
        or float(np.max(np.abs(weights))) > MAX_WEIGHT
        or not np.isfinite(ball_relative_xyz_m).all()
        or not math.isfinite(ball_vx_m_s)
        or not np.isfinite(joint_position_rad).all()
        or not np.isfinite(joint_velocity_rad_s).all()
    ):
        raise ValueError("unbounded or nonfinite temporal first-touch input")
    if not 0.15 < gap < 1.25 or abs(lateral) > 0.5 or ball_vx_m_s >= -0.05:
        return np.zeros(len(JOINT_NAMES))
    features = np.asarray(
        (
            1.0,
            (gap - 0.7) / 0.55,
            lateral / 0.5,
            ball_vx_m_s / 0.7,
            (joint_position_rad[9] - 0.7) / 0.7,
            joint_velocity_rad_s[9] / 5.0,
        )
    )
    features = np.clip(features, -2.0, 2.0)
    envelope = min(1.0, (1.25 - gap) / 0.35, (gap - 0.15) / 0.2)
    result: np.ndarray = MAX_RESIDUAL_RAD * envelope * np.tanh(features @ weights)
    if result.shape != (len(JOINT_NAMES),) or not np.isfinite(result).all():
        raise ValueError("temporal first-touch policy produced invalid action")
    return result


def followthrough_residual(
    contact_residual_rad: np.ndarray, *, elapsed_frames: int, followthrough_frames: int
) -> np.ndarray:
    """Bounded, continuous fade from the physically applied contact target."""
    if (
        contact_residual_rad.shape != (len(JOINT_NAMES),)
        or not np.isfinite(contact_residual_rad).all()
        or float(np.max(np.abs(contact_residual_rad))) > MAX_RESIDUAL_RAD + 1e-6
        or type(elapsed_frames) is not int
        or elapsed_frames < 1
        or type(followthrough_frames) is not int
        or not 0 <= followthrough_frames <= MAX_FOLLOWTHROUGH_FRAMES
    ):
        raise ValueError("invalid bounded postcontact transition")
    if elapsed_frames > followthrough_frames:
        return np.zeros(len(JOINT_NAMES))
    result: np.ndarray = contact_residual_rad * (1.0 - elapsed_frames / (followthrough_frames + 1))
    return result


def candidate_manifest(
    *,
    courses: tuple[tuple[float, float, float], ...],
    parent_report_hash: str,
    weights_per_course: np.ndarray,
    seed: int,
    actor_state_hash: str | None = None,
) -> dict[str, Any]:
    if (
        type(seed) is not int
        or not 0 <= seed < 2**31
        or len(courses) != 16
        or len(set(courses)) != 16
        or weights_per_course.shape != (16, len(FEATURE_NAMES), len(JOINT_NAMES))
        or not np.isfinite(weights_per_course).all()
        or float(np.max(np.abs(weights_per_course))) > MAX_WEIGHT
        or not isinstance(parent_report_hash, str)
        or len(parent_report_hash) != 71
        or (
            actor_state_hash is not None
            and (not isinstance(actor_state_hash, str) or len(actor_state_hash) != 71)
        )
    ):
        raise ValueError("invalid temporal policy manifest")
    body: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "partition": "CONSUMED_DEV",
        "promotion_authorized": False,
        "parent_report_hash": parent_report_hash,
        "courses": [list(row) for row in courses],
        "joint_names": list(JOINT_NAMES),
        "feature_names": list(FEATURE_NAMES),
        "weights_per_course": weights_per_course.tolist(),
        "seed": seed,
        "actor_state_hash": actor_state_hash,
    }
    body["candidate_hash"] = hash_json(body)
    return body


def load_candidate(
    path: Path,
    *,
    expected_courses: tuple[tuple[float, float, float], ...],
    parent_report_hash: str,
) -> TemporalFirstTouchCandidate:
    data = json.loads(path.read_text(encoding="utf-8"))
    if (
        set(data)
        != {
            "schema",
            "activation_ceiling",
            "partition",
            "promotion_authorized",
            "parent_report_hash",
            "courses",
            "joint_names",
            "feature_names",
            "weights_per_course",
            "seed",
            "actor_state_hash",
            "candidate_hash",
        }
        or data.get("schema") != SCHEMA
        or data.get("activation_ceiling") != "SIM_ONLY"
        or data.get("partition") != "CONSUMED_DEV"
        or data.get("promotion_authorized") is not False
        or data.get("parent_report_hash") != parent_report_hash
        or data.get("courses") != [list(row) for row in expected_courses]
        or data.get("joint_names") != list(JOINT_NAMES)
        or data.get("feature_names") != list(FEATURE_NAMES)
        or data.get("candidate_hash")
        != hash_json({key: value for key, value in data.items() if key != "candidate_hash"})
    ):
        raise ValueError("temporal policy commitment or Parent binding invalid")
    weights = np.asarray(data["weights_per_course"], dtype=np.float64)
    canonical = candidate_manifest(
        courses=expected_courses,
        parent_report_hash=parent_report_hash,
        weights_per_course=weights,
        seed=data["seed"],
        actor_state_hash=data["actor_state_hash"],
    )
    if canonical != data:
        raise ValueError("temporal policy canonical form changed")
    return TemporalFirstTouchCandidate(parent_report_hash, data["candidate_hash"], weights)


def main() -> None:
    from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--mode", required=True, choices=("zero", "random"))
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--weight-std", type=float, default=0.5)
    args = parser.parse_args()
    parent_audit = audit_vector_first_touch(args.parent_report.parent)
    parent = json.loads(args.parent_report.read_text(encoding="utf-8"))
    if (
        parent_audit["source_report_hash"] != parent["report_hash"]
        or parent.get("torch_batch_plan_only") is not True
        or parent.get("navigation_speed_mps", 1.4) != 1.4
        or "near_ball_gap_m" in parent
        or not math.isfinite(args.weight_std)
        or not 0 < args.weight_std <= MAX_WEIGHT
    ):
        raise ValueError("temporal policy requires frozen 1.4 m/s training Parent")
    courses = tuple(
        (
            row["course"]["ball_x_m"],
            row["course"]["ball_y_local_m"],
            row["course"]["ball_vx_m_s"],
        )
        for row in parent["environments"]
    )
    weights = (
        np.zeros((16, len(FEATURE_NAMES), len(JOINT_NAMES)))
        if args.mode == "zero"
        else np.clip(
            np.random.default_rng(args.seed).normal(
                0.0, args.weight_std, (16, len(FEATURE_NAMES), len(JOINT_NAMES))
            ),
            -MAX_WEIGHT,
            MAX_WEIGHT,
        )
    )
    manifest = candidate_manifest(
        courses=courses,
        parent_report_hash=parent["report_hash"],
        weights_per_course=weights,
        seed=args.seed,
    )
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps({"candidate_hash": manifest["candidate_hash"], "course_count": 16}))


if __name__ == "__main__":
    main()
