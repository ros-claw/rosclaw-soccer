"""SIM_ONLY held-out-course diagnosis of temporal exploration; no policy update."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_temporal_success_diagnostics_v197.result.v1"
WINDOWS = ((15, 25), (25, 30), (30, 35), (35, 40), (40, 50), (50, 65))
PERMUTATIONS = 499
SEED = 197001


def _check_report(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text())
    digest = value.get("report_hash")
    if digest != hash_json({key: item for key, item in value.items() if key != "report_hash"}):
        raise ValueError(f"unsealed report: {path}")
    return value


def _load_rows(
    experience_dir: Path, report: dict[str, Any], checkpoint: Path, expected_hash: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if hash_bytes(checkpoint.read_bytes()) != expected_hash:
        raise ValueError("temporal checkpoint hash mismatch")
    with np.load(checkpoint, allow_pickle=False) as arrays:
        if set(arrays.files) != {"input_matrix", "input_bias", "output_matrix", "output_bias"}:
            raise ValueError("unsafe temporal checkpoint fields")
        first = np.asarray(arrays["input_matrix"], dtype=np.float64).reshape(32, 10)
        first_bias = np.asarray(arrays["input_bias"], dtype=np.float64).reshape(32)
        last = np.asarray(arrays["output_matrix"], dtype=np.float64).reshape(12, 32)
        last_bias = np.asarray(arrays["output_bias"], dtype=np.float64).reshape(12)
    if not all(np.isfinite(array).all() for array in (first, first_bias, last, last_bias)):
        raise ValueError("non-finite temporal checkpoint")
    vectors = []
    labels = []
    groups = []
    seen = set()
    for row in report["rows"]:
        course = int(row["course_index"])
        repetition = int(row["repetition"])
        if (course, repetition) in seen:
            raise ValueError("duplicate experience trajectory")
        seen.add((course, repetition))
        path = experience_dir / f"course-{row['course']['seed']}-rep-{repetition}.npz"
        if hash_bytes(path.read_bytes()) != row["trajectory_hash"]:
            raise ValueError("experience trajectory hash mismatch")
        with np.load(path, allow_pickle=False) as arrays:
            if set(arrays.files) != {"frames", "features", "logits", "shaped_reward"}:
                raise ValueError("unsafe experience trajectory fields")
            frames = np.asarray(arrays["frames"], dtype=np.int64)
            features = np.asarray(arrays["features"], dtype=np.float64)
            logits = np.asarray(arrays["logits"], dtype=np.float64)
        if (
            not np.array_equal(frames, np.arange(15, 65))
            or features.shape != (50, 10)
            or logits.shape != (50, 12)
            or not np.isfinite(features).all()
            or not np.isfinite(logits).all()
        ):
            raise ValueError("complete finite exploration trace required")
        mean = np.tanh(features @ first.T + first_bias) @ last.T + last_bias
        noise = logits - mean
        vectors.append(
            np.concatenate([noise[start - 15 : end - 15].mean(axis=0) for start, end in WINDOWS])
        )
        labels.append(bool(row["strict_controlled"]))
        groups.append(course)
    if len(vectors) != 128 or len(seen) != 128:
        raise ValueError("all 128 sealed trajectories required")
    return np.asarray(vectors), np.asarray(labels, dtype=bool), np.asarray(groups)


def _auc(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        raise ValueError("both outcomes required")
    comparison = positive[:, None] - negative[None, :]
    return float(np.mean((comparison > 0) + 0.5 * (comparison == 0)))


def _cross_validate(
    vectors: np.ndarray, labels: np.ndarray, groups: np.ndarray, eligible: tuple[int, ...]
) -> tuple[float, dict[int, float]]:
    directions = {}
    for course in eligible:
        mask = groups == course
        delta = vectors[mask & labels].mean(axis=0) - vectors[mask & ~labels].mean(axis=0)
        norm = float(np.linalg.norm(delta))
        directions[course] = delta / norm if norm > 1e-12 else np.zeros_like(delta)
    results = {}
    for course in eligible:
        mask = groups == course
        direction = np.mean([directions[other] for other in eligible if other != course], axis=0)
        results[course] = _auc(vectors[mask] @ direction, labels[mask])
    return float(np.mean(list(results.values()))), results


def diagnose(
    experience_dir: Path, v193_path: Path, checkpoint: Path, output: Path
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY diagnostic evidence required")
    experience = _check_report(experience_dir / "report.json")
    v193 = _check_report(v193_path)
    expected_hash = v193["history"][1]["checkpoint_hash"]
    if (
        experience["status"] != "DEVELOPMENT_TEMPORAL_EXPERIENCE_ONLY"
        or experience["checkpoint_hash"] != expected_hash
        or experience["total_controlled"] != 17
    ):
        raise ValueError("sealed temporal experience lineage required")
    vectors, labels, groups = _load_rows(experience_dir, experience, checkpoint, expected_hash)
    counts = {int(course): int(np.sum(labels[groups == course])) for course in np.unique(groups)}
    eligible = tuple(course for course, positive in counts.items() if 2 <= positive <= 6)
    if len(eligible) != 7:
        raise ValueError("seven repeated-success courses required")
    observed, per_course = _cross_validate(vectors, labels, groups, eligible)
    rng = np.random.default_rng(SEED)
    null = []
    for _ in range(PERMUTATIONS):
        shuffled = labels.copy()
        for course in eligible:
            mask = groups == course
            shuffled[mask] = rng.permutation(shuffled[mask])
        null.append(_cross_validate(vectors, shuffled, groups, eligible)[0])
    p_value = (1 + sum(value >= observed for value in null)) / (PERMUTATIONS + 1)
    eligible_mask = np.isin(groups, eligible)
    difference = vectors[eligible_mask & labels].mean(axis=0) - vectors[
        eligible_mask & ~labels
    ].mean(axis=0)
    top_indices = np.argsort(np.abs(difference))[-10:][::-1]
    top = [
        {
            "window_frames": WINDOWS[int(index) // 12],
            "joint": G1_DDS_JOINT_NAMES[int(index) % 12],
            "success_minus_failure_mean_logit_noise": float(difference[index]),
        }
        for index in top_indices
    ]
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_temporal_success_diagnostics_v197.py",
            "scripts/rsi_r1_temporal_experience_v195.py",
            "src/rosclaw_soccer/rsi/receiving_temporal_motor_expert.py",
        )
    }
    transferable = (
        observed >= 0.65 and p_value < 0.05 and sum(v > 0.5 for v in per_course.values()) >= 5
    )
    report = {
        "schema": SCHEMA,
        "experience_report_hash": experience["report_hash"],
        "checkpoint_hash": expected_hash,
        "source_hashes": sources,
        "partition": "LEAVE_ONE_CONSUMED_COURSE_OUT_DIAGNOSTIC_NOT_FRESH",
        "windows": WINDOWS,
        "success_counts": counts,
        "eligible_courses": eligible,
        "observed_macro_auc": observed,
        "per_course_auc": per_course,
        "permutation_count": PERMUTATIONS,
        "permutation_seed": SEED,
        "permutation_p_value": p_value,
        "null_auc_mean": float(np.mean(null)),
        "top_associations_not_causes": top,
        "status": "TRANSFERABLE_TEMPORAL_DIRECTION_DIAGNOSTIC_ONLY"
        if transferable
        else "REJECTED_NO_TRANSFERABLE_TEMPORAL_DIRECTION",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during diagnostic")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--experience-dir", type=Path, required=True)
    parser.add_argument("--v193-report", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = diagnose(args.experience_dir, args.v193_report, args.checkpoint, args.output)
    print(
        json.dumps(
            {
                key: result[key]
                for key in ("status", "observed_macro_auc", "permutation_p_value", "report_hash")
            }
        )
    )


if __name__ == "__main__":
    main()
