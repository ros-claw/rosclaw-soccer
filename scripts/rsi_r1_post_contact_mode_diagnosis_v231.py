"""SIM_ONLY diagnose whether online contact successes occupy distinct action modes."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_post_contact_mode_diagnosis_v231.result.v1"


def _spread(values: np.ndarray) -> float | None:
    if len(values) < 2:
        return None
    center = values.mean(axis=0)
    return float(np.sqrt(np.mean(np.sum((values - center) ** 2, axis=1))))


def diagnose(v229_dir: Path, v230_report_path: Path, output: Path) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY trajectory-mode evidence required")
    v229, v230 = (_checked(path) for path in (v229_dir / "report.json", v230_report_path))
    if (
        v230["status"] != "REJECTED_POST_CONTACT_AWR_GATE"
        or v230["v229_report_hash"] != v229["report_hash"]
        or len(v230["trajectory_manifest"]) != 99
    ):
        raise ValueError("sealed failed AWR and online trajectories required")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for history in v229["history"]:
        for index, item in enumerate(history["sample_trajectories"]):
            path = v229_dir / f"sample-u{history['update']}-e{index}.npz"
            if hash_bytes(path.read_bytes()) != item["trajectory_hash"]:
                raise ValueError("sealed online trajectory required")
            with np.load(path, allow_pickle=False) as arrays:
                frames = np.asarray(arrays["frames"], dtype=np.int64)
                features = np.asarray(arrays["features"], dtype=np.float64)
                actions = np.asarray(arrays["logits"], dtype=np.float64)
            if (
                features.shape != (len(frames), 48)
                or actions.shape != (len(frames), 12)
                or not np.isfinite(features).all()
                or not np.isfinite(actions).all()
            ):
                raise ValueError("finite aligned trajectory required")
            n = min(len(frames), 10)
            if not n:
                continue
            groups[item["course"]["seed"]].append(
                {
                    "success": bool(
                        item["safe"]
                        and item["controlled_reception"]
                        and not item["own_nonfoot_frames"]
                    ),
                    "first_frame": int(frames[0]),
                    "initial_features": features[0, :10],
                    "mean_logits": actions[:n].mean(axis=0),
                    "first_logits": actions[0],
                }
            )
    rows = []
    for seed, episodes in sorted(groups.items()):
        good = [episode for episode in episodes if episode["success"]]
        bad = [episode for episode in episodes if not episode["success"]]
        good_actions = np.asarray([episode["mean_logits"] for episode in good]).reshape(-1, 12)
        bad_actions = np.asarray([episode["mean_logits"] for episode in bad]).reshape(-1, 12)
        good_states = np.asarray([episode["initial_features"] for episode in good]).reshape(-1, 10)
        bad_states = np.asarray([episode["initial_features"] for episode in bad]).reshape(-1, 10)
        gap = (
            float(np.linalg.norm(good_actions.mean(axis=0) - bad_actions.mean(axis=0)))
            if len(good) and len(bad)
            else None
        )
        state_gap = (
            float(np.linalg.norm(good_states.mean(axis=0) - bad_states.mean(axis=0)))
            if len(good) and len(bad)
            else None
        )
        rows.append(
            {
                "seed": seed,
                "successes": len(good),
                "failures": len(bad),
                "success_mean_action": good_actions.mean(axis=0).tolist() if len(good) else None,
                "failure_mean_action": bad_actions.mean(axis=0).tolist() if len(bad) else None,
                "success_action_spread": _spread(good_actions),
                "failure_action_spread": _spread(bad_actions),
                "success_failure_action_gap": gap,
                "success_failure_initial_state_gap": state_gap,
                "first_post_action_frame_success": sorted(
                    {episode["first_frame"] for episode in good}
                ),
                "first_post_action_frame_failure": sorted(
                    {episode["first_frame"] for episode in bad}
                ),
            }
        )
    informative = [row for row in rows if row["successes"] and row["failures"]]
    source = "scripts/rsi_r1_post_contact_mode_diagnosis_v231.py"
    report = {
        "schema": SCHEMA,
        "v230_report_hash": v230["report_hash"],
        "v229_report_hash": v229["report_hash"],
        "source_hash": hash_bytes((root / source).read_bytes()),
        "partition": "SEALED_CONSUMED_ONLINE_CONTACT_TRAJECTORY_DIAGNOSIS_ONLY",
        "rows": rows,
        "informative_course_count": len(informative),
        "action_separation_ranks": sorted(
            (
                {
                    "seed": row["seed"],
                    "gap": row["success_failure_action_gap"],
                    "success_spread": row["success_action_spread"],
                    "failure_spread": row["failure_action_spread"],
                    "initial_state_gap": row["success_failure_initial_state_gap"],
                }
                for row in informative
            ),
            key=lambda row: row["gap"],
            reverse=True,
        ),
        "status": "DEVELOPMENT_CONTACT_MODE_DIAGNOSIS_ONLY",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    output.mkdir(parents=True)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if hash_bytes((root / source).read_bytes()) != report["source_hash"]:
        raise ValueError("source drift during contact-mode diagnosis")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--v229-dir", type=Path, required=True)
    parser.add_argument("--v230-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = diagnose(args.v229_dir, args.v230_report, args.output)
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "status",
                    "informative_course_count",
                    "action_separation_ranks",
                    "report_hash",
                )
            }
        )
    )


if __name__ == "__main__":
    main()
