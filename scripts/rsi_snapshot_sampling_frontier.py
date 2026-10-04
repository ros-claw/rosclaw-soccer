"""Snapshot completed consumed-training records without touching active workers."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.sampling_frontier_diagnostics import contact_timeline, sampling_frontier
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once


def snapshot(root: Path, output: Path) -> dict[str, Any]:
    declaration = _sealed(root / "commitment.json")
    if (
        declaration.get("schema") != "soccer.rsi.current_proposal_collection.v1"
        or declaration.get("private_fresh_accessed") is not False
        or declaration.get("hardware_authorized") is not False
    ):
        raise ValueError("declared consumed SIM-only collection required")
    jobs = {job["group"]: job for job in declaration["jobs"]}
    if len(jobs) != 160 or set(jobs) != set(range(160)):
        raise ValueError("complete predeclared 160-episode collection required")
    rows, pins = [], {}
    timelines = []
    for path in sorted(root.glob("row-*.json")):
        data = path.read_bytes()
        row = json.loads(data)
        group = row.get("group")
        if type(group) is not int or group not in jobs:
            raise ValueError("row does not belong to declared collection")
        job = jobs[group]
        if any(
            row.get(k) != job.get(k)
            for k in (
                "group",
                "seed",
                "lane",
                "sampling_seed",
                "baseline_course_index",
                "sample_index",
                "view_hash",
            )
        ):
            raise ValueError("row differs from original declared job")
        if (
            row.get("execution_origin") != "NEW"
            or row.get("actual_behavior_model_hash") != declaration.get("behavior_model_hash")
            or row.get("behavior_generation") != declaration.get("behavior_generation")
        ):
            raise ValueError("actual declared new behavior required")
        folder = Path(row["folder"])
        if folder.resolve() != (root / f"sample-{group}").resolve():
            raise ValueError("row physical evidence outside declared collection")
        review_path = folder / "review.json"
        review = _sealed(review_path)
        if (
            review["report_hash"] != row["review_hash"]
            or {k: v for k, v in row["outcome"].items() if k != "reward"} != review
            or review.get("physical_substeps") != 3000
            or any(
                review.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")
            )
            or any(
                review.get(k) is not True
                for k in (
                    "actual_mujoco_dynamics_replayed",
                    "actual_pd_torque_reconstructed",
                    "neural_target_reconstructed",
                )
            )
        ):
            raise ValueError("complete independent source review required")
        frames = Path(row["learning_frames_path"])
        if frames.resolve() != (root / f"learning-frames-{group}.npz").resolve():
            raise ValueError("learning data outside declared collection")
        if hash_bytes(frames.read_bytes()) != row["learning_frames_hash"]:
            raise ValueError("retained learning frames changed")
        foundation_path = root / f"foundation-review-{group}.json"
        foundation = _sealed(foundation_path)
        if (
            foundation["report_hash"] != row.get("executed_torch_review_hash")
            or foundation.get("recomputed_executed_calls") != 300
            or foundation.get("physical_report_hash") != row.get("report_hash")
            or any(
                foundation.get(k) is not False
                for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("all original foundation calls must be reviewed")
        trace_path = folder / "physical_trace.npz"
        trace_hash = hash_bytes(trace_path.read_bytes())
        if trace_hash != foundation.get("physical_trace_hash"):
            raise ValueError("original physical trace changed")
        with np.load(trace_path, allow_pickle=False) as trace:
            forces = trace["force_n"]
            if forces.shape != (300, 1, 6):
                raise ValueError("original single-lane force trace required")
            timeline = contact_timeline(forces[:, 0], row["outcome"])
        timelines.append(dict(group=group, **timeline))
        pins[str(path)] = hash_bytes(data)
        pins[str(review_path)] = hash_bytes(review_path.read_bytes())
        pins[str(frames)] = row["learning_frames_hash"]
        pins[str(foundation_path)] = hash_bytes(foundation_path.read_bytes())
        pins[str(trace_path)] = trace_hash
        rows.append(row)
    result: dict[str, Any] = sampling_frontier(rows, expected_episodes=len(jobs))
    result.update(
        source_commitment_hash=declaration["report_hash"],
        source_records_hash=hash_json(rows),
        source_file_hashes=pins,
        source_records=rows,
        contact_timelines=timelines,
        contact_timeline_counts=dict(sorted(Counter(t["kind"] for t in timelines).items())),
        sealed_original_reviews_checked=True,
        original_learning_bytes_checked=True,
        new_physical_executions=0,
        runner_source_hash=hash_bytes(Path(__file__).read_bytes()),
    )
    if any(hash_bytes(Path(p).read_bytes()) != h for p, h in pins.items()):
        raise ValueError("completed source evidence changed during snapshot")
    if _sealed(root / "commitment.json") != declaration:
        raise ValueError("collection declaration changed during snapshot")
    result["report_hash"] = hash_json(result)
    write_once(output, result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = snapshot(args.collection.resolve(), args.output)
    print(
        json.dumps(
            {
                k: v
                for k, v in result.items()
                if k
                not in ("source_records", "source_file_hashes", "contexts", "contact_timelines")
            }
        )
    )


if __name__ == "__main__":
    main()
