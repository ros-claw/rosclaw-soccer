"""SIM_ONLY test successful coherent latent teachers across all high-zone courses."""

from __future__ import annotations

import argparse
import json
import multiprocessing
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_coherent_exploration_v198 import _checked
from rsi_r1_contact_manifold_v222 import context
from rsi_r1_latent_contact_exploration_v232 import _run_task
from rsi_r1_measured_skill_router_v207 import _score, clean
from rsi_r1_protected_kinematic_fresh_v202 import _load_policy

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_coherent_teacher_transfer_v233.result.v1"


def transfer(
    asset_root: Path,
    policy_path: Path,
    parent_report: Path,
    right_report: Path,
    refine_report: Path,
    lateral_report: Path,
    v205_dir: Path,
    map_report_path: Path,
    v214_report_path: Path,
    v215_dir: Path,
    v216_report_path: Path,
    v218_dir: Path,
    v220_report_path: Path,
    v223_report_path: Path,
    v224_report_path: Path,
    v229_dir: Path,
    v232_dir: Path,
    output: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY coherent transfer evidence required")
    v223, v224, v229, v232 = (
        _checked(path)
        for path in (
            v223_report_path,
            v224_report_path,
            v229_dir / "report.json",
            v232_dir / "report.json",
        )
    )
    if (
        v232["status"] != "DEVELOPMENT_COHERENT_LATENT_EXPLORATION_ONLY"
        or v232["v229_report_hash"] != v229["report_hash"]
        or v232["sample_score"][:2] != [35, 136]
        or v224["v223_report_hash"] != v223["report_hash"]
    ):
        raise ValueError("sealed coherent success trajectories required")
    common, (base_weights, _, _), lineage = context(
        asset_root,
        policy_path,
        parent_report,
        right_report,
        refine_report,
        lateral_report,
        v205_dir,
        map_report_path,
        v214_report_path,
        v215_dir,
        v216_report_path,
        v218_dir,
        v220_report_path,
    )
    if lineage["v220_report_hash"] != v224["v220_report_hash"]:
        raise ValueError("sealed precontact motor lineage required")
    pre_weights = replace(
        base_weights,
        output_bias=tuple(
            float(value)
            for value in np.asarray(base_weights.output_bias) + np.asarray(v224["best_offset"])
        ),
    )
    post_weights = _load_policy(v229_dir / "update-1.npz", v229["history"][0]["checkpoint_hash"])
    high_courses = tuple(ReceivingCourse(**row["course"]) for row in v232["course_summary"])
    if len(high_courses) != 11:
        raise ValueError("eleven high-zone contact courses required")
    groups: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in v232["episode_records"]:
        if not (
            record["safe"] and record["controlled_reception"] and not record["own_nonfoot_frames"]
        ):
            continue
        path = v232_dir / f"episode-{record['index']}.npz"
        if hash_bytes(path.read_bytes()) != record["trajectory_hash"]:
            raise ValueError("sealed successful coherent physical trajectory required")
        groups[record["course"]["seed"]].append(record)
    selected = []
    for _seed, records in sorted(groups.items()):
        records.sort(
            key=lambda row: (
                min(0.35 - row["distance_m"], 0.35 - row["speed_mps"])
                - 0.02 * float(np.linalg.norm(row["latent_offset"])),
                -row["index"],
            ),
            reverse=True,
        )
        selected.extend(records[:2])
    if not 10 <= len(selected) <= 14:
        raise ValueError("nontrivial successful coherent teacher shortlist required")

    def tasks(weights: Any) -> list[tuple[Any, ...]]:
        return [
            (common[0], common[1], course, *common[2:], pre_weights, weights, 0, 0.0)
            for course in high_courses
        ]

    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_coherent_teacher_transfer_v233.py",
            "scripts/rsi_r1_latent_contact_exploration_v232.py",
            "src/rosclaw_soccer/rsi/receiving_latent_phase_motor.py",
            "src/rosclaw_soccer/training/receiving_experiment.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    with ProcessPoolExecutor(
        max_workers=8, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        baseline = list(pool.map(_run_task, tasks(post_weights)))
        old = {
            row["course"]["seed"]: row
            for row in v229["history"][0]["deterministic_rows"]
            if row["course"]["seed"] in {course.seed for course in high_courses}
        }
        if any(
            row["physical_trace_hash"] != old[row["course"]["seed"]]["physical_trace_hash"]
            for row in baseline
        ):
            raise ValueError("coherent teacher parent must replay zero-latent evidence")
        candidates = []
        for record in selected:
            candidate_weights = replace(
                post_weights,
                output_bias=tuple(
                    float(value)
                    for value in np.asarray(post_weights.output_bias)
                    + np.asarray(record["latent_offset"])
                ),
            )
            rows = list(pool.map(_run_task, tasks(candidate_weights)))
            score = _score(rows)
            retained = all(
                not (clean(old) and old["controlled_reception"])
                or (clean(new) and new["controlled_reception"])
                for old, new in zip(baseline, rows, strict=True)
            )
            candidates.append(
                {
                    "source_episode": record["index"],
                    "source_course": record["course"],
                    "source_trajectory_hash": record["trajectory_hash"],
                    "latent_offset": record["latent_offset"],
                    "score": score,
                    "old_success_retained": retained,
                    "rows": rows,
                }
            )
            print(
                json.dumps(
                    {
                        "source_episode": record["index"],
                        "source_seed": record["course"]["seed"],
                        "score": score,
                        "retained": retained,
                    }
                ),
                flush=True,
            )
    qualified_candidates = [
        row
        for row in candidates
        if row["old_success_retained"]
        and row["score"][0] >= _score(baseline)[0] + 2
        and row["score"][1] >= _score(baseline)[1]
    ]
    best = max(qualified_candidates or candidates, key=lambda row: tuple(row["score"]))
    qualified = bool(qualified_candidates)
    report = {
        "schema": SCHEMA,
        "v232_report_hash": v232["report_hash"],
        "source_hashes": sources,
        "partition": "CONSUMED_SUCCESSFUL_COHERENT_TEACHERS_CROSS_COURSE_TRANSFER_ONLY",
        "shortlist": [row["index"] for row in selected],
        "baseline": baseline,
        "baseline_score": _score(baseline),
        "candidates": candidates,
        "best_source_episode": best["source_episode"],
        "best_score": best["score"],
        "status": "DEVELOPMENT_COHERENT_TRANSFER_CANDIDATE_ONLY"
        if qualified
        else "REJECTED_COHERENT_TRANSFER_GATE",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during coherent teacher transfer")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in (
        "asset-root",
        "policy",
        "parent-report",
        "right-report",
        "refine-report",
        "lateral-report",
        "v205-dir",
        "map-report",
        "v214-report",
        "v215-dir",
        "v216-report",
        "v218-dir",
        "v220-report",
        "v223-report",
        "v224-report",
        "v229-dir",
        "v232-dir",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    report = transfer(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.right_report,
        args.refine_report,
        args.lateral_report,
        args.v205_dir,
        args.map_report,
        args.v214_report,
        args.v215_dir,
        args.v216_report,
        args.v218_dir,
        args.v220_report,
        args.v223_report,
        args.v224_report,
        args.v229_dir,
        args.v232_dir,
        args.output,
    )
    print(
        json.dumps(
            {key: report[key] for key in ("status", "baseline_score", "best_score", "report_hash")}
        )
    )


if __name__ == "__main__":
    main()
