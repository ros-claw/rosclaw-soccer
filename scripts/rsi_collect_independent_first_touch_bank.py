"""Collect auditable SIM_ONLY one-G1/one-ball first-touch action outcomes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.independent_first_touch_bank import (
    FRAMES,
    post_contact_displacement,
)
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.taskspace_swing_probe import choose_swing_side
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SEEDS = (20260953, 20260954)
LANES = (0, 2, 4, 6, 8, 10, 12, 14)
ARMS = ("parent", "cap005", "cap015")


def collect_one(
    root: Path,
    runner: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    seed: int,
    lane: int,
    gpu: int,
    source_hash: str,
) -> dict[str, Any]:
    """Run three fully independent simulator processes in a fixed course."""
    parent_folder = root / f"seed{seed}-lane{lane}-parent"
    results: dict[str, Any] = {}
    for arm in ARMS:
        folder = root / f"seed{seed}-lane{lane}-{arm}"
        log_path = root / "logs" / f"seed{seed}-lane{lane}-{arm}.log"
        if not (folder / "report.json").is_file():
            if folder.exists():
                raise ValueError(f"incomplete output exists; inspect without deleting: {folder}")
            command = [
                sys.executable,
                str(runner),
                "--g1-usd",
                str(g1_usd),
                "--model-root",
                str(model_root),
                "--output-dir",
                str(folder),
                "--frames",
                str(FRAMES),
                "--env-count",
                "1",
                "--training-course-seed",
                str(seed),
                "--single-course-lane",
                str(lane),
                "--inference-threads",
                "1",
                "--record-body-trace",
                "--record-foot-geometry",
                "--torch-batch-plan-only",
                "--headless",
                "--device",
                "cuda:0",
            ]
            if arm != "parent":
                command += [
                    "--late-swing-policy",
                    str(actor),
                    "--parent-report",
                    str(parent_folder / "report.json"),
                    "--revalidate-swing-side",
                    "--late-swing-lateral-cap-m",
                    "0.05" if arm == "cap005" else "0.15",
                ]
            env = os.environ.copy()
            env.update(
                OMNI_KIT_ACCEPT_EULA="YES",
                CUDA_VISIBLE_DEVICES=str(gpu),
                PYTHONPATH=os.pathsep.join(
                    [str(runner.parent.parent / "src"), "/code/rosclaw/rosclaw_test/src"]
                ),
            )
            with log_path.open("w", encoding="utf-8") as log:
                completed = subprocess.run(
                    command,
                    cwd=runner.parent.parent,
                    env=env,
                    stdout=log,
                    stderr=subprocess.STDOUT,
                    check=False,
                )
            if completed.returncode or not (folder / "report.json").is_file():
                raise RuntimeError(f"Isaac run failed for {seed}/{lane}/{arm}; see {log_path}")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        audit = audit_vector_first_touch(folder)
        if (
            report.get("source_hash") != source_hash
            or report.get("training_course_seed") != seed
            or report.get("single_course_lane") != lane
            or len(report["environments"]) != 1
        ):
            raise ValueError(f"unbound source or independent course in {folder}")
        action_audit = None
        if arm != "parent":
            parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
            if (
                report.get("parent_report_hash") != parent["report_hash"]
                or report.get("asset_hash") != parent["asset_hash"]
                or report.get("sonic_qualification_hash") != parent["sonic_qualification_hash"]
                or report.get("course_catalog_hash") != parent["course_catalog_hash"]
                or report.get("taskspace_revalidate_swing_side") is not True
                or report.get("taskspace_lateral_cap_m") != (0.05 if arm == "cap005" else 0.15)
                or report.get("taskspace_probe_hash")
                != hash_bytes(Path(choose_swing_side.__code__.co_filename).read_bytes())
            ):
                raise ValueError(f"unpaired actor arm in {folder}")
            with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as replay:
                action_audit = audit_taskspace_swing_trace(replay, report, frames=FRAMES, count=1)
        entry = report["environments"][0]
        first = entry["first_contact_frame"]
        with np.load(folder / "trace.npz", allow_pickle=False) as physics:
            displacement = post_contact_displacement(physics["ball_position_m"][:, 0], first)
        bodies = entry["contact_body_indices"]
        results[arm] = {
            "report_hash": report["report_hash"],
            "audit_hash": audit["report_hash"],
            "course": entry["course"],
            "first_contact_frame": first,
            "contact_body_indices": bodies,
            "clean_foot_only": bool(bodies and set(bodies) <= {0, 1}),
            "minimum_pelvis_z_m": entry["minimum_pelvis_z_m"],
            "gate_selected": report.get("selected_taskspace_mask", [False])[0],
            "action_audit": action_audit,
            **displacement,
        }
    if any(results[arm]["course"] != results["parent"]["course"] for arm in ARMS):
        raise ValueError(f"course drift between arms: {seed}/{lane}")
    return {"seed": seed, "lane": lane, "gpu": gpu, "arms": results}


def collect_gpu_courses(
    gpu: int,
    courses: list[tuple[int, int]],
    root: Path,
    runner: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    source_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Own one GPU per worker; never overlap two Isaac processes on it."""
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed, lane in courses:
        try:
            episodes.append(
                collect_one(root, runner, g1_usd, model_root, actor, seed, lane, gpu, source_hash)
            )
            print(f"BANK_COURSE_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"BANK_COURSE_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return episodes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        not runner.is_file()
        or not args.g1_usd.is_file()
        or not args.model_root.is_dir()
        or not args.late_swing_policy.is_file()
        or (args.output_root.exists() and not args.resume)
        or (args.resume and not args.output_root.is_dir())
    ):
        parser.error("qualified SIM_ONLY assets and a new output root (or --resume) required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    source_hash = hash_bytes(runner.read_bytes())
    courses = [(seed, lane) for seed in SEEDS for lane in LANES]
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                collect_gpu_courses,
                gpu,
                courses[gpu::4],
                args.output_root,
                runner,
                args.g1_usd,
                args.model_root,
                args.late_swing_policy,
                source_hash,
            )
            for gpu in range(4)
        ]
        for future in as_completed(futures):
            gpu_episodes, gpu_failures = future.result()
            episodes.extend(gpu_episodes)
            failures.extend(gpu_failures)
    summary: dict[str, Any] = {
        "schema": "rsi_independent_first_touch_bank_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "actor_hash": hash_bytes(args.late_swing_policy.read_bytes()),
        "episodes": sorted(episodes, key=lambda row: (row["seed"], row["lane"])),
        "failures": failures,
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures or len(episodes) != len(courses):
        raise SystemExit(1)
    print(f"BANK_COMPLETE={summary['report_hash']}", flush=True)


if __name__ == "__main__":
    main()
