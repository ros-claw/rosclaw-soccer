"""Collect and audit paired independent Isaac swing-acquisition outcomes."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.independent_first_touch_bank import FRAMES, post_contact_displacement
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

LANES = tuple(range(0, 16, 2))
ARMS = {"acquisition_055": 0.55, "acquisition_095": 0.95}


def collect_pair(
    root: Path,
    parent_root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    seed: int,
    lane: int,
    gpu: int,
    source_hash: str,
) -> dict[str, Any]:
    """Run two distinct simulator processes and audit both before comparing."""
    parent_path = parent_root / f"seed{seed}-lane{lane}-parent" / "report.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    features: np.ndarray[Any, Any] | None = None
    for arm, gap in ARMS.items():
        folder = root / f"seed{seed}-lane{lane}-{arm}"
        log_path = root / "logs" / f"seed{seed}-lane{lane}-{arm}.log"
        if not (folder / "report.json").is_file():
            if folder.exists():
                raise ValueError(f"incomplete output; inspect before resuming: {folder}")
            command = [
                str(isaac_python),
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
                "--late-swing-policy",
                str(actor),
                "--parent-report",
                str(parent_path),
                "--revalidate-swing-side",
                "--late-swing-lateral-cap-m",
                "0.15",
                "--late-swing-forward-cap-m",
                "0.08",
                "--late-swing-side-acquisition-gap-m",
                str(gap),
                "--headless",
                "--device",
                "cuda:0",
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
                raise RuntimeError(f"Isaac failed for {seed}/{lane}/{arm}; see {log_path}")
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        physical = audit_vector_first_touch(folder)
        if (
            report.get("source_hash") != source_hash
            or report.get("asset_hash") != parent.get("asset_hash")
            or report.get("sonic_qualification_hash") != parent.get("sonic_qualification_hash")
            or report.get("course_catalog_hash") != parent.get("course_catalog_hash")
            or report.get("parent_report_hash") != parent.get("report_hash")
            or report.get("late_swing_side_acquisition_gap_m") != gap
            or report.get("taskspace_lateral_cap_m") != 0.15
            or report.get("taskspace_forward_m") != 0.08
            or report.get("taskspace_revalidate_swing_side") is not True
            or report.get("training_course_seed") != seed
            or report.get("single_course_lane") != lane
            or report.get("late_swing_actor_hash") != load_late_swing_actor(actor)["actor_hash"]
        ):
            raise ValueError(f"unpaired physical arm: {folder}")
        with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as replay:
            action = audit_taskspace_swing_trace(replay, report, frames=FRAMES, count=1)
            current_features = np.asarray(replay["frame30_gate_features"][0], dtype=float)
            if features is None:
                features = current_features
            elif not np.allclose(features, current_features, atol=1e-10, rtol=0):
                raise ValueError(f"counterfactual precontact features drifted: {seed}/{lane}")
        entry = report["environments"][0]
        with np.load(folder / "trace.npz", allow_pickle=False) as physics:
            displacement = post_contact_displacement(
                physics["ball_position_m"][:, 0], entry["first_contact_frame"]
            )
        bodies = entry["contact_body_indices"]
        results[arm] = {
            "report_hash": report["report_hash"],
            "physical_audit_hash": physical["report_hash"],
            "action_audit": action,
            "course": entry["course"],
            "first_contact_frame": entry["first_contact_frame"],
            "contact_body_indices": bodies,
            "clean_foot_only": bool(bodies and set(bodies) <= {0, 1}),
            "minimum_pelvis_z_m": entry["minimum_pelvis_z_m"],
            "maximum_lateral_excursion_m": report["single_instance_max_lateral_excursion_m"],
            **displacement,
        }
    if results["acquisition_055"]["course"] != results["acquisition_095"]["course"]:
        raise ValueError(f"paired courses differ: {seed}/{lane}")
    return {"seed": seed, "lane": lane, "gpu": gpu, "arms": results}


def collect_gpu(
    gpu: int, courses: list[tuple[int, int]], **kwargs: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed, lane in courses:
        try:
            episodes.append(collect_pair(seed=seed, lane=lane, gpu=gpu, **kwargs))
            print(f"EARLY_BANK_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"EARLY_BANK_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return episodes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--parent-root", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    parser.add_argument("--seeds", nargs="+", type=int, default=[20260953, 20260954])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    seeds = tuple(args.seeds)
    courses = [(seed, lane) for seed in seeds for lane in LANES]
    if (
        seeds not in ((20260953, 20260954), (20260955, 20260956), (20260957,))
        or not runner.is_file()
        or not args.isaac_python.is_file()
        or not args.g1_usd.is_file()
        or not args.model_root.is_dir()
        or not args.late_swing_policy.is_file()
        or not all(
            (args.parent_root / f"seed{s}-lane{lane}-parent/report.json").is_file()
            for s, lane in courses
        )
        or (args.output_root.exists() and not args.resume)
        or (args.resume and not args.output_root.is_dir())
    ):
        parser.error("qualified assets, complete parent bank, and new/resumable output required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    source_hash = hash_bytes(runner.read_bytes())
    common = dict(
        root=args.output_root,
        parent_root=args.parent_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        source_hash=source_hash,
    )
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(collect_gpu, gpu, courses[gpu::4], **common) for gpu in range(4)]
        for future in as_completed(futures):
            gpu_episodes, gpu_failures = future.result()
            episodes.extend(gpu_episodes)
            failures.extend(gpu_failures)
    summary: dict[str, Any] = {
        "schema": "rsi_independent_early_acquisition_bank_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "actor_hash": load_late_swing_actor(args.late_swing_policy)["actor_hash"],
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
    print(f"EARLY_BANK_COMPLETE={summary['report_hash']}", flush=True)


if __name__ == "__main__":
    main()
