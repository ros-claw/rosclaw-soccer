"""Collect independent frozen-SONIC parent reports for new training seeds."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

LANES = tuple(range(0, 16, 2))


def collect_gpu(
    gpu: int,
    courses: list[tuple[int, int]],
    *,
    root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    source_hash: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed, lane in courses:
        folder = root / f"seed{seed}-lane{lane}-parent"
        log_path = root / "logs" / f"seed{seed}-lane{lane}-parent.log"
        try:
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
                    "300",
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
                    raise RuntimeError(f"Isaac failed; see {log_path}")
            report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
            audit = audit_vector_first_touch(folder)
            if (
                report.get("source_hash") != source_hash
                or report.get("training_course_seed") != seed
                or report.get("single_course_lane") != lane
                or report.get("asset_hash") != hash_bytes(g1_usd.read_bytes())
                or report.get("late_swing_actor_hash") is not None
            ):
                raise ValueError(f"unbound frozen parent: {folder}")
            episodes.append(
                {
                    "seed": seed,
                    "lane": lane,
                    "gpu": gpu,
                    "report_hash": report["report_hash"],
                    "physical_audit_hash": audit["report_hash"],
                    "course": report["environments"][0]["course"],
                }
            )
            print(f"PARENT_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"PARENT_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return episodes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--seeds", required=True, nargs="+", type=int)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    seeds = tuple(args.seeds)
    courses = [(seed, lane) for seed in seeds for lane in LANES]
    if (
        seeds
        not in (
            (20260953,),
            (20260955, 20260956),
            (20260957,),
            tuple(range(20260958, 20260966)),
        )
        or not runner.is_file()
        or not args.isaac_python.is_file()
        or not args.g1_usd.is_file()
        or not args.model_root.is_dir()
        or (args.output_root.exists() and not args.resume)
        or (args.resume and not args.output_root.is_dir())
    ):
        parser.error("v279 requires predeclared seeds, qualified assets and new/resumable output")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    source_hash = hash_bytes(runner.read_bytes())
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [
            pool.submit(
                collect_gpu,
                gpu,
                courses[gpu::4],
                root=args.output_root,
                runner=runner,
                isaac_python=args.isaac_python,
                g1_usd=args.g1_usd,
                model_root=args.model_root,
                source_hash=source_hash,
            )
            for gpu in range(4)
        ]
        for future in as_completed(futures):
            found, failed = future.result()
            episodes.extend(found)
            failures.extend(failed)
    summary: dict[str, Any] = {
        "schema": "rsi_independent_first_touch_parents_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "episodes": sorted(episodes, key=lambda row: (row["seed"], row["lane"])),
        "failures": failures,
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_root / "parent_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures or len(episodes) != len(courses):
        raise SystemExit(1)
    print(f"PARENTS_COMPLETE={summary['report_hash']}", flush=True)


if __name__ == "__main__":
    main()
