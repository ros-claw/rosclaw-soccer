"""Audit paired SIM_ONLY G1 lateral-contact-centering interventions."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.late_swing_memory import load_late_swing_actor
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.rsi.vector_first_touch_evidence import audit_vector_first_touch
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

COURSES = (
    (20260953, 0),
    (20260953, 12),
    (20260954, 2),
    (20260954, 6),
    (20260955, 0),
    (20260955, 2),
    (20260956, 0),
)
ARMS = {"lead000": 0.0, "leadn004": -0.04}


def collect_one(
    *,
    seed: int,
    lane: int,
    gpu: int,
    root: Path,
    old_parent_root: Path,
    new_parent_root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    source_hash: str,
) -> dict[str, Any]:
    parent_root = old_parent_root if seed <= 20260954 else new_parent_root
    parent_path = parent_root / f"seed{seed}-lane{lane}-parent/report.json"
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    results: dict[str, Any] = {}
    for arm, lead in ARMS.items():
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
                "0.95",
                "--late-swing-lateral-lead-m",
                str(lead),
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
            or report.get("late_swing_actor_hash") != load_late_swing_actor(actor)["actor_hash"]
            or report.get("taskspace_lateral_lead_m") != lead
            or report.get("taskspace_lateral_cap_m") != 0.15
            or report.get("late_swing_side_acquisition_gap_m") != 0.95
            or report.get("taskspace_revalidate_swing_side") is not True
            or report.get("training_course_seed") != seed
            or report.get("single_course_lane") != lane
        ):
            raise ValueError(f"unpaired lead arm: {folder}")
        with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as replay:
            action = audit_taskspace_swing_trace(replay, report, frames=300, count=1)
        entry = report["environments"][0]
        first = entry["first_contact_frame"]
        bodies = entry["contact_body_indices"]
        with np.load(folder / "trace.npz", allow_pickle=False) as physics:
            displacement = post_contact_displacement(physics["ball_position_m"][:, 0], first)
            if first is not None and bodies and set(bodies) <= {0, 1}:
                force = physics["ball_body_contact_force_peak_n"][first, 0]
                foot = 0 if force[0] >= force[1] else 1
            else:
                foot = None
        with np.load(folder / "body_trace.npz", allow_pickle=False) as body:
            contact_y = (
                float(
                    body["ball_position_before_step_m"][first, 0, 1]
                    - body["foot_geometry_position_before_step_m"][first, 0, foot, 1]
                )
                if first is not None and foot is not None
                else None
            )
        results[arm] = {
            "report_hash": report["report_hash"],
            "physical_audit_hash": physical["report_hash"],
            "action_audit": action,
            "course": entry["course"],
            "first_contact_frame": first,
            "contact_body_indices": bodies,
            "clean_foot_only": bool(bodies and set(bodies) <= {0, 1}),
            "contact_ball_minus_foot_y_m": contact_y,
            "maximum_lateral_excursion_m": report["single_instance_max_lateral_excursion_m"],
            "minimum_pelvis_z_m": entry["minimum_pelvis_z_m"],
            **displacement,
        }
    if results["lead000"]["course"] != results["leadn004"]["course"]:
        raise ValueError(f"course drift: {seed}/{lane}")
    return {"seed": seed, "lane": lane, "gpu": gpu, "arms": results}


def collect_gpu(
    gpu: int, courses: list[tuple[int, int]], **kwargs: Any
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    episodes: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    for seed, lane in courses:
        try:
            episodes.append(collect_one(seed=seed, lane=lane, gpu=gpu, **kwargs))
            print(f"LEAD_PAIR_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"LEAD_PAIR_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return episodes, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--old-parent-root", required=True, type=Path)
    parser.add_argument("--new-parent-root", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
        or not all(
            (
                (args.old_parent_root if seed <= 20260954 else args.new_parent_root)
                / f"seed{seed}-lane{lane}-parent/report.json"
            ).is_file()
            for seed, lane in COURSES
        )
        or (args.output_root.exists() and not args.resume)
        or (args.resume and not args.output_root.is_dir())
    ):
        parser.error("qualified assets, parents, and new/resumable output required")
    args.output_root.mkdir(parents=True, exist_ok=True)
    (args.output_root / "logs").mkdir(exist_ok=True)
    source_hash = hash_bytes(runner.read_bytes())
    common = dict(
        root=args.output_root,
        old_parent_root=args.old_parent_root,
        new_parent_root=args.new_parent_root,
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
        futures = [
            pool.submit(collect_gpu, gpu, list(COURSES[gpu::4]), **common) for gpu in range(4)
        ]
        for future in as_completed(futures):
            found, failed = future.result()
            episodes.extend(found)
            failures.extend(failed)
    ordered = sorted(episodes, key=lambda row: (row["seed"], row["lane"]))
    improved = sum(
        all(
            a[key] is not None and c[key] is not None and abs(c[key]) < abs(a[key])
            for key in ("contact_ball_minus_foot_y_m", "lateral_60_m")
        )
        for row in ordered
        for a, c in [(row["arms"]["lead000"], row["arms"]["leadn004"])]
    )
    out_base = sum(row["arms"]["lead000"]["maximum_lateral_excursion_m"] > 4 for row in ordered)
    out_candidate = sum(
        row["arms"]["leadn004"]["maximum_lateral_excursion_m"] > 4 for row in ordered
    )
    passed = bool(
        not failures
        and len(ordered) == len(COURSES)
        and improved >= 4
        and out_candidate < out_base
        and all(
            row["arms"]["leadn004"]["clean_foot_only"] >= row["arms"]["lead000"]["clean_foot_only"]
            and row["arms"]["leadn004"]["minimum_pelvis_z_m"] >= 0.65
            for row in ordered
        )
    )
    summary: dict[str, Any] = {
        "schema": "rsi_contact_centering_lead_bank_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "episodes": ordered,
        "failures": failures,
        "joint_contact_and_flight_improvements": improved,
        "out_of_play_baseline": out_base,
        "out_of_play_candidate": out_candidate,
        "development_gate_passed": passed,
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    if failures or len(ordered) != len(COURSES):
        raise SystemExit(1)
    print(f"LEAD_BANK_COMPLETE={summary['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={passed}", flush=True)


if __name__ == "__main__":
    main()
