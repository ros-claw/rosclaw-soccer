"""Run a predeclared FRESH paired physical exam for a frozen selector and SONIC parent."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rsi_sonic_ball_contact_probe import run

from rosclaw_soccer.rsi.sonic_contact_selector import choose_lateral, load_selector
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _course(
    *,
    model_root: Path,
    stadium_assets: Path,
    output_root: Path,
    x: float,
    y: float,
    lateral: float,
    candidate: bool,
    selector_hash: str,
) -> dict[str, Any]:
    mode = "candidate" if candidate else "parent"
    label = f"x{int(round(x * 1000)):04d}-y{int(round(y * 1000)):04d}-{mode}"
    report: dict[str, Any] = run(
        model_root=model_root,
        stadium_assets=stadium_assets,
        output_dir=output_root / label,
        ball_x_m=x,
        ball_y_m=y,
        frames=300,
        run_speed_mps=1.4,
        run_lateral_mps=lateral if candidate else 0.0,
        stop_frame=120,
        left_hip_residual_rad=-0.15 if candidate else 0.0,
        left_knee_residual_rad=0.15 if candidate else 0.0,
        partition="FRESH",
        selector_hash=selector_hash if candidate else None,
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--selector", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--ball-x-m", type=float, choices=(1.8, 1.9, 2.0, 2.1), default=1.8)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists() or not 1 <= args.workers <= 4:
        raise ValueError("new exam output and 1–4 workers required")
    selector = load_selector(args.selector)
    selector_file_hash = hash_bytes(args.selector.read_bytes())
    runner_hash = hash_bytes(Path(__file__).read_bytes())
    probe_file = Path(__file__).with_name("rsi_sonic_ball_contact_probe.py")
    probe_hash = hash_bytes(probe_file.read_bytes())
    y_values = (0.05, 0.09, 0.13, 0.17) if args.ball_x_m == 2.1 else (0.04, 0.08, 0.12, 0.16)
    courses = [(args.ball_x_m, y) for y in y_values]
    tasks = [
        (x, y, choose_lateral(selector, ball_x_m=x, ball_y_m=y), candidate)
        for x, y in courses
        for candidate in (False, True)
    ]
    args.output_root.mkdir(parents=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = [
            executor.submit(
                _course,
                model_root=args.model_root,
                stadium_assets=args.stadium_assets,
                output_root=args.output_root,
                x=x,
                y=y,
                lateral=lateral,
                candidate=candidate,
                selector_hash=selector["model_hash"],
            )
            for x, y, lateral, candidate in tasks
        ]
        for future in as_completed(futures):
            report = future.result()
            reports.append(report)
            print(
                json.dumps(
                    {
                        "ball": report["ball_initial_xy_m"],
                        "candidate": report["selector_hash"] is not None,
                        "lateral": report["run_lateral_mps"],
                        "foot_goal": report["first_robot_ball_contact_is_foot"]
                        and report["whole_ball_goal_crossed"],
                        "peak_ball_speed_mps": report["peak_ball_speed_mps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if (
        hash_bytes(args.selector.read_bytes()) != selector_file_hash
        or hash_bytes(Path(__file__).read_bytes()) != runner_hash
        or hash_bytes(probe_file.read_bytes()) != probe_hash
    ):
        raise RuntimeError("frozen selector or physical runner changed during fresh exam")
    parent = [report for report in reports if report["selector_hash"] is None]
    candidate = [report for report in reports if report["selector_hash"] is not None]

    def foot_goals(rows: list[dict[str, Any]]) -> int:
        return sum(
            bool(row["first_robot_ball_contact_is_foot"] and row["whole_ball_goal_crossed"])
            for row in rows
        )

    manifest = {
        "schema": "rosclaw_soccer.rsi.sonic_contact_selector_holdout.v1",
        "partition": "FRESH",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "selector_hash": selector["model_hash"],
        "selector_file_hash": selector_file_hash,
        "runner_hash": runner_hash,
        "probe_hash": probe_hash,
        "courses": [[x, y] for x, y in courses],
        "physical_execution_count": len(reports),
        "parent_foot_goals": foot_goals(parent),
        "candidate_foot_goals": foot_goals(candidate),
        "report_hashes": sorted(report["report_hash"] for report in reports),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    with (args.output_root / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
