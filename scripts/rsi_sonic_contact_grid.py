"""Generate a pinned SIM_ONLY discovery grid for conditional SONIC approach learning.

Each course runs real MuJoCo integration. This is data collection, not a policy
promotion, and no test course is sampled or replaced after seeing outcomes.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rsi_sonic_ball_contact_probe import run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _course(
    *,
    model_root: Path,
    stadium_assets: Path,
    output_root: Path,
    x: float,
    y: float,
    lateral: float,
    residual: bool,
) -> dict[str, Any]:
    mode = "res" if residual else "parent"
    label = (
        f"x{int(round(1000 * x)):04d}-y{int(round(1000 * y)):04d}"
        f"-lat{int(round(1000 * (lateral + 0.3))):04d}-{mode}"
    )
    return run(
        model_root=model_root,
        stadium_assets=stadium_assets,
        output_dir=output_root / label,
        ball_x_m=x,
        ball_y_m=y,
        frames=300,
        run_speed_mps=1.4,
        run_lateral_mps=lateral,
        stop_frame=120,
        left_hip_residual_rad=-0.15 if residual else 0.0,
        left_knee_residual_rad=0.15 if residual else 0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--stadium-assets", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.output_root.exists() or not 1 <= args.workers <= 4:
        raise ValueError("new output root and 1–4 workers required")
    runner_hash = hash_bytes(Path(__file__).read_bytes())
    probe_file = Path(__file__).with_name("rsi_sonic_ball_contact_probe.py")
    probe_hash = hash_bytes(probe_file.read_bytes())
    x_values = (1.40, 1.50, 1.60)
    y_values = (0.025, 0.100, 0.175)
    lateral_values = (-0.10, 0.00, 0.10)
    courses = [
        (x, y, lateral, True) for x in x_values for y in y_values for lateral in lateral_values
    ] + [(x, y, 0.0, False) for x in x_values for y in y_values]
    args.output_root.mkdir(parents=True)
    reports = []
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {
            executor.submit(
                _course,
                model_root=args.model_root,
                stadium_assets=args.stadium_assets,
                output_root=args.output_root,
                x=x,
                y=y,
                lateral=lateral,
                residual=residual,
            ): (x, y, lateral, residual)
            for x, y, lateral, residual in courses
        }
        for future in as_completed(futures):
            result = future.result()
            reports.append(result)
            print(
                json.dumps(
                    {
                        "ball": result["ball_initial_xy_m"],
                        "lateral": result["run_lateral_mps"],
                        "residual": result["left_contact_residual_rad"],
                        "foot_first": result["first_robot_ball_contact_is_foot"],
                        "goal": result["whole_ball_goal_crossed"],
                        "peak_speed_mps": result["peak_ball_speed_mps"],
                    },
                    sort_keys=True,
                ),
                flush=True,
            )
    if (
        hash_bytes(Path(__file__).read_bytes()) != runner_hash
        or hash_bytes(probe_file.read_bytes()) != probe_hash
    ):
        raise RuntimeError("grid or physics runner source changed during execution")
    manifest = {
        "schema": "rosclaw_soccer.rsi.sonic_contact_grid.v1",
        "partition": "DISCOVERY",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "runner_hash": runner_hash,
        "probe_hash": probe_hash,
        "courses": [[x, y, lateral, residual] for x, y, lateral, residual in courses],
        "report_hashes": sorted(report["report_hash"] for report in reports),
        "course_count": len(courses),
        "goal_count": sum(bool(report["whole_ball_goal_crossed"]) for report in reports),
        "foot_first_goal_count": sum(
            bool(report["whole_ball_goal_crossed"] and report["first_robot_ball_contact_is_foot"])
            for report in reports
        ),
    }
    manifest["manifest_hash"] = hash_json(manifest)
    with (args.output_root / "manifest.json").open("x", encoding="utf-8") as stream:
        json.dump(manifest, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(json.dumps(manifest, sort_keys=True))


if __name__ == "__main__":
    main()
