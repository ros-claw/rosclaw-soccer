"""Re-run consumed v53 ball states under an opt-in team turn-rate cap."""

from __future__ import annotations

import argparse
import json
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_diverse_curriculum_collect import training_courses
from scripts.rsi_team_lateral_phase_orchestrate_v52 import _run_one


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("v54 development output already exists")
    protocol = json.loads(args.protocol.read_text())
    names = [item["name"] for item in protocol["arms"]]
    if (
        protocol.get("schema") != "rsi_team_diverse_curriculum_protocol_v38"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or protocol.get("maximum_yaw_rate_radps") != 0.40
        or protocol.get("curriculum")
        != {
            "coordinate_seed": 20261032,
            "identity_seed_base": 1700000,
            "batch_count": 1,
            "training_scene_count": 32,
            "ball_x_m": [3.25, 4.25],
            "ball_y_m": [-0.70, -0.35],
            "ball_vx_mps": [-0.90, -0.20],
        }
        or names
        != [
            "parent",
            "baseline",
            "lateral10",
            "gate22_cap05",
            "gate22_cap10",
            "gate32_cap10",
            "gate22_cap10_guard12",
        ]
        or len(training_courses(protocol)) != 32
    ):
        raise ValueError("invalid consumed-course team yaw development protocol")
    if shutil.disk_usage(args.output_root.parent).free < 3_000_000_000:
        raise ValueError("insufficient physical evidence disk space")
    root = Path(__file__).parents[1]
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_diverse_curriculum_collect.py"),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    args.output_root.mkdir(parents=True)
    outcomes = []
    with ThreadPoolExecutor(max_workers=7) as pool:
        jobs = {
            pool.submit(_run_one, args.protocol, args.asset_root, args.output_root, name): name
            for name in names
        }
        for job in as_completed(jobs):
            value = job.result()
            outcomes.append(value)
            print(
                f"{value['arm']} exit={value['exit_code']} hash={value['report_hash']}", flush=True
            )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during v54 development replay")
    result = {
        "schema": "rsi_team_yaw_stability_orchestration_report_v54",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": hash_bytes(args.protocol.read_bytes()),
        "source_hashes": sources,
        "physical_scene_count": 32,
        "planned_episode_count": 224,
        "job_results": sorted(outcomes, key=lambda value: names.index(value["arm"])),
        "all_jobs_complete": all(
            value["exit_code"] == 0 and value["report_hash"] for value in outcomes
        ),
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "orchestration_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        "RSI_TEAM_V54_ORCHESTRATION="
        + json.dumps(
            {"report_hash": result["report_hash"], "all_jobs_complete": result["all_jobs_complete"]}
        ),
        flush=True,
    )
    if not result["all_jobs_complete"]:
        raise ValueError("incomplete v54 development physics jobs")


if __name__ == "__main__":
    main()
