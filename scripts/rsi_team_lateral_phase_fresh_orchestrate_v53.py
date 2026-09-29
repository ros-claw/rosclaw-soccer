"""Run the frozen v53 independent physical exam without selecting a new arm."""

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
        parser.error("v53 fresh exam output already exists")
    protocol = json.loads(args.protocol.read_text())
    names = [item["name"] for item in protocol["arms"]]
    if (
        protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
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
        or protocol.get("fresh_gate")
        != {
            "primary_arm": "gate22_cap10",
            "minimum_useful_gain_over_baseline": 3,
            "minimum_safe_contact_gain_over_baseline": 5,
            "maximum_unsafe_excess_over_baseline": 0,
            "require_primary_and_parent_complete": True,
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
        raise ValueError("invalid frozen v53 fresh physical exam")
    if shutil.disk_usage(args.output_root.parent).free < 3_000_000_000:
        raise ValueError("insufficient physical exam disk space")
    root = Path(__file__).parents[1]
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_lateral_phase_orchestrate_v52.py"),
        Path(__file__).with_name("rsi_team_diverse_curriculum_collect.py"),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
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
        raise ValueError("fresh exam source changed during physical collection")
    result = {
        "schema": "rsi_team_lateral_phase_fresh_orchestration_report_v53",
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
        "RSI_TEAM_V53_ORCHESTRATION="
        + json.dumps(
            {"report_hash": result["report_hash"], "all_jobs_complete": result["all_jobs_complete"]}
        ),
        flush=True,
    )
    if not result["all_jobs_complete"]:
        raise ValueError("incomplete v53 physical exam jobs")


if __name__ == "__main__":
    main()
