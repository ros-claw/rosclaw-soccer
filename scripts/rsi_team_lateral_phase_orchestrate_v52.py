"""Run the seven frozen v52 hard-lateral G1 arms in independent processes."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_team_diverse_curriculum_collect import training_courses


def _run_one(protocol: Path, asset_root: Path, output_root: Path, arm: str) -> dict[str, Any]:
    folder = output_root / arm
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    result = subprocess.run(
        [
            sys.executable,
            str(Path(__file__).with_name("rsi_team_diverse_curriculum_collect.py")),
            "--asset-root",
            str(asset_root),
            "--protocol",
            str(protocol),
            "--output-dir",
            str(folder),
            "--arm",
            arm,
            "--batch-index",
            "0",
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    path = folder / "report.json"
    return {
        "arm": arm,
        "exit_code": result.returncode,
        "report_hash": json.loads(path.read_text()).get("report_hash") if path.is_file() else None,
        "last_output_lines": result.stdout.splitlines()[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    args = parser.parse_args()
    if args.output_root.exists():
        parser.error("v52 output already exists")
    protocol = json.loads(args.protocol.read_text())
    names = [item["name"] for item in protocol["arms"]]
    course = protocol["curriculum"]
    if (
        protocol.get("schema") != "rsi_team_diverse_curriculum_protocol_v38"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or course
        != {
            "coordinate_seed": 20261031,
            "identity_seed_base": 1600000,
            "batch_count": 1,
            "training_scene_count": 32,
            "ball_x_m": [3.25, 4.25],
            "ball_y_m": [-0.52, -0.35],
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
        raise ValueError("invalid frozen v52 physical challenge")
    if shutil.disk_usage(args.output_root.parent).free < 3_000_000_000:
        raise ValueError("insufficient physical evidence disk space")
    root = Path(__file__).parents[1]
    sources = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes())
        for path in (
            Path(__file__),
            Path(__file__).with_name("rsi_team_diverse_curriculum_collect.py"),
            Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
            root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
            root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
        )
    }
    args.output_root.mkdir(parents=True)
    outcomes = []
    with ThreadPoolExecutor(max_workers=7) as pool:
        futures = {
            pool.submit(_run_one, args.protocol, args.asset_root, args.output_root, name): name
            for name in names
        }
        for future in as_completed(futures):
            value = future.result()
            outcomes.append(value)
            print(
                f"{value['arm']} exit={value['exit_code']} hash={value['report_hash']}", flush=True
            )
    if sources != {path: hash_bytes((root / path).read_bytes()) for path in sources}:
        raise ValueError("v52 physical source changed during experiment")
    result = {
        "schema": "rsi_team_lateral_phase_orchestration_report_v52",
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
        "RSI_TEAM_V52_ORCHESTRATION="
        + json.dumps(
            {"report_hash": result["report_hash"], "all_jobs_complete": result["all_jobs_complete"]}
        ),
        flush=True,
    )
    if not result["all_jobs_complete"]:
        raise ValueError("incomplete v52 physical arm job")


if __name__ == "__main__":
    main()
