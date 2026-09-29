"""Resume-safe parallel SIM_ONLY collection of 512 diverse six-G1 football states."""

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


def _completed(path: Path, *, batch: int, arm: str, protocol_hash: str) -> bool:
    report_path = path / "report.json"
    if not report_path.is_file():
        if path.exists():
            raise ValueError(f"partial arm output requires operator review: {path}")
        return False
    report = json.loads(report_path.read_text())
    if (
        report.get("report_hash")
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report.get("protocol_hash") != protocol_hash
        or report.get("batch_index") != batch
        or report.get("arm", {}).get("name") != arm
        or len(report.get("rows", [])) != 32
    ):
        raise ValueError(f"completed arm output is unsealed or wrong batch: {path}")
    return True


def _run_job(
    *,
    root: Path,
    asset_root: Path,
    protocol: Path,
    batch: int,
    arm: str,
) -> dict[str, Any]:
    output = root / f"b{batch:02d}" / arm
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
            str(output),
            "--arm",
            arm,
            "--batch-index",
            str(batch),
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    report_path = output / "report.json"
    return {
        "batch": batch,
        "arm": arm,
        "exit_code": result.returncode,
        "report_hash": (
            json.loads(report_path.read_text()).get("report_hash")
            if report_path.is_file()
            else None
        ),
        "last_output_lines": result.stdout.splitlines()[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("bounded process count must be 1..12")
    protocol = json.loads(args.protocol.read_text())
    curriculum = protocol.get("curriculum", {})
    names = [arm["name"] for arm in protocol["arms"]]
    if (
        protocol.get("schema") != "rsi_team_diverse_curriculum_protocol_v38"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or curriculum.get("coordinate_seed") != 20261030
        or curriculum.get("identity_seed_base") != 1500000
        or curriculum.get("batch_count") != 16
        or curriculum.get("training_scene_count") != 32
        or curriculum.get("ball_x_m") != [3.25, 4.25]
        or curriculum.get("ball_y_m") != [-1.05, -0.35]
        or curriculum.get("ball_vx_mps") != [-0.90, -0.20]
        or names
        != [
            "parent",
            "adaptive_c19",
            "phase_front04_positive",
            "phase_front04_lat12",
            "phase_stay_lat12",
            "fixed_left_negative",
            "phase_front04_negative",
        ]
    ):
        raise ValueError("invalid frozen v49 physical curriculum")
    states = [
        (scenario.ball_initial_position_m, scenario.ball_initial_velocity_mps)
        for batch in range(16)
        for scenario in training_courses(protocol, batch)
    ]
    if len(states) != 512 or len(set(states)) != 512:
        raise ValueError("physical training states are not distinct")
    if shutil.disk_usage(args.output_root.parent).free < 10_000_000_000:
        raise ValueError("less than 10 GB free for physical training evidence")
    root = Path(__file__).parents[1]
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_diverse_curriculum_collect.py"),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    sources = {str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths}
    protocol_hash = hash_bytes(args.protocol.read_bytes())
    args.output_root.mkdir(parents=True, exist_ok=True)
    for batch in range(16):
        (args.output_root / f"b{batch:02d}").mkdir(exist_ok=True)
    tasks = [
        (batch, arm)
        for batch in range(16)
        for arm in names
        if not _completed(
            args.output_root / f"b{batch:02d}" / arm,
            batch=batch,
            arm=arm,
            protocol_hash=protocol_hash,
        )
    ]
    print(f"v49 jobs remaining={len(tasks)} of 112 workers={args.workers}", flush=True)
    outcomes: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(
                _run_job,
                root=args.output_root,
                asset_root=args.asset_root,
                protocol=args.protocol,
                batch=batch,
                arm=arm,
            ): (batch, arm)
            for batch, arm in tasks
        }
        for future in as_completed(jobs):
            value = future.result()
            outcomes.append(value)
            print(
                f"b{value['batch']:02d}/{value['arm']} exit={value['exit_code']} "
                f"report={value['report_hash']}",
                flush=True,
            )
    if sources != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during parallel curriculum collection")
    result = {
        "schema": "rsi_team_diverse_curriculum_orchestration_report_v49",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": protocol_hash,
        "source_hashes": sources,
        "physical_scene_count": 512,
        "planned_episode_count": 3584,
        "reused_complete_job_count": 112 - len(tasks),
        "job_results": sorted(
            outcomes, key=lambda value: (value["batch"], names.index(value["arm"]))
        ),
        "all_jobs_complete": len(outcomes) == len(tasks)
        and all(value["exit_code"] == 0 and value["report_hash"] for value in outcomes),
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "orchestration_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        "RSI_TEAM_V49_COLLECTION="
        + json.dumps(
            {key: result[key] for key in ("report_hash", "all_jobs_complete")},
            sort_keys=True,
        ),
        flush=True,
    )
    if not result["all_jobs_complete"]:
        raise ValueError("one or more physical training jobs incomplete; no audit/promotion")


if __name__ == "__main__":
    main()
