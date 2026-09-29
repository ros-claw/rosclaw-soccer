"""Collect a sealed, bounded multi-arm six-G1 paired MuJoCo curriculum."""

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


def _completed(path: Path, *, batch: int, arm: str, protocol_hash: str) -> str | None:
    report_path = path / "report.json"
    if not report_path.is_file():
        if path.exists():
            raise ValueError(f"partial physical arm output requires review: {path}")
        return None
    report = json.loads(report_path.read_text())
    if (
        report.get("report_hash")
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report.get("protocol_hash") != protocol_hash
        or report.get("batch_index") != batch
        or report.get("arm", {}).get("name") != arm
        or len(report.get("rows", ())) != 32
    ):
        raise ValueError(f"unsealed or wrong physical arm output: {path}")
    return str(report["report_hash"])


def _job(protocol: Path, asset_root: Path, root: Path, batch: int, arm: str) -> dict[str, Any]:
    folder = root / f"b{batch:02d}" / arm
    env = os.environ.copy()
    env.update(OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1")
    run = subprocess.run(
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
            str(batch),
        ],
        cwd=Path(__file__).parents[1],
        env=env,
        check=False,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    report_path = folder / "report.json"
    return {
        "batch": batch,
        "arm": arm,
        "exit_code": run.returncode,
        "report_hash": (
            json.loads(report_path.read_text()).get("report_hash")
            if report_path.is_file()
            else None
        ),
        "last_output_lines": run.stdout.splitlines()[-5:],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", required=True, type=Path)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()
    if not 1 <= args.workers <= 12:
        parser.error("bounded 1..12 worker count required")
    protocol = json.loads(args.protocol.read_text())
    curriculum = protocol.get("curriculum", {})
    names = [arm["name"] for arm in protocol["arms"]]
    batches = curriculum.get("batch_count")
    if (
        protocol.get("schema") != "rsi_team_diverse_curriculum_protocol_v38"
        or protocol.get("activation_ceiling") != "SIM_ONLY"
        or protocol.get("development_only") is not True
        or protocol.get("promotion_authorized") is not False
        or type(batches) is not int
        or batches not in (1, 2, 16)
        or curriculum.get("training_scene_count") != 32
        or len(names) not in (3, 7)
        or len(set(names)) != len(names)
        or names[0:2] != ["parent", "baseline"]
        or (batches == 16 and names != ["parent", "baseline", "gate22_cap10"])
        or (batches != 16 and len(names) != 7)
        or protocol.get("frames") != 250
    ):
        raise ValueError("invalid bounded paired physical curriculum")
    states = [
        (scene.ball_initial_position_m, scene.ball_initial_velocity_mps)
        for batch in range(batches)
        for scene in training_courses(protocol, batch)
    ]
    if len(states) != 32 * batches or len(set(states)) != len(states):
        raise ValueError("duplicate or seed-only physical scenes")
    if shutil.disk_usage(args.output_root.parent).free < 4_000_000_000:
        raise ValueError("less than 4 GB for physical evidence")
    root = Path(__file__).parents[1]
    source_paths = (
        Path(__file__),
        Path(__file__).with_name("rsi_team_diverse_curriculum_collect.py"),
        Path(__file__).with_name("rsi_team_taskspace_first_touch.py"),
        root / "src/rosclaw_soccer/rsi/taskspace_swing_probe.py",
        root / "src/rosclaw_soccer/rsi/taskspace_swing_evidence.py",
        root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
    )
    source_hashes = {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }
    args.output_root.mkdir(parents=True, exist_ok=True)
    for batch in range(batches):
        (args.output_root / f"b{batch:02d}").mkdir(exist_ok=True)
    protocol_hash = hash_bytes(args.protocol.read_bytes())
    complete_jobs = {
        (batch, arm): _completed(
            args.output_root / f"b{batch:02d}" / arm,
            batch=batch,
            arm=arm,
            protocol_hash=protocol_hash,
        )
        for batch in range(batches)
        for arm in names
    }
    outcomes = [
        {"batch": batch, "arm": arm, "exit_code": 0, "report_hash": commitment, "reused": True}
        for (batch, arm), commitment in complete_jobs.items()
        if commitment is not None
    ]
    tasks = [
        (batch, arm) for (batch, arm), commitment in complete_jobs.items() if commitment is None
    ]
    print(f"paired physics jobs remaining={len(tasks)} of {batches * len(names)}", flush=True)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        jobs = {
            pool.submit(_job, args.protocol, args.asset_root, args.output_root, batch, arm): (
                batch,
                arm,
            )
            for batch, arm in tasks
        }
        for future in as_completed(jobs):
            value = future.result()
            outcomes.append(value)
            print(
                f"b{value['batch']:02d}/{value['arm']} exit={value['exit_code']} "
                f"hash={value['report_hash']}",
                flush=True,
            )
    if source_hashes != {
        str(path.relative_to(root)): hash_bytes(path.read_bytes()) for path in source_paths
    }:
        raise ValueError("physical source changed during parallel collection")
    complete = len(outcomes) == batches * len(names) and all(
        value["exit_code"] == 0 and value["report_hash"] for value in outcomes
    )
    result = {
        "schema": "rsi_team_paired_physics_orchestration_report_v1",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "protocol_hash": protocol_hash,
        "source_hashes": source_hashes,
        "physical_scene_count": len(states),
        "planned_episode_count": len(states) * len(names),
        "reused_complete_job_count": len(complete_jobs) - len(tasks),
        "job_results": sorted(outcomes, key=lambda row: (row["batch"], names.index(row["arm"]))),
        "all_jobs_complete": complete,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "orchestration_report.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(
        "RSI_TEAM_PAIRED_COLLECTION="
        + json.dumps({"report_hash": result["report_hash"], "all_jobs_complete": complete}),
        flush=True,
    )
    if not complete:
        raise ValueError("one or more physical paired curriculum jobs failed")


if __name__ == "__main__":
    main()
