"""Sealed parent/candidate Fresh8 physical comparison of learned receiver phase."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_online_receive_phase_v111 import _episode, _weights

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit(asset_root: Path, protocol_path: Path, output: Path, workers: int = 4) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scenes = [
        {"id": "f01", "ball_x_m": 2.22, "ball_y_m": -0.75, "seed": 712101},
        {"id": "f02", "ball_x_m": 2.26, "ball_y_m": -0.77, "seed": 712102},
        {"id": "f03", "ball_x_m": 2.30, "ball_y_m": -0.75, "seed": 712103},
        {"id": "f04", "ball_x_m": 2.34, "ball_y_m": -0.77, "seed": 712104},
        {"id": "f05", "ball_x_m": 2.22, "ball_y_m": -0.79, "seed": 712105},
        {"id": "f06", "ball_x_m": 2.30, "ball_y_m": -0.71, "seed": 712106},
        {"id": "f07", "ball_x_m": 2.26, "ball_y_m": -0.73, "seed": 712107},
        {"id": "f08", "ball_x_m": 2.34, "ball_y_m": -0.79, "seed": 712108},
    ]
    selected = [
        -0.1853410496026612,
        -0.2120161425477506,
        -0.10038265224958914,
        0.1304744032562346,
        0.16730409297832222,
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_receive_phase_fresh_v113.protocol.v1"
        or protocol["partition"] != "FRESH_UNSEEN_LOCAL_B6"
        or protocol["training_selection_path"]
        != "/data/rosclaw_overflow/rsi-r1-retention-constrained-phase-v112/selection.json"
        or protocol["training_selection_hash"]
        != "sha256:ee4321a90b8d02c356b17ae1cafd3053e5bdc0fd4e077ff143b9552e9a85eb73"
        or protocol["candidate_weights"] != selected
        or protocol["parent_weights"] != [0.0] * 5
        or protocol["scenes"] != scenes
        or protocol["paired_rollout_count"] != 16
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen sealed Fresh8 receive phase course required")
    training = json.loads(Path(str(protocol["training_selection_path"])).read_text())
    if (
        training.get("report_hash") != protocol["training_selection_hash"]
        or training["report_hash"]
        != hash_json({key: value for key, value in training.items() if key != "report_hash"})
        or training.get("status") != "CONSUMED_RETENTION_PHASE_PASSED"
        or training.get("selected_weights") != selected
        or training.get("fresh_evaluation_run") is not False
    ):
        raise ValueError("frozen training selection is unbound")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable Fresh8 evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_receive_phase_fresh_v113.py",
            "scripts/rsi_r1_online_receive_phase_v111.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_phase_actor.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    parent_weights = _weights(np.zeros(5))
    candidate_weights = _weights(np.asarray(selected, dtype=float))
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _episode,
                asset_root,
                output / arm / str(scene["id"]),
                scene,
                parent_weights if arm == "parent" else candidate_weights,
            )
            for arm in ("parent", "candidate")
            for scene in scenes
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("Fresh8 source changed during physical exam")
    parent_rows, candidate_rows = rows[:8], rows[8:]
    parent_b6 = sum(row["local_b6_passed"] for row in parent_rows)
    candidate_b6 = sum(row["local_b6_passed"] for row in candidate_rows)
    safe = sum(row["safe"] and row["robot_collision_free"] for row in candidate_rows)
    clean_dynamic = sum(row["clean_transfer"] and row["dynamic_incoming"] for row in candidate_rows)
    passed = bool(
        safe == 8 and clean_dynamic >= 6 and candidate_b6 >= 4 and candidate_b6 - parent_b6 >= 3
    )
    result = {
        "schema": "rosclaw_soccer.rsi.r1_receive_phase_fresh_v113.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "training_selection_hash": training["report_hash"],
        "source_hashes": sources,
        "actual_rollouts": len(rows),
        "parent": parent_rows,
        "candidate": candidate_rows,
        "parent_local_b6_count": parent_b6,
        "candidate_local_b6_count": candidate_b6,
        "candidate_safe_collision_free_count": safe,
        "candidate_clean_dynamic_count": clean_dynamic,
        "candidate_full_goal_count": sum(row["chain_success"] for row in candidate_rows),
        "status": "FRESH_LOCAL_B6_PASSED" if passed else "FRESH_LOCAL_B6_REJECTED",
        "local_policy_qualified": passed,
        "full_team_chain_qualified": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "audit.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = audit(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "parent_b6": result["parent_local_b6_count"],
                "candidate_b6": result["candidate_local_b6_count"],
                "candidate_safe": result["candidate_safe_collision_free_count"],
                "candidate_clean_dynamic": result["candidate_clean_dynamic_count"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
