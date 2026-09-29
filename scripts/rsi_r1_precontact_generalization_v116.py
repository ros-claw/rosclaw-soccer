"""Train the precontact receiving phase under a hard old-skill retention gate."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_online_receive_phase_v111 import _candidate, _weights

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _first_foot_score(row: dict[str, Any]) -> int:
    return int(
        row["safe"]
        and row["robot_collision_free"]
        and row["clean_transfer"]
        and row["dynamic_incoming"]
        and row["first_receiver_foot_frame"] is not None
        and not row["receiver_shin_before_foot"]
    )


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text())
    parent_path = Path(str(protocol["parent_postcontact_audit"]))
    parent = json.loads(parent_path.read_text())
    scenes = protocol["training_scenes"]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_precontact_generalization_v116.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_PRECONTACT_GENERALIZATION"
        or parent["report_hash"] != protocol["parent_postcontact_hash"]
        or parent["report_hash"]
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent["status"] != "REJECTED_NO_DEV_GAIN"
        or protocol["seed"] != 116
        or protocol["candidate_count"] != 12
        or protocol["rollout_count"] != 48
        or len(scenes) != 4
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen precontact generalization course required")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external physical evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_precontact_generalization_v116.py",
            "scripts/rsi_r1_online_receive_phase_v111.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_phase_actor.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    anchor = np.asarray(protocol["anchor_weights"], dtype=np.float64)
    rng = np.random.default_rng(116)
    weights = [_weights(anchor)]
    weights.extend(_weights(anchor + rng.normal(0.0, 0.12, 5)) for _ in range(7))
    weights.extend(_weights(anchor + rng.normal(0.0, 0.30, 5)) for _ in range(4))
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(_candidate, asset_root, output / f"candidate-{index:02d}", scenes, w, index)
            for index, w in enumerate(weights)
        ]
        rows = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("precontact learning source drift")
    for row in rows:
        row["anchors_retained"] = all(bool(scene["local_b6_passed"]) for scene in row["scenes"][:2])
        development = row["scenes"][2:]
        row["development_first_foot_count"] = sum(_first_foot_score(scene) for scene in development)
        row["development_second_foot_count"] = sum(
            bool(scene["local_b6_passed"]) for scene in development
        )
        row["development_shin_first_count"] = sum(
            bool(scene["receiver_shin_before_foot"]) for scene in development
        )
    baseline = rows[0]
    eligible = [
        row
        for row in rows[1:]
        if row["anchors_retained"]
        and all(scene["safe"] and scene["robot_collision_free"] for scene in row["scenes"])
        and row["development_shin_first_count"] <= baseline["development_shin_first_count"]
        and (
            row["development_second_foot_count"] > baseline["development_second_foot_count"]
            or row["development_first_foot_count"] > baseline["development_first_foot_count"]
        )
    ]
    eligible.sort(
        key=lambda row: (
            -row["development_second_foot_count"],
            -row["development_first_foot_count"],
            row["development_shin_first_count"],
            row["index"],
        )
    )
    winner = eligible[0] if baseline["anchors_retained"] and eligible else None
    result = {
        "schema": "rosclaw_soccer.rsi.r1_precontact_generalization_v116.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "parent_postcontact_hash": parent["report_hash"],
        "source_hashes": sources,
        "actual_rollouts": len(rows) * len(scenes),
        "candidates": rows,
        "baseline_retained": baseline["anchors_retained"],
        "selected_weights": None if winner is None else winner["weights"],
        "status": "CONSUMED_DEV_IMPROVEMENT" if winner else "REJECTED_NO_DEV_GAIN",
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "selection.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "selected_weights": result["selected_weights"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
