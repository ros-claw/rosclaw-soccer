"""Constrained online receive policy search that cannot forget the Parent event."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_online_receive_phase_v111 import _candidate, _weights

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def _retained(row: dict[str, Any]) -> bool:
    old = row["scenes"][0]
    return bool(
        row["safe_pair"]
        and old["local_b6_passed"]
        and old["first_receiver_foot_frame"] is not None
        and abs(old["first_receiver_foot_frame"] - 66) <= 4
    )


def train(asset_root: Path, protocol_path: Path, output: Path, workers: int = 3) -> dict[str, Any]:
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    scenes = [
        {"id": "s02", "ball_x_m": 2.24, "ball_y_m": -0.76, "seed": 512003},
        {"id": "s04", "ball_x_m": 2.32, "ball_y_m": -0.76, "seed": 512005},
    ]
    if (
        protocol["schema"] != "rosclaw_soccer.rsi.r1_retention_constrained_phase_v112.protocol.v1"
        or protocol["partition"] != "CONSUMED_DEV_RETENTION_SEARCH"
        or protocol["parent_report"]
        != "/data/rosclaw_overflow/rsi-r1-online-receive-phase-v111/selection.json"
        or protocol["parent_report_hash"]
        != "sha256:c54463d7330d8f802333ba2db165d934d7dc2880e400590a65d00bd81d4e5e3c"
        or protocol["retention_anchor"] != {"generation": 1, "candidate_index": 4}
        or protocol["training_scenes"] != scenes
        or protocol["seed"] != 112
        or protocol["first_generation_candidates"] != 12
        or protocol["first_generation_sigma"] != 0.18
        or protocol["second_generation_candidates"] != 8
        or protocol["second_generation_sigma"] != 0.12
        or protocol["rollout_count"] != 40
        or not 1 <= workers <= 4
        or output.exists()
    ):
        raise ValueError("frozen bounded retention-constrained course required")
    parent_path = Path(str(protocol["parent_report"]))
    parent = json.loads(parent_path.read_text(encoding="utf-8"))
    commitment = parent.get("report_hash")
    if (
        commitment != protocol["parent_report_hash"]
        or commitment
        != hash_json({key: value for key, value in parent.items() if key != "report_hash"})
        or parent.get("status") != "REJECTED_TRAINING"
        or parent.get("parent_reproduced") is not True
        or parent["generation_1"][4]["index"] != 4
        or parent["generation_1"][4]["scenes"][0]["local_b6_passed"] is not True
        or parent["generation_1"][4]["scenes"][1]["receiver_shin_before_foot"] is not True
    ):
        raise ValueError("v111 retained anchor is not bound or qualified")
    root = Path(__file__).resolve().parents[1]
    if output.resolve().is_relative_to(root):
        raise ValueError("external immutable retention evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_retention_constrained_phase_v112.py",
            "scripts/rsi_r1_online_receive_phase_v111.py",
            "scripts/rsi_r1_receiver_bridge_v71.py",
            "src/rosclaw_soccer/rsi/team_receive_phase_actor.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    anchor = np.asarray(parent["generation_1"][4]["weights"], dtype=np.float64)
    rng = np.random.default_rng(112)
    first_weights = [(0.0,) * 5, _weights(anchor)]
    first_weights.extend(_weights(anchor + rng.normal(0.0, 0.18, 5)) for _ in range(5))
    first_weights.extend(_weights(rng.normal(0.0, 0.18, 5)) for _ in range(5))
    output.mkdir(parents=True)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"generation-0-candidate-{i:02d}",
                scenes,
                weights,
                i,
            )
            for i, weights in enumerate(first_weights)
        ]
        first = [future.result() for future in futures]
    parent_reproduced = _retained(first[0])
    anchor_reproduced = _retained(first[1])
    retained = [row for row in first if _retained(row)]
    retained.sort(key=lambda row: (-row["scenes"][1]["reward"], -row["reward"], row["index"]))
    center = np.asarray(retained[0]["weights"] if retained else first[0]["weights"])
    second_weights = [_weights(center + rng.normal(0.0, 0.12, 5)) for _ in range(8)]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = [
            pool.submit(
                _candidate,
                asset_root,
                output / f"generation-1-candidate-{i:02d}",
                scenes,
                weights,
                i,
            )
            for i, weights in enumerate(second_weights)
        ]
        second = [future.result() for future in futures]
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("retention source changed during physical learning")
    selected = [row for row in first[2:] + second if row["two_scene_b6_passed"]]
    selected.sort(key=lambda row: (-row["reward"], row["index"]))
    winner = selected[0] if parent_reproduced and anchor_reproduced and selected else None
    result = {
        "schema": "rosclaw_soccer.rsi.r1_retention_constrained_phase_v112.result.v1",
        "protocol_hash": hash_bytes(protocol_path.read_bytes()),
        "parent_report_hash": commitment,
        "source_hashes": sources,
        "actual_rollouts": 2 * (len(first) + len(second)),
        "zero_parent_reproduced": parent_reproduced,
        "retained_anchor_reproduced": anchor_reproduced,
        "generation_0": first,
        "retained_generation_0_indices": [row["index"] for row in retained],
        "updated_center": center.tolist(),
        "generation_1": second,
        "status": (
            "INVALID_PARENT_REPRODUCTION"
            if not parent_reproduced or not anchor_reproduced
            else "CONSUMED_RETENTION_PHASE_PASSED"
            if winner
            else "REJECTED_TRAINING"
        ),
        "selected_weights": None if winner is None else winner["weights"],
        "fresh_evaluation_run": False,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    result["report_hash"] = hash_json(result)
    (output / "selection.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    result = train(args.asset_root, args.protocol, args.output, args.workers)
    print(
        json.dumps(
            {
                "status": result["status"],
                "retained_generation_0_indices": result["retained_generation_0_indices"],
                "selected_weights": result["selected_weights"],
                "report_hash": result["report_hash"],
            }
        )
    )


if __name__ == "__main__":
    main()
