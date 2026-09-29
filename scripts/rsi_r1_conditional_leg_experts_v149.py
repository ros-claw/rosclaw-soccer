"""SIM_ONLY opposite-side pre-contact experts with explicit skill retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_event_reflex_proxy_v146 import replay
from rsi_r1_pd_proxy_search_v140 import load_tape
from rsi_r1_precontact_leg_feasibility_v147 import make_tape

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_conditional_leg_experts_v149.result.v1"


def left_score(row: dict[str, Any]) -> float:
    if not row["safe"] or row["first_foot_substep"] is None or row["nonfoot_frames"]:
        return -100.0
    controlled = row["terminal_foot_distance_m"] <= 0.35 and row["terminal_ball_speed_mps"] <= 0.35
    return float(
        100.0 * float(controlled)
        - 2.0 * row["terminal_foot_distance_m"]
        - 3.0 * row["terminal_ball_speed_mps"]
    )


def search(
    asset_root: Path, capture_dir: Path, right_parent: Path, output: Path, *, seed: int
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new bounded external SIM_ONLY conditional expert evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    parent = json.loads(right_parent.read_text())
    if (
        capture["status"] != "CAPTURE_QUALIFIED"
        or parent["status"] != "DEVELOPMENT_PAIRED_CLEAN_UNVALIDATED"
        or parent["capture_report_hash"] != capture["report_hash"]
        or any(
            item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
            for item in (capture, parent)
        )
    ):
        raise ValueError("qualified paired parent capture required")
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_conditional_leg_experts_v149.py",
            "scripts/rsi_r1_precontact_leg_feasibility_v147.py",
            "scripts/rsi_r1_event_reflex_proxy_v146.py",
        )
    }
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    tapes = [
        load_tape(capture_dir / f"course-{row['course']['seed']}.npz", row["trace_hash"])
        for row in capture["rows"]
    ]
    if len(tapes) != 2:
        raise ValueError("two opposed physical courses required")
    output.mkdir(parents=True)
    right_weights = np.asarray(parent["best_right"]["weights"], dtype=np.float64)
    right = replay(model, make_tape(tapes[1], right_weights), np.zeros(6))
    if not right["safe"] or right["first_foot_substep"] is None or right["nonfoot_frames"]:
        raise ValueError("right learned clean skill must be retained")
    rng = np.random.default_rng(seed)
    mean = right_weights.copy()
    std = np.full(12, 0.35, dtype=np.float64)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(6):
        samples = np.clip(rng.normal(mean, std, size=(64, 12)), -1, 1)
        samples[0] = right_weights
        ranked = []
        for weights in samples:
            left = replay(model, make_tape(tapes[0], weights), np.zeros(6))
            left["weights"] = weights.tolist()
            ranked.append((left_score(left), weights, left))
        ranked.sort(key=lambda item: item[0], reverse=True)
        if best is None or ranked[0][0] > left_score(best):
            best = ranked[0][2]
        elite = np.stack([item[1] for item in ranked[:8]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.05, elite.std(axis=0))
        row = {
            "generation": generation + 1,
            "left_clean_count": sum(
                item[2]["safe"]
                and item[2]["first_foot_substep"] is not None
                and not item[2]["nonfoot_frames"]
                for item in ranked
            ),
            "left_controlled_count": sum(
                item[2]["safe"]
                and item[2]["first_foot_substep"] is not None
                and not item[2]["nonfoot_frames"]
                and item[2]["terminal_foot_distance_m"] <= 0.35
                and item[2]["terminal_ball_speed_mps"] <= 0.35
                for item in ranked
            ),
            "best_left": ranked[0][2],
        }
        history.append(row)
        (output / "progress.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "generation": row["generation"],
                    "clean": row["left_clean_count"],
                    "controlled": row["left_controlled_count"],
                    "best": row["best_left"],
                }
            ),
            flush=True,
        )
    assert best is not None
    left_controlled = (
        best["safe"]
        and best["first_foot_substep"] is not None
        and not best["nonfoot_frames"]
        and best["terminal_foot_distance_m"] <= 0.35
        and best["terminal_ball_speed_mps"] <= 0.35
    )
    right_controlled = (
        right["terminal_foot_distance_m"] <= 0.35 and right["terminal_ball_speed_mps"] <= 0.35
    )
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "parent_report_hash": parent["report_hash"],
        "source_hashes": source_hashes,
        "partition": "CONSUMED_OPPOSITE_SIDE_EXPERTS_PHYSICAL_PROXY",
        "seed": seed,
        "episode_count": 385,
        "right_retained": right,
        "best_left": best,
        "history": history,
        "status": "DEVELOPMENT_BILATERAL_CONTROLLED_UNVALIDATED"
        if left_controlled and right_controlled
        else "REJECTED_BILATERAL_CONTROLLED_GATE",
        "full_world_audition_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(
        hash_bytes((root / name).read_bytes()) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("source drift during conditional expert search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--right-parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = search(
        args.asset_root, args.capture_dir, args.right_parent, args.output, seed=args.seed
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
