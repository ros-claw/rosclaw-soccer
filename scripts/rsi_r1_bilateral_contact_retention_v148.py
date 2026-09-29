"""SIM_ONLY retention-constrained bilateral pre-contact leg optimization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsi_r1_event_reflex_proxy_v146 import replay
from rsi_r1_pd_proxy_search_v140 import load_tape
from rsi_r1_precontact_leg_feasibility_v147 import make_tape

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_bilateral_contact_retention_v148.result.v1"


def grade(rows: list[dict[str, Any]]) -> float:
    clean = [
        row["safe"] and row["first_foot_substep"] is not None and not row["nonfoot_frames"]
        for row in rows
    ]
    if not all(clean):
        return float(-100.0 + 10.0 * sum(clean))
    distances = [float(row["terminal_foot_distance_m"]) for row in rows]
    speeds = [float(row["terminal_ball_speed_mps"]) for row in rows]
    controlled = [
        distance <= 0.35 and speed <= 0.35
        for distance, speed in zip(distances, speeds, strict=True)
    ]
    return 100.0 + 25.0 * sum(controlled) - 4.0 * sum(distances) - 2.0 * sum(speeds)


def optimize(
    asset_root: Path, capture_dir: Path, parent_report: Path, output: Path, *, seed: int
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new external SIM_ONLY bilateral evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    parent = json.loads(parent_report.read_text())
    if (
        capture["status"] != "CAPTURE_QUALIFIED"
        or parent["status"] != "DEVELOPMENT_PAIRED_CLEAN_UNVALIDATED"
        or parent["capture_report_hash"] != capture["report_hash"]
        or any(
            row["report_hash"] != hash_json({k: v for k, v in row.items() if k != "report_hash"})
            for row in (capture, parent)
        )
    ):
        raise ValueError("qualified source tape and development parent required")
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_bilateral_contact_retention_v148.py",
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
        raise ValueError("two opposed physical receiving courses required")
    output.mkdir(parents=True)

    def evaluate(weights: NDArray[np.float64]) -> dict[str, Any]:
        rows = [replay(model, make_tape(tape, weights), np.zeros(6)) for tape in tapes]
        return {"weights": weights.tolist(), "sides": rows, "score": grade(rows)}

    parent_weights = np.asarray(parent["best_right"]["weights"], dtype=np.float64)
    baseline = evaluate(parent_weights)
    if not all(
        row["safe"] and row["first_foot_substep"] is not None and not row["nonfoot_frames"]
        for row in baseline["sides"]
    ):
        raise ValueError("paired parent clean contact retention required")
    rng = np.random.default_rng(seed)
    mean = parent_weights.copy()
    std = np.full(12, 0.18, dtype=np.float64)
    best = baseline
    history = []
    for generation in range(5):
        samples = np.clip(rng.normal(mean, std, size=(48, 12)), -1, 1)
        samples[0] = parent_weights
        ranked = [evaluate(weights) for weights in samples]
        ranked.sort(key=lambda row: row["score"], reverse=True)
        if ranked[0]["score"] > best["score"]:
            best = ranked[0]
        elite = np.stack([np.asarray(row["weights"]) for row in ranked[:8]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.04, elite.std(axis=0))
        row = {
            "generation": generation + 1,
            "paired_clean_count": sum(row["score"] >= 100 for row in ranked),
            "paired_controlled_count": sum(
                row["score"] >= 150 - 4 * 0.7 - 2 * 0.7 for row in ranked
            ),
            "best": ranked[0],
        }
        history.append(row)
        (output / "progress.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {
                    "generation": row["generation"],
                    "paired_clean": row["paired_clean_count"],
                    "paired_controlled": row["paired_controlled_count"],
                    "best": ranked[0],
                }
            ),
            flush=True,
        )
    both_controlled = all(
        row["safe"]
        and row["first_foot_substep"] is not None
        and not row["nonfoot_frames"]
        and row["terminal_foot_distance_m"] <= 0.35
        and row["terminal_ball_speed_mps"] <= 0.35
        for row in best["sides"]
    )
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "parent_report_hash": parent["report_hash"],
        "source_hashes": source_hashes,
        "partition": "CONSUMED_BILATERAL_CONTACT_RETENTION_PROXY",
        "seed": seed,
        "episode_count": 482,
        "baseline": baseline,
        "best": best,
        "history": history,
        "status": "DEVELOPMENT_PAIRED_CONTROLLED_UNVALIDATED"
        if both_controlled
        else "REJECTED_PAIRED_CONTROLLED_GATE",
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
        raise ValueError("source drift during bilateral contact training")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = optimize(
        args.asset_root, args.capture_dir, args.parent_report, args.output, seed=args.seed
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
