"""SIM_ONLY pre-contact 12-leg feasibility search on a qualified physical tape.

This tests whether bounded early joint targets can remove the right shin hit.
The parent team-level decision and contact-teacher schedule remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsi_r1_event_reflex_proxy_v146 import replay
from rsi_r1_pd_proxy_search_v140 import load_tape

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.r1_precontact_leg_feasibility_v147.result.v1"


def make_tape(
    original: dict[str, NDArray[np.float64]], weights: NDArray[np.float64]
) -> dict[str, NDArray[np.float64]]:
    if weights.shape != (12,) or not np.isfinite(weights).all() or np.any(np.abs(weights) > 1):
        raise ValueError("finite bounded bilateral early leg targets required")
    target = original["motor_pd_target_rad"].copy()
    # Earliest change is at 0.28 s, over 0.2 s before the right first-foot event.
    # A smooth 80 ms ramp avoids a discontinuous motor target.
    for index in range(140, 400):
        fraction = min(1.0, (index - 140) / 40.0)
        if index > 300:
            fraction *= max(0.0, 1.0 - (index - 300) / 100.0)
        target[index, :12] += 0.25 * fraction * weights
    return {**original, "motor_pd_target_rad": target}


def score(row: dict[str, Any]) -> float:
    if not row["safe"] or row["first_foot_substep"] is None:
        return -1000.0
    return float(
        100.0 * float(not row["nonfoot_frames"])
        + 20.0 * min(0.03, row["minimum_right_shin_gap_m"])
        - 2.0 * row["terminal_foot_distance_m"]
        - row["terminal_ball_speed_mps"]
    )


def search(asset_root: Path, capture_dir: Path, output: Path, *, seed: int) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new bounded external SIM_ONLY evidence required")
    capture = json.loads((capture_dir / "report.json").read_text())
    if capture["status"] != "CAPTURE_QUALIFIED" or capture["report_hash"] != hash_json(
        {k: v for k, v in capture.items() if k != "report_hash"}
    ):
        raise ValueError("qualified physical capture required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    tapes = [
        load_tape(capture_dir / f"course-{row['course']['seed']}.npz", row["trace_hash"])
        for row in capture["rows"]
    ]
    if len(tapes) != 2:
        raise ValueError("paired bilateral physical tapes required")
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_precontact_leg_feasibility_v147.py",
            "scripts/rsi_r1_event_reflex_proxy_v146.py",
        )
    }
    output.mkdir(parents=True)
    zero = np.zeros(12, dtype=np.float64)
    baseline = [replay(model, make_tape(tape, zero), np.zeros(6)) for tape in tapes]
    if (
        [
            row["first_foot_substep"] // 10 if row["first_foot_substep"] is not None else None
            for row in baseline
        ]
        != [33, 25]
        or [row["nonfoot_frames"] for row in baseline] != [[], [26]]
        or max(row["max_parent_torque_error_nm"] for row in baseline) > 1e-8
    ):
        raise ValueError("zero-action bilateral parent reproduction required")
    rng = np.random.default_rng(seed)
    mean = np.zeros(12, dtype=np.float64)
    std = np.full(12, 0.5, dtype=np.float64)
    history: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(6):
        candidates = np.clip(rng.normal(mean, std, size=(64, 12)), -1, 1)
        candidates[0] = 0
        ranked = []
        for weights in candidates:
            right = replay(model, make_tape(tapes[1], weights), np.zeros(6))
            right["weights"] = weights.tolist()
            ranked.append((score(right), weights, right))
        ranked.sort(key=lambda item: item[0], reverse=True)
        elite = np.stack([row[1] for row in ranked[:8]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.07, elite.std(axis=0))
        winner = ranked[0][2]
        if best is None or score(winner) > score(best):
            best = winner
        row = {
            "generation": generation + 1,
            "right_clean_count": sum(not x[2]["nonfoot_frames"] for x in ranked),
            "right_foot_count": sum(x[2]["first_foot_substep"] is not None for x in ranked),
            "best_right": winner,
        }
        history.append(row)
        (output / "progress.json").write_text(json.dumps(history, indent=2, allow_nan=False) + "\n")
        print(
            json.dumps(
                {"generation": row["generation"], "clean": row["right_clean_count"], "best": winner}
            ),
            flush=True,
        )
    assert best is not None
    left = replay(model, make_tape(tapes[0], np.asarray(best["weights"])), np.zeros(6))
    paired = all(
        row["safe"] and row["first_foot_substep"] is not None and not row["nonfoot_frames"]
        for row in (left, best)
    )
    report = {
        "schema": SCHEMA,
        "capture_report_hash": capture["report_hash"],
        "source_hashes": source_hashes,
        "partition": "CONSUMED_PRECONTACT_BILATERAL_LEG_PROXY_FEASIBILITY",
        "seed": seed,
        "episode_count": 387,
        "baseline": baseline,
        "best_right": best,
        "left_retention": left,
        "history": history,
        "status": "DEVELOPMENT_PAIRED_CLEAN_UNVALIDATED"
        if paired
        else "REJECTED_PAIRED_CLEAN_GATE",
        "fixed_high_level_teacher_schedule": True,
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
        raise ValueError("source drift during physical feasibility search")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = search(args.asset_root, args.capture_dir, args.output, seed=args.seed)
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
