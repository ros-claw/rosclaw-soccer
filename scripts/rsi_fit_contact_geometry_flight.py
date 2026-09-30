"""Audit and test a SIM_ONLY first-contact ball-flight geometry model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_train_early_acquisition_selector import ARMS, load_verified_bank

from rosclaw_soccer.sim.contracts import hash_json


def contact_rows(root: Path, bank: dict[str, Any]) -> list[dict[str, Any]]:
    """Use only audited, measured first-foot contacts, never intended targets."""
    rows: list[dict[str, Any]] = []
    for episode in bank["episodes"]:
        for arm in ARMS:
            summary = episode["arms"][arm]
            if not summary["clean_foot_only"]:
                continue
            first = summary["first_contact_frame"]
            if first is None or summary["lateral_60_m"] is None:
                raise ValueError("clean first contact lacks 60-frame observation")
            folder = root / f"seed{episode['seed']}-lane{episode['lane']}-{arm}"
            with np.load(folder / "trace.npz", allow_pickle=False) as physics:
                force = physics["ball_body_contact_force_peak_n"][first, 0]
                foot = 0 if force[0] >= force[1] else 1
            with np.load(folder / "body_trace.npz", allow_pickle=False) as body:
                rel_y = float(
                    body["ball_position_before_step_m"][first, 0, 1]
                    - body["foot_geometry_position_before_step_m"][first, 0, foot, 1]
                )
            if not np.isfinite(rel_y):
                raise ValueError("non-finite measured foot-to-ball contact geometry")
            rows.append(
                {
                    "seed": episode["seed"],
                    "lane": episode["lane"],
                    "arm": arm,
                    "first_contact_frame": first,
                    "contacting_foot": foot,
                    "ball_minus_foot_y_m": rel_y,
                    "lateral_60_m": summary["lateral_60_m"],
                }
            )
    return rows


def evaluate(training_roots: list[Path], test_root: Path) -> dict[str, Any]:
    train_banks = [load_verified_bank(root)[0] for root in training_roots]
    test_bank = load_verified_bank(test_root)[0]
    banks = [*train_banks, test_bank]
    if (
        len(training_roots) != 2
        or [sorted({row["seed"] for row in bank["episodes"]}) for bank in banks]
        != [[20260953, 20260954], [20260955, 20260956], [20260957]]
        or any(
            bank[key] != train_banks[0][key]
            for bank in banks[1:]
            for key in ("source_hash", "asset_hash", "actor_hash")
        )
    ):
        raise ValueError("one-shot test requires frozen source/asset/actor and disjoint seeds")
    training = [
        row
        for root, bank in zip(training_roots, train_banks, strict=True)
        for row in contact_rows(root, bank)
    ]
    test = contact_rows(test_root, test_bank)
    x = np.asarray([row["ball_minus_foot_y_m"] for row in training])
    y = np.asarray([row["lateral_60_m"] for row in training])
    xt = np.asarray([row["ball_minus_foot_y_m"] for row in test])
    yt = np.asarray([row["lateral_60_m"] for row in test])
    if (
        len(training) < 32
        or len(test) < 8
        or not all(np.isfinite(value).all() for value in (x, y, xt, yt))
    ):
        raise ValueError("insufficient finite independent foot-only contact episodes")
    design = np.column_stack((np.ones(len(x)), x))
    beta = np.linalg.lstsq(design, y, rcond=None)[0]
    predicted = beta[0] + beta[1] * xt
    baseline = float(np.mean(y))
    mae = float(np.mean(np.abs(predicted - yt)))
    baseline_mae = float(np.mean(np.abs(baseline - yt)))
    passed = bool(mae <= 0.30 and mae <= 0.7 * baseline_mae)
    observations = [
        {**row, "predicted_lateral_60_m": float(value)}
        for row, value in zip(test, predicted, strict=True)
    ]
    result: dict[str, Any] = {
        "schema": "rsi_contact_geometry_flight_one_shot_v1",
        "activation_ceiling": "SIM_ONLY",
        "training_bank_hashes": [bank["report_hash"] for bank in train_banks],
        "test_bank_hash": test_bank["report_hash"],
        "training_clean_contact_count": len(training),
        "test_clean_contact_count": len(test),
        "training_mean_lateral_60_m": baseline,
        "intercept_m": float(beta[0]),
        "slope": float(beta[1]),
        "test_mae_m": mae,
        "test_mean_baseline_mae_m": baseline_mae,
        "test_observations": observations,
        "development_gate_passed": passed,
        "policy_input_authorized": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-bank-root", required=True, nargs=2, type=Path)
    parser.add_argument("--test-bank-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.test_bank_root in args.training_bank_root:
        parser.error("new output and disjoint training/test bank roots required")
    report = evaluate(args.training_bank_root, args.test_bank_root)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"CONTACT_GEOMETRY_TEST={report['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={report['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
