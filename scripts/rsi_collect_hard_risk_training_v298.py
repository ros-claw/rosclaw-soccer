"""Outcome-blind hard-ball TRAIN_CONSUMED bank for causal motor/risk learning."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_shadow_approach_fresh_v295 import _course

V296_HASH = "sha256:540da31452042c69d61a1ecb598915c0f416051c422e2b637c1de19e3ea573a1"
V297_HASH = "sha256:f3e74406ea3f7895981df6d1816f1d8f2197e20f95df80124b103bb6bb158499"


def courses() -> tuple[tuple[int, int], ...]:
    selected = [
        (seed, lane)
        for seed in range(20261161, 20261700)
        for lane, (x, y, vx) in enumerate(sample_training_courses(seed, 16))
        if lane % 2 == 0 and vx <= -0.6 and 2.2 <= x <= 2.4 and -0.10 <= y <= -0.06
    ][:40]
    if len(selected) != 40 or selected[-1][0] != 20261446 or len(set(selected)) != 40:
        raise ValueError("outcome-blind training course catalog drifted")
    return tuple(selected)


def _gpu_courses(
    gpu: int,
    catalog: tuple[tuple[int, int], ...],
    common: dict[str, Any],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for index, (seed, lane) in enumerate(catalog):
        if index % 4 != gpu:
            continue
        try:
            rows.append(_course(seed, lane, gpu, **common))
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"HARD_RISK_TRAIN_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--v296-report", required=True, type=Path)
    parser.add_argument("--v297-report", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    earlier = json.loads(args.v296_report.read_text(encoding="utf-8"))
    failed_model = json.loads(args.v297_report.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        args.output_root.exists()
        or earlier.get("report_hash") != V296_HASH
        or earlier.get("report_hash")
        != hash_json({key: value for key, value in earlier.items() if key != "report_hash"})
        or earlier.get("hard_fresh_gate_passed") is not True
        or failed_model.get("report_hash") != V297_HASH
        or failed_model.get("report_hash")
        != hash_json({key: value for key, value in failed_model.items() if key != "report_hash"})
        or any(model["safety_eligible"] for model in failed_model.get("models", []))
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed v296 and failed v297, qualified assets, new output required")
    catalog = courses()
    args.output_root.mkdir(parents=True)
    (args.output_root / "logs").mkdir()
    source_hash = hash_bytes(runner.read_bytes())
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        source_hash=source_hash,
    )
    rows = []
    failures = []
    with ThreadPoolExecutor(max_workers=4) as pool:
        futures = [pool.submit(_gpu_courses, gpu, catalog, common) for gpu in range(4)]
        for future in as_completed(futures):
            gpu_rows, gpu_failures = future.result()
            rows.extend(gpu_rows)
            failures.extend(gpu_failures)
    rows.sort(key=lambda row: catalog.index((row["seed"], row["lane"])))
    complete = (
        len(rows) == len(catalog)
        and {(row["seed"], row["lane"]) for row in rows} == set(catalog)
        and not failures
    )
    result: dict[str, Any] = {
        "schema": "rsi_isaac_hard_risk_train_bank_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "TRAIN_CONSUMED_NOT_FRESH",
        "v296_report_hash": V296_HASH,
        "v297_report_hash": V297_HASH,
        "course_selection_hash": hash_json(catalog),
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        "complete": complete,
        "baseline_high_quality": sum(row["arms"]["gain_08"]["high_quality"] for row in rows),
        "aggressive_high_quality": sum(row["arms"]["gain_12"]["high_quality"] for row in rows),
        "baseline_clean_foot": sum(row["arms"]["gain_08"]["clean_foot_only"] for row in rows),
        "aggressive_clean_foot": sum(row["arms"]["gain_12"]["clean_foot_only"] for row in rows),
        "candidate_new_nonfoot": sum(
            row["arms"]["gain_08"]["clean_foot_only"]
            and not row["arms"]["gain_12"]["clean_foot_only"]
            for row in rows
        ),
        "candidate_new_out_of_play": sum(
            row["arms"]["gain_08"]["maximum_lateral_excursion_m"] <= 4
            and row["arms"]["gain_12"]["maximum_lateral_excursion_m"] > 4
            for row in rows
        ),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"HARD_RISK_TRAIN_BANK={result['report_hash']}", flush=True)
    print(f"COMPLETE={complete}", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
