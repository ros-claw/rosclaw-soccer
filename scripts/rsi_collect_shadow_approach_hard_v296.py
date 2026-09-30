"""Fresh high-speed, negative-side hard-course examination of shadow approach."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_shadow_approach_fresh_v295 import _course

V295_HASH = "sha256:eefecf2089224345a10d4b9e45f382685c71ccc16de778a6190510667fe658f8"
COURSES = (
    (20261022, 2),
    (20261035, 0),
    (20261043, 0),
    (20261047, 2),
    (20261050, 6),
    (20261060, 6),
    (20261070, 6),
    (20261094, 6),
    (20261095, 2),
    (20261097, 2),
    (20261107, 0),
    (20261108, 0),
    (20261113, 0),
    (20261146, 4),
    (20261148, 2),
)


def preflight() -> str:
    chosen = [
        (seed, lane)
        for seed in range(20261009, 20261161)
        for lane, (x, y, vx) in enumerate(sample_training_courses(seed, 16))
        if lane % 2 == 0 and vx <= -0.6 and 2.2 <= x <= 2.4 and -0.10 <= y <= -0.06
    ]
    if tuple(chosen) != COURSES:
        raise ValueError("preregistered outcome-blind hard catalog drifted")
    return hash_json(COURSES)


def _gpu_courses(
    gpu: int, common: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for index, (seed, lane) in enumerate(COURSES):
        if index % 4 != gpu:
            continue
        try:
            rows.append(_course(seed, lane, gpu, **common))
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"SHADOW_HARD_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def score(rows: list[dict[str, Any]], failures: list[dict[str, Any]]) -> dict[str, Any]:
    complete = (
        len(rows) == len(COURSES)
        and {(row["seed"], row["lane"]) for row in rows} == set(COURSES)
        and not failures
    )
    base = [row["arms"]["gain_08"] for row in rows]
    selected = [row["arms"][row["selected_arm"]] for row in rows]
    aggressive = [row["arms"]["gain_12"] for row in rows]
    high_gain = sum(c["high_quality"] for c in selected) - sum(b["high_quality"] for b in base)
    reward_gain = sum(c["reward"] - b["reward"] for b, c in zip(base, selected, strict=True)) / len(
        COURSES
    )
    clean_loss = sum(
        b["clean_foot_only"] and not c["clean_foot_only"]
        for b, c in zip(base, selected, strict=True)
    )
    new_out = sum(
        b["maximum_lateral_excursion_m"] <= 4 and c["maximum_lateral_excursion_m"] > 4
        for b, c in zip(base, selected, strict=True)
    )
    vetoed = [row for row in rows if row["selected_gain"] == 0.8]
    return {
        "complete": complete,
        "hard_course_count": len(rows),
        "baseline_high_quality": sum(row["high_quality"] for row in base),
        "selected_high_quality": sum(row["high_quality"] for row in selected),
        "aggressive_high_quality_diagnostic": sum(row["high_quality"] for row in aggressive),
        "baseline_clean_foot": sum(row["clean_foot_only"] for row in base),
        "selected_clean_foot": sum(row["clean_foot_only"] for row in selected),
        "aggressive_clean_foot_diagnostic": sum(row["clean_foot_only"] for row in aggressive),
        "veto_count": len(vetoed),
        "veto_true_risk_count": sum(
            not row["arms"]["gain_12"]["clean_foot_only"] for row in vetoed
        ),
        "veto_false_risk_count": sum(row["arms"]["gain_12"]["clean_foot_only"] for row in vetoed),
        "unvetoed_new_nonfoot_count": sum(
            row["selected_gain"] == 1.2
            and row["arms"]["gain_08"]["clean_foot_only"]
            and not row["arms"]["gain_12"]["clean_foot_only"]
            for row in rows
        ),
        "high_quality_gain": high_gain,
        "mean_reward_gain": reward_gain,
        "clean_foot_loss": clean_loss,
        "new_out_of_play": new_out,
        "hard_fresh_gate_passed": bool(
            complete
            and high_gain >= 3
            and reward_gain >= 0.3
            and clean_loss == 0
            and new_out == 0
            and all(c["minimum_pelvis_z_m"] >= 0.65 for c in selected)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--v295-report", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    earlier = json.loads(args.v295_report.read_text(encoding="utf-8"))
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    if (
        args.output_root.exists()
        or earlier.get("report_hash") != V295_HASH
        or earlier.get("report_hash")
        != hash_json({key: value for key, value in earlier.items() if key != "report_hash"})
        or earlier.get("fresh_gate_passed") is not True
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed passing v295 evidence and new qualified output required")
    catalog_hash = preflight()
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
        futures = [pool.submit(_gpu_courses, gpu, common) for gpu in range(4)]
        for future in as_completed(futures):
            gpu_rows, gpu_failures = future.result()
            rows.extend(gpu_rows)
            failures.extend(gpu_failures)
    rows.sort(key=lambda row: COURSES.index((row["seed"], row["lane"])))
    result: dict[str, Any] = {
        "schema": "rsi_isaac_shadow_approach_hard_fresh_v1",
        "activation_ceiling": "SIM_ONLY",
        "v295_report_hash": V295_HASH,
        "course_catalog_selection_hash": catalog_hash,
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        **score(rows, failures),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"SHADOW_APPROACH_HARD_FRESH={result['report_hash']}", flush=True)
    print(f"HARD_FRESH_GATE_PASSED={result['hard_fresh_gate_passed']}", flush=True)
    if not result["complete"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
