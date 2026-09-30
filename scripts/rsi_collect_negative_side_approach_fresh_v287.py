"""One-shot fresh-seed evaluation of bounded negative-side G1 approach."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run

SEEDS = (20260967, 20260968, 20260969, 20260970)
LANES = tuple(range(0, 16, 2))
COURSES = tuple((seed, lane) for seed in SEEDS for lane in LANES)
EXPECTED_V286_HASH = "sha256:a97b1b650acbf0c2cce75ce4b0de0eebab68b2eaf429709e439fec36106cd06c"


def high_quality(row: dict[str, Any]) -> bool:
    forward = row["forward_60_m"]
    lateral = row["lateral_60_m"]
    return bool(
        row["clean_foot_only"]
        and forward is not None
        and lateral is not None
        and forward >= 1
        and abs(lateral) / max(forward, 0.01) <= 0.3
        and row["maximum_lateral_excursion_m"] <= 4
    )


def _course(
    seed: int,
    lane: int,
    *,
    root: Path,
    runner: Path,
    isaac_python: Path,
    g1_usd: Path,
    model_root: Path,
    actor: Path,
    source_hash: str,
) -> dict[str, Any]:
    gpu = (lane // 2) % 4
    results = {}
    courses = []
    for arm, gain, negative_only in (
        ("baseline", 0.0, False),
        ("negative_side", 0.8, True),
    ):
        parent, parent_receipt = _run(
            root,
            runner,
            isaac_python,
            g1_usd,
            model_root,
            actor,
            seed,
            lane,
            gpu,
            arm,
            gain,
            "parent",
            negative_only=negative_only,
        )
        report, outcome = _run(
            root,
            runner,
            isaac_python,
            g1_usd,
            model_root,
            actor,
            seed,
            lane,
            gpu,
            arm,
            gain,
            "actor",
            negative_only=negative_only,
        )
        if parent["source_hash"] != source_hash or report["source_hash"] != source_hash:
            raise ValueError("fresh runner source drifted")
        courses.append(report["environments"][0]["course"])
        results[arm] = {
            "parent_report_hash": parent["report_hash"],
            "parent_command_audit_hash": parent_receipt["command_audit_hash"],
            "actor_report_hash": report["report_hash"],
            "high_quality": high_quality(outcome),
            **outcome,
        }
    if courses[0] != courses[1]:
        raise ValueError("fresh ball course differs across arms")
    print(f"NEGATIVE_SIDE_AUDITED seed={seed} lane={lane} gpu={gpu}", flush=True)
    return {"seed": seed, "lane": lane, "gpu": gpu, "course": courses[0], "arms": results}


def _gpu_courses(
    gpu: int, common: dict[str, Any]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rows = []
    failures = []
    for seed, lane in COURSES:
        if (lane // 2) % 4 != gpu:
            continue
        try:
            rows.append(_course(seed, lane, **common))
        except Exception as exc:
            failures.append({"seed": seed, "lane": lane, "error": str(exc)})
            print(f"NEGATIVE_SIDE_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--v286-bank", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    v286 = json.loads(args.v286_bank.read_text(encoding="utf-8"))
    if (
        args.output_root.exists()
        or v286.get("report_hash") != EXPECTED_V286_HASH
        or v286.get("development_gate_passed") is not False
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed failed development result, fresh output and qualified assets required")
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
    by_seed = []
    for seed in SEEDS:
        seed_rows = [row for row in rows if row["seed"] == seed]
        by_seed.append(
            {
                "seed": seed,
                "baseline_high_quality": sum(
                    row["arms"]["baseline"]["high_quality"] for row in seed_rows
                ),
                "candidate_high_quality": sum(
                    row["arms"]["negative_side"]["high_quality"] for row in seed_rows
                ),
            }
        )
    baseline = [row["arms"]["baseline"] for row in rows]
    candidate = [row["arms"]["negative_side"] for row in rows]
    base_high = sum(row["high_quality"] for row in baseline)
    cand_high = sum(row["high_quality"] for row in candidate)
    base_out = sum(row["maximum_lateral_excursion_m"] > 4 for row in baseline)
    cand_out = sum(row["maximum_lateral_excursion_m"] > 4 for row in candidate)
    base_reward = sum(row["reward"] for row in baseline) / max(len(baseline), 1)
    cand_reward = sum(row["reward"] for row in candidate) / max(len(candidate), 1)
    complete = len(rows) == len(COURSES) and failures == []
    safe = sum(row["clean_foot_only"] for row in candidate) >= sum(
        row["clean_foot_only"] for row in baseline
    ) and all(row["minimum_pelvis_z_m"] >= 0.65 for row in candidate)
    passed = bool(
        complete
        and safe
        and cand_high >= base_high
        and base_out >= cand_out + 1
        and cand_reward >= base_reward + 0.2
        and all(
            row["candidate_high_quality"] >= row["baseline_high_quality"] - 1 for row in by_seed
        )
    )
    result: dict[str, Any] = {
        "schema": "rsi_isaac_negative_side_approach_fresh_v1",
        "activation_ceiling": "SIM_ONLY",
        "v286_report_hash": EXPECTED_V286_HASH,
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        "complete": complete,
        "by_seed": by_seed,
        "baseline_high_quality": base_high,
        "candidate_high_quality": cand_high,
        "baseline_out_of_play": base_out,
        "candidate_out_of_play": cand_out,
        "baseline_mean_reward": base_reward,
        "candidate_mean_reward": cand_reward,
        "safety_retained": safe,
        "fresh_gate_passed": passed,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"NEGATIVE_SIDE_FRESH={result['report_hash']}", flush=True)
    print(f"FRESH_GATE_PASSED={passed}", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
