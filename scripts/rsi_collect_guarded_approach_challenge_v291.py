"""Preflight and run stratified SIM_ONLY G1 approach challenge, without outcome cherry-picking."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.conservative_approach_rectangle import load_guarded_approach_policy
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_guarded_approach_online_fresh_v290 import _course

CHALLENGE = (
    (20260973, 2),
    (20260973, 6),
    (20260974, 0),
    (20260974, 2),
    (20260975, 2),
    (20260975, 4),
    (20260976, 4),
    (20260977, 4),
    (20260978, 4),
    (20260978, 6),
    (20260980, 4),
    (20260981, 4),
    (20260982, 2),
)
PROTECTION = ((20260973, 4), (20260974, 4), (20260976, 2), (20260976, 6))
COURSES = CHALLENGE + PROTECTION
V290_HASH = "sha256:e27c927f9802b846d14ca6d74f739ef5c71830670272507654893289cb2a1ecf"


def preflight() -> str:
    chosen = []
    for seed in range(20260973, 20260983):
        catalog = sample_training_courses(seed, 16)
        for lane, (ball_x, ball_y, ball_vx) in enumerate(catalog):
            if lane % 2 == 0 and ball_vx < 0 and 2.0 <= ball_x <= 2.55 and -0.09 <= ball_y <= -0.02:
                chosen.append((seed, lane))
    if set(chosen) != set(CHALLENGE) or len(chosen) != len(CHALLENGE):
        raise ValueError("preregistered outcome-blind challenge catalog drifted")
    for seed, lane in PROTECTION:
        ball_x, ball_y, ball_vx = sample_training_courses(seed, 16)[lane]
        if lane % 2 or ball_vx >= 0 or not 2.0 <= ball_x <= 2.55 or ball_y <= 0:
            raise ValueError("preregistered positive-side protection drifted")
    return hash_json({"challenge": CHALLENGE, "protection": PROTECTION})


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
            print(f"STRATIFIED_CHALLENGE_FAILED seed={seed} lane={lane}: {exc}", flush=True)
    return rows, failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--v290-report", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--isaac-python", required=True, type=Path)
    parser.add_argument("--g1-usd", required=True, type=Path)
    parser.add_argument("--model-root", required=True, type=Path)
    parser.add_argument("--late-swing-policy", required=True, type=Path)
    args = parser.parse_args()
    runner = Path(__file__).with_name("rsi_isaac_vector_first_touch.py")
    earlier = json.loads(args.v290_report.read_text(encoding="utf-8"))
    _, policy_hash = load_guarded_approach_policy(args.policy)
    catalog_hash = preflight()
    if (
        args.output_root.exists()
        or earlier.get("report_hash") != V290_HASH
        or earlier.get("report_hash")
        != hash_json({key: value for key, value in earlier.items() if key != "report_hash"})
        or earlier.get("fresh_gate_passed") is not False
        or not all(
            path.is_file()
            for path in (runner, args.isaac_python, args.g1_usd, args.late_swing_policy)
        )
        or not args.model_root.is_dir()
    ):
        parser.error("sealed failed v290 evidence, new output and qualified assets required")
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
        policy=args.policy,
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
    hard = [row for row in rows if (row["seed"], row["lane"]) in CHALLENGE]
    protect = [row for row in rows if (row["seed"], row["lane"]) in PROTECTION]
    base = [row["arms"]["baseline"] for row in hard]
    candidate = [row["arms"]["guarded_online"] for row in hard]
    base_out = sum(row["maximum_lateral_excursion_m"] > 4 for row in base)
    cand_out = sum(row["maximum_lateral_excursion_m"] > 4 for row in candidate)
    new_out = sum(
        b["maximum_lateral_excursion_m"] <= 4 and c["maximum_lateral_excursion_m"] > 4
        for b, c in zip(base, candidate, strict=True)
    )
    high_gain = sum(c["high_quality"] for c in candidate) - sum(b["high_quality"] for b in base)
    reward_gain = sum(
        c["reward"] - b["reward"] for b, c in zip(base, candidate, strict=True)
    ) / max(len(hard), 1)
    protection_passed = all(
        (b := row["arms"]["baseline"])["contact_body_indices"]
        == (c := row["arms"]["guarded_online"])["contact_body_indices"]
        and b["clean_foot_only"] == c["clean_foot_only"]
        and b["high_quality"] == c["high_quality"]
        and b["first_contact_frame"] == c["first_contact_frame"]
        and b["forward_60_m"] is not None
        and c["forward_60_m"] is not None
        and b["lateral_60_m"] is not None
        and c["lateral_60_m"] is not None
        and abs(b["forward_60_m"] - c["forward_60_m"]) <= 0.01
        and abs(b["lateral_60_m"] - c["lateral_60_m"]) <= 0.01
        and c["command_active_frames"] == 0
        and c["minimum_pelvis_z_m"] >= 0.65
        for row in protect
    )
    complete = len(rows) == len(COURSES) and failures == []
    hard_passed = bool(
        len(hard) == len(CHALLENGE)
        and all(c["command_active_frames"] > 0 for c in candidate)
        and base_out >= 2
        and cand_out <= base_out - 2
        and new_out == 0
        and high_gain >= 3
        and sum(c["clean_foot_only"] for c in candidate) >= sum(b["clean_foot_only"] for b in base)
        and all(c["minimum_pelvis_z_m"] >= 0.65 for c in candidate)
        and reward_gain >= 0.5
    )
    result: dict[str, Any] = {
        "schema": "rsi_isaac_guarded_approach_stratified_challenge_v1",
        "activation_ceiling": "SIM_ONLY",
        "v290_report_hash": V290_HASH,
        "policy_hash": policy_hash,
        "course_selection_hash": catalog_hash,
        "source_hash": source_hash,
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "courses": rows,
        "failures": failures,
        "complete": complete,
        "hard_baseline_out_of_play": base_out,
        "hard_candidate_out_of_play": cand_out,
        "hard_new_out_of_play": new_out,
        "hard_high_quality_gain": high_gain,
        "hard_mean_reward_gain": reward_gain,
        "hard_gate_passed": hard_passed,
        "protection_gate_passed": len(protect) == len(PROTECTION) and protection_passed,
        "development_gate_passed": complete
        and hard_passed
        and len(protect) == len(PROTECTION)
        and protection_passed,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (args.output_root / "bank_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(f"STRATIFIED_CHALLENGE={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)
    if not complete:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
