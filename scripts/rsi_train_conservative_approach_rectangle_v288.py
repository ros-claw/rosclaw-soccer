"""Grouped cross-validation of an abstaining, SIM_ONLY G1 approach gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.conservative_approach_rectangle import fit_approach_rectangle
from rosclaw_soccer.sim.contracts import hash_json

EXPECTED = {
    "v286": "sha256:a97b1b650acbf0c2cce75ce4b0de0eebab68b2eaf429709e439fec36106cd06c",
    "v287": "sha256:b828f167c3b85c95728f32a86bf86f987e5a4bec9b6d387b51e4fdcd914190dc",
}


def high_quality(row: dict[str, Any]) -> bool:
    forward = row["forward_60_m"]
    lateral = row["lateral_60_m"]
    return bool(
        row["clean_foot_only"]
        and forward is not None
        and lateral is not None
        and forward >= 1.0
        and abs(lateral) / max(forward, 0.01) <= 0.3
        and row["maximum_lateral_excursion_m"] <= 4.0
    )


def load_courses(roots: dict[str, Path]) -> list[dict[str, Any]]:
    courses = []
    for label, root in roots.items():
        bank = json.loads((root / "bank_summary.json").read_text(encoding="utf-8"))
        if (
            bank.get("report_hash") != EXPECTED[label]
            or bank.get("report_hash")
            != hash_json({key: value for key, value in bank.items() if key != "report_hash"})
            or bank.get("failures") != []
            or bank.get("complete") is not True
            or len(bank.get("courses", [])) != (11 if label == "v286" else 32)
        ):
            raise ValueError(f"incomplete or drifted {label} development bank")
        candidate_name = "lateral_tracking" if label == "v286" else "negative_side"
        for course in bank["courses"]:
            contexts = []
            for arm in ("baseline", candidate_name):
                folder = root / f"seed{course['seed']}-lane{course['lane']}-{arm}-actor"
                report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
                audit_lateral_approach(folder)
                if (
                    report["report_hash"] != course["arms"][arm]["actor_report_hash"]
                    or report["source_hash"] != bank["source_hash"]
                    or report["asset_hash"] != bank["asset_hash"]
                    or report["environments"][0]["course"] != course["course"]
                ):
                    raise ValueError(f"drifted independent physical arm: {folder}")
                with np.load(folder / "body_trace.npz", allow_pickle=False) as trace:
                    root_pose = trace["root_pose_xyzw_m"][0, 0]
                    ball = trace["ball_position_before_step_m"][0, 0]
                    contexts.append(np.asarray((ball[0] - root_pose[0], ball[1] - root_pose[1])))
            if not np.allclose(contexts[0], contexts[1], atol=1e-8, rtol=0):
                raise ValueError("counterfactual frame-zero state drifted")
            base = course["arms"]["baseline"]
            candidate = course["arms"][candidate_name]
            courses.append(
                {
                    "seed": course["seed"],
                    "lane": course["lane"],
                    "source_bank": label,
                    "frame_zero_context_m": contexts[0].tolist(),
                    "baseline_reward": base["reward"],
                    "candidate_reward": candidate["reward"],
                    "baseline_clean": base["clean_foot_only"],
                    "candidate_clean": candidate["clean_foot_only"],
                    "baseline_out": base["maximum_lateral_excursion_m"] > 4,
                    "candidate_out": candidate["maximum_lateral_excursion_m"] > 4,
                    "baseline_high_quality": high_quality(base),
                    "candidate_high_quality": high_quality(candidate),
                }
            )
    if len({(row["seed"], row["lane"]) for row in courses}) != len(courses):
        raise ValueError("duplicate physical course across development banks")
    return courses


def cross_validate(
    courses: list[dict[str, Any]], *, x_margin_m: float = 0.0, y_margin_m: float = 0.0
) -> dict[str, Any]:
    x = np.asarray([row["frame_zero_context_m"] for row in courses], dtype=float)
    gain = np.asarray([row["candidate_reward"] - row["baseline_reward"] for row in courses])
    clean_loss = np.asarray(
        [row["baseline_clean"] and not row["candidate_clean"] for row in courses]
    )
    new_out = np.asarray([not row["baseline_out"] and row["candidate_out"] for row in courses])
    folds = []
    all_selected: list[tuple[dict[str, Any], bool]] = []
    for seed in sorted({row["seed"] for row in courses}):
        test = np.asarray([row["seed"] == seed for row in courses])
        model = fit_approach_rectangle(x[~test], gain[~test], clean_loss[~test], new_out[~test])
        choices = [
            model.choose(*x[i], x_margin_m=x_margin_m, y_margin_m=y_margin_m)
            for i in np.flatnonzero(test)
        ]
        selected_rows = [row for row in courses if row["seed"] == seed]
        fold_gain = sum(
            row["candidate_reward"] - row["baseline_reward"] if chosen else 0.0
            for row, chosen in zip(selected_rows, choices, strict=True)
        )
        clean_losses = sum(
            row["baseline_clean"] and not row["candidate_clean"]
            for row, chosen in zip(selected_rows, choices, strict=True)
            if chosen
        )
        new_outs = sum(
            not row["baseline_out"] and row["candidate_out"]
            for row, chosen in zip(selected_rows, choices, strict=True)
            if chosen
        )
        high_gain = sum(
            int(row["candidate_high_quality"]) - int(row["baseline_high_quality"])
            for row, chosen in zip(selected_rows, choices, strict=True)
            if chosen
        )
        fold = {
            "test_seed": seed,
            "train_x_max_m": model.x_max_m,
            "train_y_min_m": model.y_min_m,
            "train_support": model.support,
            "selected_count": sum(choices),
            "reward_gain_sum": fold_gain,
            "clean_losses": clean_losses,
            "new_out_of_play": new_outs,
            "high_quality_gain": high_gain,
            "selected_course_lanes": [
                row["lane"] for row, chosen in zip(selected_rows, choices, strict=True) if chosen
            ],
        }
        folds.append(fold)
        all_selected.extend(
            (row, chosen) for row, chosen in zip(selected_rows, choices, strict=True)
        )
    selected_count = sum(chosen for _, chosen in all_selected)
    high_gain = sum(fold["high_quality_gain"] for fold in folds)
    gain_sum = sum(fold["reward_gain_sum"] for fold in folds)
    clean_safe = all(fold["clean_losses"] == 0 for fold in folds)
    out_safe = all(fold["new_out_of_play"] == 0 for fold in folds)
    wins = sum(fold["reward_gain_sum"] > 1e-9 for fold in folds)
    passed = bool(
        clean_safe
        and out_safe
        and high_gain >= 2
        and gain_sum / len(courses) >= 0.2
        and wins >= 4
        and selected_count >= 4
    )
    full = fit_approach_rectangle(x, gain, clean_loss, new_out)
    result: dict[str, Any] = {
        "schema": (
            "rsi_isaac_guarded_approach_margin_cv_v1"
            if x_margin_m or y_margin_m
            else "rsi_isaac_learned_approach_rectangle_cv_v1"
        ),
        "activation_ceiling": "SIM_ONLY",
        "input_report_hashes": EXPECTED,
        "episode_count": len(courses),
        "folds": folds,
        "selected_count": selected_count,
        "high_quality_gain": high_gain,
        "reward_gain_mean": gain_sum / len(courses),
        "strict_positive_seed_folds": wins,
        "all_clean_safe": clean_safe,
        "all_out_of_play_safe": out_safe,
        "development_gate_passed": passed,
        "full_fit_rectangle": {
            "x_max_m": full.x_max_m,
            "y_min_m": full.y_min_m,
            "support": full.support,
            "training_gain": full.training_gain,
        },
        "fresh_seed_71_72_authorized": passed,
        "promotion_authorized": False,
    }
    if x_margin_m or y_margin_m:
        result["x_margin_m"] = x_margin_m
        result["y_margin_m"] = y_margin_m
        result["effective_full_fit_rectangle"] = {
            "x_max_m": None if full.x_max_m is None else full.x_max_m - x_margin_m,
            "y_min_m": None if full.y_min_m is None else full.y_min_m + y_margin_m,
        }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v286-root", required=True, type=Path)
    parser.add_argument("--v287-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new CV output required")
    courses = load_courses({"v286": args.v286_root, "v287": args.v287_root})
    result = cross_validate(courses)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"APPROACH_RECTANGLE_CV={result['report_hash']}", flush=True)
    print(f"DEVELOPMENT_GATE_PASSED={result['development_gate_passed']}", flush=True)


if __name__ == "__main__":
    main()
