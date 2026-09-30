"""Seed-held-out, SIM_ONLY frame-0 risk learning from audited consumed courses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_evaluate_causal_risk_distillation_v297 import new_harm

V297_HASH = "sha256:f3e74406ea3f7895981df6d1816f1d8f2197e20f95df80124b103bb6bb158499"
V298_HASH = "sha256:b0086ab12bc84957058ea76efdc0b89e50c34ad7ae3a27f490286ef3f5a04740"


def load_rows(v297_path: Path, v298_path: Path) -> list[dict[str, Any]]:
    earlier = json.loads(v297_path.read_text(encoding="utf-8"))
    bank = json.loads(v298_path.read_text(encoding="utf-8"))
    if (
        earlier.get("report_hash") != V297_HASH
        or earlier.get("report_hash")
        != hash_json({key: value for key, value in earlier.items() if key != "report_hash"})
        or bank.get("report_hash") != V298_HASH
        or bank.get("report_hash")
        != hash_json({key: value for key, value in bank.items() if key != "report_hash"})
        or bank.get("complete") is not True
        or len(earlier.get("rows", [])) != 41
        or len(bank.get("courses", [])) != 40
    ):
        raise ValueError("complete sealed v297/v298 training evidence required")
    rows = [
        {
            "source_bank": row["bank"],
            "seed": row["seed"],
            "lane": row["lane"],
            "course": row["course"],
            "new_harm": row["new_harm"],
            "baseline_high_quality": row["baseline_high_quality"],
            "candidate_high_quality": row["candidate_high_quality"],
            "baseline_clean_foot_only": row["baseline_clean_foot_only"],
            "candidate_clean_foot_only": row["candidate_clean_foot_only"],
        }
        for row in earlier["rows"]
    ]
    rows.extend(
        {
            "source_bank": "v298_train",
            "seed": row["seed"],
            "lane": row["lane"],
            "course": row["course"],
            "new_harm": new_harm(row["arms"]["gain_08"], row["arms"]["gain_12"]),
            "baseline_high_quality": row["arms"]["gain_08"]["high_quality"],
            "candidate_high_quality": row["arms"]["gain_12"]["high_quality"],
            "baseline_clean_foot_only": row["arms"]["gain_08"]["clean_foot_only"],
            "candidate_clean_foot_only": row["arms"]["gain_12"]["clean_foot_only"],
        }
        for row in bank["courses"]
    )
    if (
        len(rows) != 81
        or sum(row["new_harm"] for row in rows) != 11
        or len({(row["seed"], row["lane"]) for row in rows}) != 81
        or len({row["seed"] for row in rows}) < 60
    ):
        raise ValueError("81 independent courses with 11 physical harm labels required")
    return rows


def _models() -> dict[str, Any]:
    return {
        "tree_depth3": DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=0,
        ),
        "forest_depth4": RandomForestClassifier(
            n_estimators=100,
            max_depth=4,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=0,
            n_jobs=1,
        ),
        "logistic_c01": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight={0: 1, 1: 4},
                max_iter=1000,
                random_state=0,
            ),
        ),
    }


def crossvalidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = np.asarray(
        [
            [
                row["course"]["ball_x_m"],
                row["course"]["ball_y_local_m"],
                row["course"]["ball_vx_m_s"],
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["new_harm"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    outputs = []
    for name, model in _models().items():
        predicted = np.zeros(len(rows), dtype=np.int64)
        for train, test in LeaveOneGroupOut().split(x, y, groups):
            model.fit(x[train], y[train])
            predicted[test] = model.predict(x[test])
        true_positive = int(np.count_nonzero((y == 1) & (predicted == 1)))
        false_positive = int(np.count_nonzero((y == 0) & (predicted == 1)))
        false_negative = int(np.count_nonzero((y == 1) & (predicted == 0)))
        selected_high = sum(
            row["baseline_high_quality"] if veto else row["candidate_high_quality"]
            for row, veto in zip(rows, predicted, strict=True)
        )
        selected_clean = sum(
            row["baseline_clean_foot_only"] if veto else row["candidate_clean_foot_only"]
            for row, veto in zip(rows, predicted, strict=True)
        )
        outputs.append(
            {
                "model": name,
                "split": "leave_one_seed_out",
                "true_positive": true_positive,
                "false_positive": false_positive,
                "false_negative": false_negative,
                "selected_high_quality": selected_high,
                "selected_clean_foot_only": selected_clean,
                "baseline_high_quality": sum(row["baseline_high_quality"] for row in rows),
                "candidate_high_quality": sum(row["candidate_high_quality"] for row in rows),
                "safety_eligible_for_fresh_exam": false_negative == 0,
                "predictions": [int(value) for value in predicted],
            }
        )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v297-report", required=True, type=Path)
    parser.add_argument("--v298-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    rows = load_rows(args.v297_report, args.v298_report)
    models = crossvalidate(rows)
    result: dict[str, Any] = {
        "schema": "rsi_causal_approach_risk_learning_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "CONSUMED_DEVELOPMENT",
        "v297_report_hash": V297_HASH,
        "v298_report_hash": V298_HASH,
        "sklearn_version": sklearn.__version__,
        "rows": rows,
        "models": models,
        "fresh_exam_eligible": any(model["safety_eligible_for_fresh_exam"] for model in models),
        "policy_exported": False,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["report_hash"])
    for model in models:
        print(
            model["model"],
            model["true_positive"],
            model["false_positive"],
            model["false_negative"],
        )
    print(f"FRESH_EXAM_ELIGIBLE={result['fresh_exam_eligible']}")


if __name__ == "__main__":
    main()
