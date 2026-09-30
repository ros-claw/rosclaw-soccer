"""Retrospective seed-held-out risk-model audit; never a motor promotion gate."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.tree import DecisionTreeClassifier

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.sim.contracts import hash_json

BANKS = (
    ("v292", "sha256:8172d79326422997eeb5706e7ea554a080441d292b08dbfa48180334d9f086ae", 9),
    ("v295", "sha256:eefecf2089224345a10d4b9e45f382685c71ccc16de778a6190510667fe658f8", 17),
    ("v296", "sha256:540da31452042c69d61a1ecb598915c0f416051c422e2b637c1de19e3ea573a1", 15),
)


def new_harm(baseline: dict[str, Any], candidate: dict[str, Any]) -> bool:
    return bool(
        baseline["clean_foot_only"]
        and not candidate["clean_foot_only"]
        or baseline["maximum_lateral_excursion_m"] <= 4
        and candidate["maximum_lateral_excursion_m"] > 4
    )


def _load_rows(roots: dict[str, Path]) -> list[dict[str, Any]]:
    rows = []
    for name, sealed_hash, count in BANKS:
        root = roots[name]
        bank = json.loads((root / "bank_summary.json").read_text(encoding="utf-8"))
        if (
            bank.get("report_hash") != sealed_hash
            or sealed_hash != hash_json({k: v for k, v in bank.items() if k != "report_hash"})
            or bank.get("complete") is not True
        ):
            raise ValueError(f"sealed complete {name} bank required")
        subset = bank["courses"][:count]
        if len(subset) != count:
            raise ValueError(f"incomplete {name} course subset")
        for course in subset:
            seed, lane = course["seed"], course["lane"]
            parent_folder = root / f"seed{seed}-lane{lane}-gain_12-parent"
            parent = json.loads((parent_folder / "report.json").read_text(encoding="utf-8"))
            receipt = audit_lateral_approach(parent_folder)
            if (
                parent["report_hash"] != course["arms"]["gain_12"]["parent_report_hash"]
                or receipt["report_hash"] != course["arms"]["gain_12"]["parent_command_audit_hash"]
                or parent["environments"][0]["course"] != course["course"]
            ):
                raise ValueError(f"parent provenance drift: {name}/{seed}/{lane}")
            baseline = course["arms"]["gain_08"]
            candidate = course["arms"]["gain_12"]
            rows.append(
                {
                    "bank": name,
                    "seed": seed,
                    "lane": lane,
                    "course": course["course"],
                    "parent_report_hash": parent["report_hash"],
                    "candidate_parent_right_knee": 5
                    in parent["environments"][0]["contact_body_indices"],
                    "new_harm": new_harm(baseline, candidate),
                    "baseline_high_quality": baseline["high_quality"],
                    "candidate_high_quality": candidate["high_quality"],
                    "baseline_clean_foot_only": baseline["clean_foot_only"],
                    "candidate_clean_foot_only": candidate["clean_foot_only"],
                }
            )
    return rows


def crossvalidate(rows: list[dict[str, Any]], *, shadow_feature: bool) -> dict[str, Any]:
    x = np.asarray(
        [
            [
                row["course"]["ball_x_m"],
                row["course"]["ball_y_local_m"],
                row["course"]["ball_vx_m_s"],
                *([float(row["candidate_parent_right_knee"])] if shadow_feature else []),
            ]
            for row in rows
        ],
        dtype=np.float64,
    )
    y = np.asarray([row["new_harm"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    if len(rows) != 41 or y.sum() != 4 or len(set(groups)) < 20:
        raise ValueError("41 audited courses and four harms required")
    predicted = np.zeros(len(rows), dtype=np.int64)
    for train, test in LeaveOneGroupOut().split(x, y, groups):
        model = DecisionTreeClassifier(
            max_depth=3,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=0,
        )
        model.fit(x[train], y[train])
        predicted[test] = model.predict(x[test])
    true_positive = int(np.count_nonzero((y == 1) & (predicted == 1)))
    false_positive = int(np.count_nonzero((y == 0) & (predicted == 1)))
    false_negative = int(np.count_nonzero((y == 1) & (predicted == 0)))
    selected = [
        row["baseline_high_quality"] if veto else row["candidate_high_quality"]
        for row, veto in zip(rows, predicted, strict=True)
    ]
    return {
        "feature_set": "frame0_and_shadow_parent" if shadow_feature else "frame0_only",
        "split": "leave_one_seed_out",
        "course_count": len(rows),
        "harm_count": int(y.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "predicted_veto_count": int(predicted.sum()),
        "selected_high_quality": int(sum(selected)),
        "baseline_high_quality": int(sum(row["baseline_high_quality"] for row in rows)),
        "candidate_high_quality": int(sum(row["candidate_high_quality"] for row in rows)),
        "safety_eligible": false_negative == 0,
        "promotion_authorized": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name, _, _ in BANKS:
        parser.add_argument(f"--{name}-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    roots = {name: getattr(args, f"{name}_root") for name, _, _ in BANKS}
    rows = _load_rows(roots)
    result: dict[str, Any] = {
        "schema": "rsi_retrospective_causal_risk_distillation_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "CONSUMED_EXPLORATORY_NO_FRESH_PROMOTION",
        "source_bank_hashes": {name: sealed for name, sealed, _ in BANKS},
        "rows": rows,
        "models": [
            crossvalidate(rows, shadow_feature=False),
            crossvalidate(rows, shadow_feature=True),
        ],
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["report_hash"])
    for row in result["models"]:
        print(
            row["feature_set"], row["true_positive"], row["false_positive"], row["false_negative"]
        )


if __name__ == "__main__":
    main()
