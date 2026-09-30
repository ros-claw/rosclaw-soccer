"""Export sealed logistic development policy for SIM_ONLY switch physics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rosclaw_soccer.rsi.precontact_proprio_policy import FEATURE_NAMES, V300_HASH, validate_policy
from rosclaw_soccer.sim.contracts import hash_json


def export(report: dict[str, Any]) -> dict[str, Any]:
    if (
        report.get("report_hash") != V300_HASH
        or report.get("report_hash")
        != hash_json({key: value for key, value in report.items() if key != "report_hash"})
        or report.get("switch_test_eligible") is not True
        or report.get("feature_names") != list(FEATURE_NAMES)
        or len(report.get("rows", [])) != 81
        or not any(
            row["model"] == "proprio_logistic_c01"
            and row["threshold"] == 0.2
            and row["false_negative"] == 0
            for row in report["models"]
        )
    ):
        raise ValueError("sealed no-miss precontact development evidence required")
    x = np.asarray([row["features"] for row in report["rows"]], dtype=np.float64)
    y = np.asarray([row["new_harm"] for row in report["rows"]], dtype=np.int64)
    if x.shape != (81, 13) or int(y.sum()) != 11 or not np.isfinite(x).all():
        raise ValueError("complete finite training bank required")
    fitted = make_pipeline(
        StandardScaler(),
        LogisticRegression(C=0.1, class_weight={0: 1, 1: 4}, max_iter=1000, random_state=0),
    ).fit(x, y)
    scaler = fitted.named_steps["standardscaler"]
    classifier = fitted.named_steps["logisticregression"]
    policy: dict[str, Any] = {
        "schema": "rsi_precontact_proprio_approach_policy_v1",
        "activation_ceiling": "SIM_ONLY",
        "training_report_hash": V300_HASH,
        "feature_names": list(FEATURE_NAMES),
        "decision_frame": 30,
        "threshold": 0.2,
        "aggressive_gain": 1.2,
        "fallback_gain": 0.8,
        "mean": [float(value) for value in scaler.mean_],
        "scale": [float(value) for value in scaler.scale_],
        "coefficients": [float(value) for value in classifier.coef_[0]],
        "intercept": float(classifier.intercept_[0]),
        "promotion_authorized": False,
        "video_authorized": False,
    }
    policy["policy_hash"] = hash_json(policy)
    validate_policy(policy)
    return policy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-report", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    report = json.loads(args.training_report.read_text(encoding="utf-8"))
    policy = export(report)
    args.output.write_text(json.dumps(policy, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(policy["policy_hash"])


if __name__ == "__main__":
    main()
