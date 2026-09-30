"""Audit precontact proprioception and test seed-held-out risk prediction."""

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

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.sim.contracts import hash_json

V299_HASH = "sha256:21d3173eddcbe24372aba772708b12106c180b0efa5c3a8e9dc0353bafd7e2ca"
FEATURE_NAMES = (
    "ball_minus_root_x_m",
    "ball_minus_root_y_m",
    "ball_vx_m_s",
    "root_vx_m_s",
    "root_vy_m_s",
    "left_foot_ball_distance_m",
    "right_foot_ball_distance_m",
    "left_knee_ball_distance_m",
    "right_knee_ball_distance_m",
    "left_foot_height_m",
    "right_foot_height_m",
    "left_knee_ball_closing_m_s",
    "right_knee_ball_closing_m_s",
)


def precontact_features(folder: Path, report: dict[str, Any]) -> tuple[float, ...]:
    first = report["environments"][0]["first_contact_frame"]
    if first is not None and first <= 30:
        raise ValueError("frame 30 would observe or follow physical contact")
    with np.load(folder / "body_trace.npz", allow_pickle=False) as archive:
        root = archive["root_pose_xyzw_m"][30, 0]
        velocity = archive["root_velocity_world"][30, 0]
        ball = archive["ball_position_before_step_m"][[20, 30], 0]
        ball_velocity = archive["ball_linear_velocity_before_step_m_s"][30, 0]
        feet_knees = archive["foot_geometry_position_before_step_m"][[20, 30], 0]
    distance20 = np.linalg.norm(feet_knees[0] - ball[0], axis=1)
    distance30 = np.linalg.norm(feet_knees[1] - ball[1], axis=1)
    features = (
        float(ball[1, 0] - root[0]),
        float(ball[1, 1] - root[1]),
        float(ball_velocity[0]),
        float(velocity[0]),
        float(velocity[1]),
        *(float(value) for value in distance30),
        float(feet_knees[1, 0, 2]),
        float(feet_knees[1, 1, 2]),
        float((distance20[2] - distance30[2]) / 0.2),
        float((distance20[3] - distance30[3]) / 0.2),
    )
    if len(features) != len(FEATURE_NAMES) or not np.isfinite(features).all():
        raise ValueError("finite thirteen-feature causal proprioception required")
    return features


def load_rows(v299: Path, roots: dict[str, Path]) -> list[dict[str, Any]]:
    earlier = json.loads(v299.read_text(encoding="utf-8"))
    if (
        earlier.get("report_hash") != V299_HASH
        or earlier.get("report_hash")
        != hash_json({key: value for key, value in earlier.items() if key != "report_hash"})
        or len(earlier.get("rows", [])) != 81
        or earlier.get("fresh_exam_eligible") is not False
    ):
        raise ValueError("sealed failed frame-0 model report required")
    banks = {}
    for name, root in roots.items():
        bank = json.loads((root / "bank_summary.json").read_text(encoding="utf-8"))
        if (
            bank.get("report_hash")
            != hash_json({key: value for key, value in bank.items() if key != "report_hash"})
            or bank.get("complete") is not True
        ):
            raise ValueError(f"unauthenticated {name} physical bank")
        banks[name] = {(row["seed"], row["lane"]): row for row in bank["courses"]}
    rows = []
    for row in earlier["rows"]:
        bank_name = row["source_bank"]
        root = roots[bank_name]
        source = banks[bank_name][(row["seed"], row["lane"])]
        folder = root / f"seed{row['seed']}-lane{row['lane']}-gain_12-parent"
        report = json.loads((folder / "report.json").read_text(encoding="utf-8"))
        receipt = audit_lateral_approach(folder)
        if (
            report["report_hash"] != source["arms"]["gain_12"]["parent_report_hash"]
            or receipt["report_hash"] != source["arms"]["gain_12"]["parent_command_audit_hash"]
            or report["environments"][0]["course"] != row["course"]
        ):
            raise ValueError("candidate parent provenance drifted")
        rows.append(
            {
                **row,
                "parent_report_hash": report["report_hash"],
                "features": precontact_features(folder, report),
            }
        )
    if sum(row["new_harm"] for row in rows) != 11:
        raise ValueError("risk labels drifted")
    return rows


def crossvalidate(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    x = np.asarray([row["features"] for row in rows], dtype=np.float64)
    y = np.asarray([row["new_harm"] for row in rows], dtype=np.int64)
    groups = np.asarray([row["seed"] for row in rows], dtype=np.int64)
    models = {
        "proprio_logistic_c01": make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=0.1,
                class_weight={0: 1, 1: 4},
                max_iter=1000,
                random_state=0,
            ),
        ),
        "proprio_forest_depth4": RandomForestClassifier(
            n_estimators=200,
            max_depth=4,
            min_samples_leaf=2,
            class_weight={0: 1, 1: 4},
            random_state=0,
            n_jobs=1,
        ),
    }
    outputs = []
    for name, model in models.items():
        probability = np.zeros(len(rows), dtype=np.float64)
        for train, test in LeaveOneGroupOut().split(x, y, groups):
            model.fit(x[train], y[train])
            probability[test] = model.predict_proba(x[test])[:, 1]
        for threshold in (0.2, 0.5):
            veto = probability >= threshold
            outputs.append(
                {
                    "model": name,
                    "threshold": threshold,
                    "split": "leave_one_seed_out",
                    "true_positive": int(np.count_nonzero((y == 1) & veto)),
                    "false_positive": int(np.count_nonzero((y == 0) & veto)),
                    "false_negative": int(np.count_nonzero((y == 1) & ~veto)),
                    "selected_high_quality": sum(
                        row["baseline_high_quality"] if choice else row["candidate_high_quality"]
                        for row, choice in zip(rows, veto, strict=True)
                    ),
                    "selected_clean_foot_only": sum(
                        row["baseline_clean_foot_only"]
                        if choice
                        else row["candidate_clean_foot_only"]
                        for row, choice in zip(rows, veto, strict=True)
                    ),
                    "precontact_switch_physics_proven": False,
                    "safety_eligible_for_switch_test": bool(
                        np.count_nonzero((y == 1) & ~veto) == 0
                    ),
                    "probability": [float(value) for value in probability],
                }
            )
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v299-report", required=True, type=Path)
    for name in ("v292", "v295", "v296", "v298_train"):
        parser.add_argument(f"--{name.replace('_', '-')}-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("new output required")
    roots = {name: getattr(args, f"{name}_root") for name in ("v292", "v295", "v296", "v298_train")}
    rows = load_rows(args.v299_report, roots)
    results = crossvalidate(rows)
    result: dict[str, Any] = {
        "schema": "rsi_precontact_proprio_risk_learning_v1",
        "activation_ceiling": "SIM_ONLY",
        "data_status": "CONSUMED_DEVELOPMENT",
        "v299_report_hash": V299_HASH,
        "sklearn_version": sklearn.__version__,
        "feature_names": FEATURE_NAMES,
        "rows": rows,
        "models": results,
        "switch_test_eligible": any(row["safety_eligible_for_switch_test"] for row in results),
        "policy_exported": False,
        "promotion_authorized": False,
        "video_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(result["report_hash"])
    for row in results:
        print(
            row["model"],
            row["threshold"],
            row["true_positive"],
            row["false_positive"],
            row["false_negative"],
        )
    print(f"SWITCH_TEST_ELIGIBLE={result['switch_test_eligible']}")


if __name__ == "__main__":
    main()
