"""Distill one audited failure correction into a bounded right-foot selector."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sonic_contact_selector import SCHEMA_V3, SCHEMA_V4, load_selector
from rosclaw_soccer.rsi.verify_sonic_ball_goal import _first_physics_contact, _read
from rosclaw_soccer.rsi.verify_sonic_contact_holdout import verify_holdout
from rosclaw_soccer.sim.contracts import hash_json


def fit(
    *, parent_model: Path, failed_holdout: Path, teacher: Path, stadium_assets: Path, output: Path
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("new v4 checkpoint path required")
    parent = load_selector(parent_model)
    failure = verify_holdout(failed_holdout, stadium_assets=stadium_assets)
    if (
        parent["schema"] != SCHEMA_V3
        or failure["selector_hash"] != parent["model_hash"]
        or failure["candidate_foot_goals"] != 3
        or failure["results"][2]["ball_xy_m"] != [2.1, 0.09]
        or failure["results"][3]["foot_first_goal"]
    ):
        raise ValueError("v4 requires pinned 2.1 m v3 failure lineage")
    report, arrays = _read(teacher)
    first = _first_physics_contact(arrays["physics_qpos"], stadium_assets, report["physics_hash"])
    q = arrays["qpos"]
    radius = float(report["ball_radius_m"])
    goal = np.flatnonzero(
        (q[:, 36] - radius >= 5.0) & (np.abs(q[:, 37]) + radius <= 1.2) & (q[:, 38] + radius <= 1.6)
    )
    if (
        report["partition"] != "DISCOVERY"
        or report["ball_initial_xy_m"] != [2.1, 0.09]
        or report["run_lateral_mps"] != 0.09
        or report["left_contact_residual_rad"] != [-0.15, 0.15, 0.0]
        or first is None
        or not first["is_foot"]
        or not first["geom"].startswith("right_foot")
        or not len(goal)
        or int(goal[0]) != report["goal_frame"]
        or report["promotion_authorized"]
    ):
        raise ValueError("v4 teacher lacks independently reconstructed right-foot goal")
    model: dict[str, Any] = {
        **{key: value for key, value in parent.items() if key != "model_hash"},
        "schema": SCHEMA_V4,
        "learner": "failure_conditioned_right_foot_knot_distillation",
        "continuous_right_foot_knots": [
            [0.04, 0.04],
            [0.08, 0.06],
            [0.09, 0.09],
            [0.12, 0.1],
            [0.16, 0.1],
        ],
        "parent_selector_hash": parent["model_hash"],
        "failed_holdout_verification_hash": failure["verification_hash"],
        "right_foot_correction_report_hash": hash_json(report),
    }
    model["model_hash"] = hash_json(model)
    with output.open("x", encoding="utf-8") as stream:
        json.dump(model, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--failed-holdout", required=True, type=Path)
    parser.add_argument("--teacher", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(fit(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
