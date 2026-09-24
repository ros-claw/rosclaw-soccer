"""Fit continuous right-foot approach from v2 failures and audited physics teachers."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sonic_contact_selector import (
    SCHEMA_V2,
    SCHEMA_V3,
    choose_lateral,
    load_selector,
)
from rosclaw_soccer.rsi.verify_sonic_ball_goal import _first_physics_contact, _read
from rosclaw_soccer.rsi.verify_sonic_contact_holdout import verify_holdout
from rosclaw_soccer.sim.contracts import hash_json


def fit(
    *,
    parent_model: Path,
    failed_holdout: Path,
    teachers: tuple[Path, Path, Path, Path],
    stadium_assets: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("new v3 checkpoint path required")
    parent = load_selector(parent_model)
    failure = verify_holdout(failed_holdout, stadium_assets=stadium_assets)
    if (
        parent["schema"] != SCHEMA_V2
        or failure["selector_hash"] != parent["model_hash"]
        or failure["candidate_foot_goals"] != 2
        or failure["results"][0]["ball_xy_m"][0] != 2.0
    ):
        raise ValueError("v3 requires the pinned failed 2.0 m v2 lineage")
    knot_values = []
    teacher_hashes = []
    for expected_y, path in zip((0.04, 0.08, 0.12, 0.16), teachers, strict=True):
        report, arrays = _read(path)
        first = _first_physics_contact(
            arrays["physics_qpos"], stadium_assets, report["physics_hash"]
        )
        q = arrays["qpos"]
        radius = float(report["ball_radius_m"])
        goal = np.flatnonzero(
            (q[:, 36] - radius >= 5.0)
            & (np.abs(q[:, 37]) + radius <= 1.2)
            & (q[:, 38] + radius <= 1.6)
        )
        if (
            report["partition"] not in ("DISCOVERY", "FRESH")
            or report["ball_initial_xy_m"] != [2.0, expected_y]
            or report["left_contact_residual_rad"] != [-0.15, 0.15, 0.0]
            or first is None
            or not first["is_foot"]
            or not first["geom"].startswith("right_foot")
            or not len(goal)
            or int(goal[0]) != report["goal_frame"]
            or report["promotion_authorized"]
        ):
            raise ValueError("continuous teacher lacks independently reconstructed right-foot goal")
        knot_values.append([expected_y, float(report["run_lateral_mps"])])
        teacher_hashes.append(hash_json(report))
    if [knot[1] for knot in knot_values] != [0.04, 0.06, 0.1, 0.1]:
        raise ValueError("unexpected v3 teacher action sequence")
    model: dict[str, Any] = {
        "schema": SCHEMA_V3,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "trained_actor": False,
        "learner": "phase_conditioned_piecewise_right_foot_teacher_distillation",
        "threshold_y_m": float(parent["threshold_y_m"]),
        "far_distance_switch_x_m": float(parent["far_distance_switch_x_m"]),
        "far_right_foot_switch_y_m": float(parent["far_right_foot_switch_y_m"]),
        "continuous_right_foot_start_x_m": 1.95,
        "continuous_right_foot_knots": knot_values,
        "contact_residual_rad": [-0.15, 0.15, 0.0],
        "parent_selector_hash": parent["model_hash"],
        "failed_holdout_verification_hash": failure["verification_hash"],
        "right_foot_teacher_report_hashes": sorted(teacher_hashes),
    }
    model["model_hash"] = hash_json(model)
    if [choose_lateral(model, ball_x_m=2.0, ball_y_m=y) for y in (0.04, 0.08, 0.12, 0.16)] != [
        0.04,
        0.06,
        0.1,
        0.1,
    ]:
        raise ValueError("v3 interpolation does not match physical teachers")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(model, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--failed-holdout", required=True, type=Path)
    parser.add_argument("--teachers", required=True, nargs=4, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            fit(
                parent_model=args.parent_model,
                failed_holdout=args.failed_holdout,
                teachers=tuple(args.teachers),
                stadium_assets=args.stadium_assets,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
