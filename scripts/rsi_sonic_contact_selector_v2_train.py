"""Distill a phase-aware v2 selector from a failed far holdout and right-foot fixes.

The old far holdout becomes development evidence for this NEW generation only.
No previous FRESH score may be re-used to evaluate v2.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.sonic_contact_selector import SCHEMA_V2, choose_lateral, load_selector
from rosclaw_soccer.rsi.verify_sonic_ball_goal import _first_physics_contact, _read
from rosclaw_soccer.rsi.verify_sonic_contact_holdout import verify_holdout
from rosclaw_soccer.sim.contracts import hash_json


def fit(
    *,
    parent_model: Path,
    near_holdout: Path,
    far_holdout: Path,
    right_foot_teachers: tuple[Path, Path],
    stadium_assets: Path,
    output: Path,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError("new v2 checkpoint path required")
    parent = load_selector(parent_model)
    near = verify_holdout(near_holdout, stadium_assets=stadium_assets)
    far = verify_holdout(far_holdout, stadium_assets=stadium_assets)
    if (
        parent["schema"] != "rosclaw_soccer.rsi.sonic_contact_selector.v1"
        or near["selector_hash"] != parent["model_hash"]
        or far["selector_hash"] != parent["model_hash"]
        or near["candidate_foot_goals"] != 4
        or far["candidate_foot_goals"] != 2
    ):
        raise ValueError("v2 requires the pinned v1 success/failure lineage")
    teacher_hashes = []
    for expected_y, path in zip((0.12, 0.16), right_foot_teachers, strict=True):
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
            report["partition"] != "DISCOVERY"
            or report["ball_initial_xy_m"] != [1.9, expected_y]
            or report["run_lateral_mps"] != 0.1
            or report["left_contact_residual_rad"] != [-0.15, 0.15, 0.0]
            or first is None
            or not first["is_foot"]
            or not first["geom"].startswith("right_foot")
            or not len(goal)
            or int(goal[0]) != report["goal_frame"]
            or report["promotion_authorized"]
        ):
            raise ValueError("right-foot teacher lacks independently reconstructed goal")
        teacher_hashes.append(hash_json(report))
    model = {
        "schema": SCHEMA_V2,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "trained_actor": False,
        "learner": "two_split_left_right_foot_goal_teacher_distillation",
        "threshold_y_m": float(parent["threshold_y_m"]),
        "far_distance_switch_x_m": 1.85,
        "far_right_foot_switch_y_m": 0.10,
        "contact_residual_rad": [-0.15, 0.15, 0.0],
        "parent_selector_hash": parent["model_hash"],
        "near_holdout_verification_hash": near["verification_hash"],
        "failed_far_holdout_verification_hash": far["verification_hash"],
        "right_foot_teacher_report_hashes": sorted(teacher_hashes),
    }
    model["model_hash"] = hash_json(model)
    if (
        choose_lateral(model, ball_x_m=1.9, ball_y_m=0.04) != -0.1
        or choose_lateral(model, ball_x_m=1.9, ball_y_m=0.12) != 0.1
        or choose_lateral(model, ball_x_m=1.8, ball_y_m=0.12) != 0.0
    ):
        raise ValueError("phase-aware v2 policy does not match audited physical teachers")
    with output.open("x", encoding="utf-8") as stream:
        json.dump(model, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-model", required=True, type=Path)
    parser.add_argument("--near-holdout", required=True, type=Path)
    parser.add_argument("--far-holdout", required=True, type=Path)
    parser.add_argument("--right-foot-teachers", required=True, nargs=2, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    print(
        json.dumps(
            fit(
                parent_model=args.parent_model,
                near_holdout=args.near_holdout,
                far_holdout=args.far_holdout,
                right_foot_teachers=tuple(args.right_foot_teachers),
                stadium_assets=args.stadium_assets,
                output=args.output,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
