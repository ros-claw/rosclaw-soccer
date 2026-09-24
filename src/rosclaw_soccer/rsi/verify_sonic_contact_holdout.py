"""Independently audit a paired SONIC contact-selector FRESH physical exam."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.verify_sonic_ball_goal import _first_physics_contact, _read
from rosclaw_soccer.sim.contracts import hash_json


def verify_holdout(root: Path, *, stadium_assets: Path) -> dict[str, Any]:
    path = root.expanduser().resolve()
    manifest: dict[str, Any] = json.loads((path / "manifest.json").read_text(encoding="utf-8"))
    claimed_hash = manifest.pop("manifest_hash", None)
    if (
        claimed_hash != hash_json(manifest)
        or manifest.get("schema") != "rosclaw_soccer.rsi.sonic_contact_selector_holdout.v1"
        or manifest.get("partition") != "FRESH"
        or manifest.get("physical_execution_count") != 8
        or manifest.get("courses")
        not in (
            *([[x, y] for y in (0.04, 0.08, 0.12, 0.16)] for x in (1.8, 1.9, 2.0)),
            [[2.1, y] for y in (0.05, 0.09, 0.13, 0.17)],
            [[2.15, y] for y in (0.055, 0.095, 0.135, 0.175)],
        )
        or manifest.get("promotion_authorized") is not False
    ):
        raise ValueError("fresh holdout manifest contract failed")
    reports: list[dict[str, Any]] = []
    results = []
    for x, y in manifest["courses"]:
        pair = []
        for candidate in (False, True):
            mode = "candidate" if candidate else "parent"
            course = path / f"x{round(1000 * x):04d}-y{round(1000 * y):04d}-{mode}"
            report, arrays = _read(course)
            if (
                report.get("partition") != "FRESH"
                or report.get("activation_ceiling") != "SIM_ONLY"
                or report.get("promotion_authorized") is not False
                or report.get("source_hash") != manifest["probe_hash"]
                or report.get("selector_hash") != (manifest["selector_hash"] if candidate else None)
                or report.get("ball_initial_xy_m") != [x, y]
                or report.get("left_contact_residual_rad")
                != ([-0.15, 0.15, 0.0] if candidate else [0.0, 0.0, 0.0])
                or not np.array_equal(arrays["physics_qpos"][9::10], arrays["qpos"])
            ):
                raise ValueError("fresh course state, source or policy provenance differs")
            q = arrays["qpos"]
            v = arrays["qvel"]
            radius = float(report["ball_radius_m"])
            goal = np.flatnonzero(
                (q[:, 36] - radius >= 5.0)
                & (np.abs(q[:, 37]) + radius <= 1.2)
                & (q[:, 38] + radius <= 1.6)
            )
            goal_frame = int(goal[0]) if len(goal) else None
            contact = _first_physics_contact(
                arrays["physics_qpos"], stadium_assets, report["physics_hash"]
            )
            claim = report["first_robot_ball_contact"]
            peak_speed = float(np.linalg.norm(v[:, 35:38], axis=1).max())
            pelvis_min = float(q[:, 2].min())
            if (
                contact is None
                or claim is None
                or contact["geom"] != claim["geom"]
                or abs(contact["time_sec"] - claim["time_sec"]) > 0.006
                or contact["is_foot"] is not report["first_robot_ball_contact_is_foot"]
                or goal_frame != report["goal_frame"]
                or (goal_frame is not None) is not report["whole_ball_goal_crossed"]
                or not math.isclose(peak_speed, report["peak_ball_speed_mps"], abs_tol=1e-12)
                or not math.isclose(pelvis_min, report["minimum_pelvis_height_m"], abs_tol=1e-12)
                or pelvis_min < 0.55
            ):
                raise ValueError("fresh physical contact, goal or body claim differs")
            foot_goal = bool(contact["is_foot"] and goal_frame is not None)
            row = {
                "ball_xy_m": [x, y],
                "candidate": candidate,
                "first_contact_geom": contact["geom"],
                "foot_first_goal": foot_goal,
                "goal_frame": goal_frame,
                "peak_ball_speed_mps": peak_speed,
                "minimum_pelvis_height_m": pelvis_min,
            }
            reports.append(report)
            results.append(row)
            pair.append(report)
        if pair[0]["initial_state_hash"] != pair[1]["initial_state_hash"]:
            raise ValueError("parent and candidate did not share an initial physical state")
    if (
        sorted(hash_json(report) for report in reports) != manifest["report_hashes"]
        or sum(row["foot_first_goal"] for row in results if not row["candidate"])
        != manifest["parent_foot_goals"]
        or sum(row["foot_first_goal"] for row in results if row["candidate"])
        != manifest["candidate_foot_goals"]
    ):
        raise ValueError("fresh result totals or report hashes differ")
    verification = {
        "schema": "rosclaw_soccer.rsi.sonic_contact_selector_holdout_verification.v1",
        "partition": "FRESH",
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "selector_hash": manifest["selector_hash"],
        "physical_execution_count": 8,
        "parent_foot_goals": manifest["parent_foot_goals"],
        "candidate_foot_goals": manifest["candidate_foot_goals"],
        "contact_independently_reconstructed": True,
        "manifest_hash": claimed_hash,
        "results": results,
    }
    verification["verification_hash"] = hash_json(verification)
    return verification


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--stadium-assets", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(verify_holdout(args.root, stadium_assets=args.stadium_assets), indent=2))


if __name__ == "__main__":
    main()
