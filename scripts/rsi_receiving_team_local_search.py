"""SIM_ONLY bounded ankle-torque course search on a locally fidelity-tested 8-G1 proxy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_team_short_replay import replay

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

GRID = (-1.0, -0.75, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75, 1.0)


def train(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    qualified_probe: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.parents[1]):
        raise ValueError("new external SIM_ONLY proxy-search directory required")
    gate: dict[str, Any] = json.loads(fidelity.read_text(encoding="utf-8"))
    gate_hash = gate.pop("report_hash")
    if (
        gate_hash != hash_json(gate)
        or gate["schema"] != "rosclaw_soccer.rsi.receiving_team_probe_fidelity.v1"
        or gate["local_training_proxy_fidelity_passed"] is not True
        or gate["controlled_reception"] is not False
        or gate["promotion_authorized"] is not False
    ):
        raise ValueError("sealed qualified local proxy gate required")
    prior: dict[str, Any] = json.loads(qualified_probe.read_text(encoding="utf-8"))
    prior_hash = prior.pop("report_hash")
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    capture_hash = capture.pop("report_hash")
    if (
        prior_hash != hash_json(prior)
        or prior_hash != gate["short_report_hash"]
        or capture_hash != hash_json(capture)
        or prior["capture_report_hash"] != capture_hash
    ):
        raise ValueError("local search must reuse qualified student tape")
    rows = []
    for index, amplitude in enumerate(GRID):
        proxy_dir = output_dir / f"candidate-{index:02d}"
        exam = replay(
            asset_root=asset_root,
            captured=captured,
            output_dir=proxy_dir,
            focal_probe_nm=amplitude,
        )
        if exam["capture_report_hash"] != capture_hash:
            raise RuntimeError("proxy course tape changed during search")
        with np.load(
            proxy_dir / ("short-replay.npz" if amplitude == 0.0 else "short-probe.npz"),
            allow_pickle=False,
        ) as trajectory:
            speed = np.asarray(trajectory["ball_speed_mps"], dtype=np.float64)
            distance = np.asarray(trajectory["ball_pelvis_distance_m"], dtype=np.float64)
        safe = bool(
            exam["all_robot_bodies_safe"]
            and not exam["own_nonfoot_seen"]
            and exam["replay_first_foot_frame"] is not None
        )
        # Lower speed and smaller ball-to-body distance both matter. This is
        # development ranking only; authoritative foot-retention is unknown.
        score = float(np.mean(speed[80 - 45 : 100 - 45 + 1])) + 0.4 * float(
            np.mean(distance[80 - 45 : 100 - 45 + 1])
        )
        rows.append(
            {
                "candidate_index": index,
                "ankle_torque_nm": amplitude,
                "proxy_report_hash": exam["report_hash"],
                "body_collision_safe": safe,
                "frame86_speed_mps": exam["replay_frame86_ball_speed_mps"],
                "frame86_distance_m": exam["replay_frame86_ball_pelvis_distance_m"],
                "tail_speed_mean_mps": float(np.mean(speed[80 - 45 : 100 - 45 + 1])),
                "tail_distance_mean_m": float(np.mean(distance[80 - 45 : 100 - 45 + 1])),
                "development_score": score,
            }
        )
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    baseline = rows[4]
    eligible = [row for row in rows if row["body_collision_safe"]]
    best = min(eligible, key=lambda row: row["development_score"]) if eligible else None
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("proxy-search source changed during rollout")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_team_local_search.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "proxy_fidelity_report_hash": gate_hash,
        "grid_nm": list(GRID),
        "objective": "tail speed + 0.4 * tail pelvis distance, control frames 80-100",
        "rows": rows,
        "baseline_score": baseline["development_score"],
        "selected_index": None if best is None else best["candidate_index"],
        "selected_development_score": None if best is None else best["development_score"],
        "full_world_validated": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--captured", type=Path, required=True)
    parser.add_argument("--fidelity", type=Path, required=True)
    parser.add_argument("--qualified-probe", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    report = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {"report_hash": report["report_hash"], "selected_index": report["selected_index"]}
        )
    )


if __name__ == "__main__":
    main()
