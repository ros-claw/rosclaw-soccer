"""SIM_ONLY CPU stance geometry scan on a consumed receiving course.

Perturbing the robot root at snapshot zero is a feasibility diagnostic, not a
robot controller or valid team-world evidence. Fresh8 remains sealed.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
from rsi_mjx_contact_replay_smoke import _contact_masks
from rsi_receiving_phase_contact_probe import _evaluate

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model


def scan(*, asset_root: Path, captured: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    helper = source.with_name("rsi_receiving_phase_contact_probe.py")
    helper_hash = hash_bytes(helper.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY output required")
    trace_path = captured / "motor-trace.npz"
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    commitment = capture.pop("report_hash")
    if commitment != hash_json(capture) or capture["trace_hash"] != hash_bytes(
        trace_path.read_bytes()
    ):
        raise ValueError("sealed receiving trace required")
    with np.load(trace_path, allow_pickle=False) as trace:
        qpos = np.asarray(trace["sonic_recorded_qpos"], dtype=np.float64)
        qvel = np.asarray(trace["sonic_recorded_qvel"], dtype=np.float64)
        target = np.asarray(trace["sonic_recorded_target"], dtype=np.float64)
        kp = np.asarray(trace["sonic_recorded_kp"], dtype=np.float64)
        kd = np.asarray(trace["sonic_recorded_kd"], dtype=np.float64)
    if qpos.shape != (300, 43) or qvel.shape != (300, 41):
        raise ValueError("complete recorded physical state required")
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    data = mujoco.MjData(model)
    masks = _contact_masks(model)
    rows = []
    for dx in np.linspace(-0.18, 0.18, 13):
        for dy in np.linspace(-0.18, 0.18, 13):
            shifted = qpos.copy()
            shifted[0, 0] += dx
            shifted[0, 1] += dy
            outcome = _evaluate(
                model,
                data,
                shifted,
                qvel,
                target,
                kp,
                kd,
                np.zeros(5, dtype=np.float64),
                (22, 33, 43),
                0,
                masks,
            )
            rows.append({"root_dx_m": float(dx), "root_dy_m": float(dy), **outcome})
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
    ):
        raise RuntimeError("geometry scanner source changed during physics")
    safe_clean = [row for row in rows if row["safe"] and row["clean_foot"]]
    safe_clean.sort(key=lambda row: abs(row["root_dx_m"]) + abs(row["root_dy_m"]))
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_stance_geometry_scan.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "capture_hash": commitment,
        "compiled_model_hash": compiled_model_hash(model),
        "courses": len(rows),
        "safe_clean_foot_count": len(safe_clean),
        "nearest_safe_clean": safe_clean[0] if safe_clean else None,
        "physical_transfer_authorized": False,
        "fresh8_opened": False,
    }
    report["report_hash"] = hash_json(report)
    output_dir.mkdir(parents=True)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    (output_dir / "grid.json").write_text(
        json.dumps(rows, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(scan(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
