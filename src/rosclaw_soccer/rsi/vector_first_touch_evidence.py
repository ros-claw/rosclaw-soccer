"""Fail-closed audit of independent SIM_ONLY Isaac first-touch environments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json


def audit_vector_first_touch(folder: Path) -> dict[str, Any]:
    report: dict[str, Any] = json.loads((folder / "report.json").read_text(encoding="utf-8"))
    trace_path = folder / "trace.npz"
    committed = {key: value for key, value in report.items() if key != "report_hash"}
    if (
        report.get("schema") != "rsi_isaac_vector_first_touch_smoke_v1"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("learning_authorized") is not False
        or report.get("promotion_authorized") is not False
        or report.get("report_hash") != hash_json(committed)
        or report.get("trace_hash") != hash_bytes(trace_path.read_bytes())
    ):
        raise ValueError("unauthenticated vector first-touch evidence")
    entries = report.get("environments")
    frames = report.get("frames")
    if (
        not isinstance(entries, list)
        or not 2 <= len(entries) <= 16
        or type(frames) is not int
        or not 50 <= frames <= 400
        or [row.get("environment") for row in entries] != list(range(len(entries)))
    ):
        raise ValueError("invalid independent environment ledger")
    with np.load(trace_path, allow_pickle=False) as archive:
        if set(archive.files) != {
            "ball_position_m",
            "ball_angular_velocity_rad_s",
            "ball_body_contact_force_peak_n",
        }:
            raise ValueError("vector trace contract changed")
        positions = archive["ball_position_m"]
        spin = archive["ball_angular_velocity_rad_s"]
        force = archive["ball_body_contact_force_peak_n"]
    n = len(entries)
    if (
        positions.shape != (frames, n, 3)
        or spin.shape != (frames, n, 3)
        or force.shape != (frames, n, 6)
        or not np.isfinite(positions).all()
        or not np.isfinite(spin).all()
        or not np.isfinite(force).all()
        or np.any(force < 0)
    ):
        raise ValueError("invalid vector physics trace")
    lanes = np.asarray([row["lane_y_m"] for row in entries], dtype=np.float64)
    if not np.isfinite(lanes).all() or np.min(np.diff(lanes)) < 6.0:
        raise ValueError("training lanes are not physically isolated")
    if np.max(np.abs(positions[:, :, 1] - lanes[None, :])) >= 4.0:
        raise ValueError("ball escaped its lane")
    courses = []
    clean = 0
    positives = 0
    sides: set[str] = set()
    for i, row in enumerate(entries):
        course = row["course"]
        key = (course["ball_x_m"], course["ball_y_local_m"], course["ball_vx_m_s"])
        courses.append(key)
        if (
            key[0] not in (2.4, 2.6)
            or key[1] not in (-0.1, 0.1)
            or key[2] not in (-0.5, 0.5)
            or not 0.65 <= row["minimum_pelvis_z_m"] <= 1.5
            or not np.allclose(
                row["ball_final_local_xyz_m"],
                (positions[-1, i, 0], positions[-1, i, 1] - lanes[i], positions[-1, i, 2]),
                atol=1e-5,
                rtol=0,
            )
        ):
            raise ValueError("course, body safety or final ball position disagrees")
        active = np.flatnonzero(np.max(force[:, i], axis=1) > 1.0)
        first = int(active[0]) if len(active) else None
        bodies = np.flatnonzero(np.max(force[:, i], axis=0) > 1.0).tolist()
        if row["first_contact_frame"] != first or row["contact_body_indices"] != bodies:
            raise ValueError("contact ledger disagrees with authenticated trace")
        positives += first is not None
        clean_foot_only = bool(bodies and set(bodies) <= {0, 1})
        clean += clean_foot_only
        if clean_foot_only and first is not None:
            sides.add("left" if force[first, i, 0] >= force[first, i, 1] else "right")
    if len(set(courses)) != len(courses):
        raise ValueError("duplicate courses are not independent episodes")
    result = {
        "schema": "rsi_isaac_vector_first_touch_audit_v1",
        "activation_ceiling": "SIM_ONLY",
        "source_report_hash": report["report_hash"],
        "independent_physical_episode_count": n,
        "any_contact_episode_count": positives,
        "clean_foot_only_episode_count": clean,
        "positive_foot_sides": sorted(sides),
        "imitation_training_authorized": bool(n >= 8 and clean >= 4 and sides == {"left", "right"}),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("folder", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    report = audit_vector_first_touch(args.folder)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, sort_keys=True)
        stream.write("\n")
    print(json.dumps(report, sort_keys=True))


if __name__ == "__main__":
    main()
