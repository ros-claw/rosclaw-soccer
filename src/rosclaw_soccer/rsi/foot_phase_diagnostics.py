"""Audited, observational foot/knee phase window before incoming ball contact.

This does not replay another phase or assert a counterfactual improvement. Only
pre-contact measured frames are considered; post-impact motion is excluded.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_snapshot_bank import audit_snapshot_bank
from rosclaw_soccer.sim.contracts import hash_json


def best_precontact_offset(forward: np.ndarray, *, first: int, lookback: int) -> tuple[int, float]:
    if (
        forward.ndim != 1
        or not np.isfinite(forward).all()
        or type(first) is not int
        or type(lookback) is not int
        or not 1 <= lookback <= 30
        or first < lookback
        or first >= len(forward)
    ):
        raise ValueError("invalid precontact foot phase window")
    window = forward[first - lookback : first + 1]
    return int(np.argmax(window)) - lookback, float(np.max(window) - forward[first])


def phase_opportunity(bank_folder: Path, *, lookback_frames: int = 10) -> dict[str, Any]:
    bank_audit = audit_snapshot_bank(bank_folder)
    manifest = json.loads((bank_folder / "manifest.json").read_text(encoding="utf-8"))
    if type(lookback_frames) is not int or not 1 <= lookback_frames <= 30:
        raise ValueError("bounded precontact phase window required")
    bodies: dict[Path, np.ndarray] = {}
    rows = []
    for index, source in enumerate(manifest["snapshots"]):
        folder = Path(source["source_folder"])
        if folder not in bodies:
            with np.load(folder / "body_trace.npz", allow_pickle=False) as trace:
                bodies[folder] = trace["foot_geometry_position_before_step_m"].copy()
        geometry = bodies[folder]
        first = source["start_frame"] + source["first_contact_offset"]
        lane = source["lane"]
        if first < lookback_frames or first >= geometry.shape[0]:
            raise ValueError("source contact outside complete precontact phase window")
        # Body order was authenticated by the snapshot bank auditor: right foot,
        # then right knee. These are link centers, not collision-surface margins.
        forward = geometry[:, lane, 1, 0] - geometry[:, lane, 3, 0]
        best_offset, gain = best_precontact_offset(forward, first=first, lookback=lookback_frames)
        rows.append(
            {
                "snapshot_index": index,
                "source_lane": lane,
                "current_foot_minus_knee_x_m": float(forward[first]),
                "best_prior_foot_minus_knee_x_m": float(forward[first] + gain),
                "best_prior_offset_frames": best_offset,
                "prior_gain_m": gain,
                "source_ball_vx_m_s": source["course"]["ball_vx_m_s"],
                "observational_only": True,
            }
        )
    gains = np.asarray([row["prior_gain_m"] for row in rows])
    if not np.isfinite(gains).all():
        raise ValueError("nonfinite phase opportunity")
    report = {
        "schema": "rsi_isaac_first_touch_phase_opportunity_v1",
        "activation_ceiling": "SIM_ONLY",
        "snapshot_bank_manifest_hash": bank_audit["manifest_hash"],
        "lookback_frames": lookback_frames,
        "sample_count": len(rows),
        "median_prior_gain_m": float(np.median(gains)),
        "prior_gain_at_least_0_05m_count": int(np.count_nonzero(gains >= 0.05)),
        "training_prior_gain_at_least_0_05m_count": int(np.count_nonzero(gains[:15] >= 0.05)),
        "holdout_prior_gain_at_least_0_05m_count": int(np.count_nonzero(gains[15:] >= 0.05)),
        "rows": rows,
        "counterfactual_phase_shift_tested": False,
        "learning_authorized": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot-bank", required=True, type=Path)
    parser.add_argument("--lookback-frames", type=int, default=10)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists():
        parser.error("immutable phase diagnosis already exists")
    report = phase_opportunity(args.snapshot_bank, lookback_frames=args.lookback_frames)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}))


if __name__ == "__main__":
    main()
