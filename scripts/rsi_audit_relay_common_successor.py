"""Measure ordered actual return contacts in same-exposure paired SIM trials."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.relay_common_successor import common_successor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("review", "baseline", "candidate", "snapshots", "scene", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("review-hash", "baseline-hash", "candidate-hash", "snapshot-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    pins = (
        (args.review, args.review_hash),
        (args.baseline, args.baseline_hash),
        (args.candidate, args.candidate_hash),
        (args.snapshots, args.snapshot_hash),
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in pins)
    ):
        raise ValueError("exact bound receipts and new external output required")
    review, baseline, candidate, snapshots = (json.loads(p.read_text()) for p, _ in pins)
    if (
        review.get("status") != "LONG_RECEIVER_PAIRED_SIM_DIAGNOSIS_NOT_POSSESSION"
        or review.get("baseline_hash") != args.baseline_hash
        or review.get("candidate_hash") != args.candidate_hash
        or review.get("snapshot_hash") != args.snapshot_hash
        or digest(args.scene) != snapshots.get("scene_hash")
        or args.scene.is_symlink()
        or any(
            d.get(k) is not False
            for d in (review, baseline, candidate)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "independent_fresh",
            )
        )
    ):
        raise ValueError("matched SIM paired outcome evidence required")
    b = {r["index"]: r for r in baseline["rows"]}
    c = {r["index"]: r for r in candidate["rows"]}
    pairs = {r["index"]: r for r in review["rows"]}
    if (
        any(len(d["rows"]) != 24 for d in (baseline, candidate, review))
        or len(b) != 24
        or set(b) != set(c)
        or set(b) != set(pairs)
    ):
        raise ValueError("all unique matched cases required")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    teammate = NativeVelocityLayout(
        model, robot_prefix="a/", ball_body_name="soccer_ball", player_count=2
    )
    layout = dict(
        ball_q=teammate.bq, ball_v=teammate.bv, teammate_q=teammate.rq, teammate_v=teammate.rv
    )
    rows = []
    for index in sorted(b):
        pair = pairs[index]
        measured = []
        for receipt, source in ((args.baseline, b[index]), (args.candidate, c[index])):
            path = receipt.parent / source["physical_trajectory"]
            if (
                path.is_symlink()
                or path.resolve().parent != receipt.parent.resolve()
                or digest(path) != source["physical_trajectory_hash"]
            ):
                raise ValueError("exact confined trajectory required")
            with np.load(path, allow_pickle=False) as data:
                h = {k: data[k].copy() for k in ("qpos", "qvel", "ball_forces")}
            if (
                digest(path) != source["physical_trajectory_hash"]
                or len(h["qpos"]) != source["consecutive_controls"]
            ):
                raise ValueError("unchanged actual physical exposure required")
            measured.append(
                common_successor(
                    h,
                    common=pair["common_controls"],
                    return_start=source["receiver_velocity_controls"],
                    layout=layout,
                    timestep=float(model.opt.timestep),
                )
            )
        rows.append(
            dict(
                index=index,
                context=pair["context"],
                common_controls=pair["common_controls"],
                baseline=measured[0],
                candidate=measured[1],
                baseline_body_failure=pair["baseline_body_failure"],
                candidate_body_failure=pair["candidate_body_failure"],
            )
        )
    if any(digest(p) != h for p, h in pins) or digest(args.scene) != snapshots["scene_hash"]:
        raise ValueError("inputs changed during audit")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="ORDERED_TEAM_CONTACT_COMMON_EXPOSURE_NOT_POSSESSION",
            source_head=args.expected_head,
            review_hash=args.review_hash,
            baseline_hash=args.baseline_hash,
            candidate_hash=args.candidate_hash,
            snapshot_hash=args.snapshot_hash,
            rows=rows,
            independent_fresh=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
