"""Audit same-snapshot full physical SIM continuations; not a catch proof."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiver_long_pair import compare_long_pair
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "snapshots", "scene", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("baseline-hash", "candidate-hash", "snapshot-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    inputs = (
        (args.baseline, args.baseline_hash),
        (args.candidate, args.candidate_hash),
        (args.snapshots, args.snapshot_hash),
    )
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in inputs)
    ):
        raise ValueError("pinned receipts and new external output required")
    baseline, candidate, snapshots = (json.loads(path.read_text()) for path, _ in inputs)
    if (
        baseline.get("status") != "RELAY_NONDEFAULT_SMOOTH_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
        or candidate.get("status")
        != "RELAY_LONG_PROBE_CAUSAL_RETURN_PHYSICAL_CANDIDATE_NOT_SUCCESS"
        or baseline.get("smooth_transition_duration") != 10
        or candidate.get("smooth_transition_duration") != 10
        or baseline.get("snapshot_hash") != args.snapshot_hash
        or args.scene.is_symlink()
        or digest(args.scene) != snapshots.get("scene_hash")
        or any(
            baseline.get(k) != candidate.get(k)
            for k in (
                "source_hash",
                "audit_hash",
                "snapshot_hash",
                "frontier_hash",
                "role_proof_hash",
                "expanded_velocity_proof_hash",
            )
        )
        or any(
            receipt.get(k) is not False
            for receipt in (baseline, candidate)
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "future_controls_used",
            )
        )
    ):
        raise ValueError("matched unpromoted causal source and baseline required")
    b = {row["index"]: row for row in baseline["rows"]}
    c = {row["index"]: row for row in candidate["rows"]}
    s = {row["index"]: row for row in snapshots["rows"]}
    if (
        any(len(receipt["rows"]) != 24 for receipt in (baseline, candidate, snapshots))
        or len(b) != 24
        or set(b) != set(c)
        or set(b) != set(s)
    ):
        raise ValueError("all 24 unique paired source identities required")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    native = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    layout = dict(ball_q=native.bq, ball_v=native.bv, receiver_q=native.rq, receiver_v=native.rv)
    rows = []
    for index in sorted(b):
        if any(
            b[index].get(k) != c[index].get(k) or b[index].get(k) != s[index].get(k)
            for k in ("context", "side", "receiver_lateral_offset_m")
        ):
            raise ValueError("case context and lateral offset must match snapshot")
        histories = []
        for receipt, row in ((args.baseline, b[index]), (args.candidate, c[index])):
            path = receipt.parent / row["physical_trajectory"]
            if (
                path.is_symlink()
                or path.resolve().parent != receipt.parent.resolve()
                or digest(path) != row["physical_trajectory_hash"]
            ):
                raise ValueError("confined exact physical history required")
            with np.load(path, allow_pickle=False) as data:
                histories.append(
                    {
                        k: data[k].copy()
                        for k in (
                            "qpos",
                            "qvel",
                            "torque",
                            "ball_forces",
                            "executed_receiver_action",
                            "receiver_observation",
                        )
                    }
                )
            if digest(path) != row["physical_trajectory_hash"]:
                raise ValueError("physical history changed during read")
        comparison = compare_long_pair(*histories, layout=layout)
        if any(
            type(row.get("body_failure")) is not bool
            or row.get("consecutive_controls") != len(history["qpos"])
            for row, history in zip((b[index], c[index]), histories, strict=True)
        ):
            raise ValueError("explicit failures and measured trajectory lengths required")
        rows.append(
            dict(
                index=index,
                context=b[index]["context"],
                baseline_body_failure=b[index]["body_failure"],
                candidate_body_failure=c[index]["body_failure"],
                baseline_ready=b[index]["receiving_ready"],
                candidate_ready=c[index]["receiving_ready"],
                **comparison,
            )
        )
    if (
        any(digest(path) != expected for path, expected in inputs)
        or digest(args.scene) != snapshots["scene_hash"]
    ):
        raise ValueError("pinned inputs changed during audit")
    clean(args.expected_head)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="LONG_RECEIVER_PAIRED_SIM_DIAGNOSIS_NOT_POSSESSION",
            baseline_hash=args.baseline_hash,
            candidate_hash=args.candidate_hash,
            snapshot_hash=args.snapshot_hash,
            scene_hash=snapshots["scene_hash"],
            source_head=args.expected_head,
            rows=rows,
            independent_fresh=False,
            independent_replay=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
