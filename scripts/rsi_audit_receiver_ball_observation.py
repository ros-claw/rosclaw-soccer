"""Audit causal ball features against independently saved native next states."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiver_ball_observation import receiver_ball_observation
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("snapshots", "baseline", "scene", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("snapshot-hash", "baseline-hash", "expected-head"):
        parser.add_argument("--" + key, required=True)
    args = parser.parse_args()
    clean(args.expected_head)
    if (
        args.output.exists()
        or args.output.resolve().is_relative_to(Path.cwd())
        or args.snapshots.is_symlink()
        or args.baseline.is_symlink()
        or args.scene.is_symlink()
        or digest(args.snapshots) != args.snapshot_hash
        or digest(args.baseline) != args.baseline_hash
    ):
        raise ValueError("exact immutable inputs and unused external output required")
    snapshots = json.loads(args.snapshots.read_text())
    baseline = json.loads(args.baseline.read_text())
    if (
        baseline.get("snapshot_hash") != args.snapshot_hash
        or snapshots.get("scene_hash") != digest(args.scene)
        or baseline.get("coupled_physics_continuation_verified") is not True
        or any(
            receipt.get(flag) is not False
            for receipt in (snapshots, baseline)
            for flag in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("unpromoted snapshot-bound physical continuation required")
    snap_rows = {r["index"]: r for r in snapshots["rows"]}
    base_rows = {r["index"]: r for r in baseline["rows"]}
    if len(snap_rows) != 24 or len(base_rows) != 24 or set(snap_rows) != set(base_rows):
        raise ValueError("complete unique 24-case paired bank required")
    model = mujoco.MjModel.from_xml_path(str(args.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for index in sorted(snap_rows):
        snap, base = snap_rows[index], base_rows[index]
        snapshot_path = args.snapshots.parent / snap["snapshot"]
        trajectory_path = args.baseline.parent / base["physical_trajectory"]
        if (
            snapshot_path.is_symlink()
            or trajectory_path.is_symlink()
            or digest(snapshot_path) != snap["snapshot_hash"]
            or digest(trajectory_path) != base["physical_trajectory_hash"]
            or not base["coupled_native_substeps_exact"]
        ):
            raise ValueError("exact original snapshot and physical trajectory required")
        data = mujoco.MjData(model)
        with np.load(snapshot_path, allow_pickle=False) as saved:
            mujoco.mj_setState(model, data, saved["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
        mujoco.mj_forward(model, data)
        layout.angular_velocity_body(data)
        q0, v0 = layout.canonical(data)
        current = receiver_ball_observation(q0, v0)
        with np.load(trajectory_path, allow_pickle=False) as physical:
            data.qpos[:] = physical["qpos"][0, 0]
            data.qvel[:] = physical["qvel"][0, 0]
        mujoco.mj_forward(model, data)
        q1, v1 = layout.canonical(data)
        following = receiver_ball_observation(q1, v1)
        finite_difference = (following[:3] - current[:3]) / model.opt.timestep
        rows.append(
            dict(
                index=index,
                context=snap["context"],
                feature=current.tolist(),
                finite_difference_velocity_error_mps=(finite_difference - current[3:]).tolist(),
            )
        )
        if (
            digest(snapshot_path) != snap["snapshot_hash"]
            or digest(trajectory_path) != base["physical_trajectory_hash"]
        ):
            raise ValueError("physical input changed during audit")
    if (
        digest(args.snapshots) != args.snapshot_hash
        or digest(args.baseline) != args.baseline_hash
        or digest(args.scene) != snapshots["scene_hash"]
    ):
        raise ValueError("input changed during audit")
    clean(args.expected_head)
    errors = np.array([r["finite_difference_velocity_error_mps"] for r in rows])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        args.output,
        dict(
            status="SIM_BALL_OBSERVATION_FINITE_DIFFERENCE_AUDIT_NOT_POLICY",
            source_head=args.expected_head,
            snapshot_hash=args.snapshot_hash,
            baseline_hash=args.baseline_hash,
            scene_hash=snapshots["scene_hash"],
            timestep_s=float(model.opt.timestep),
            max_error_norm_mps=float(np.linalg.norm(errors, axis=1).max()),
            rows=rows,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
