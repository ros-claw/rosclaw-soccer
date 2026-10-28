"""Compare matched actual private SIM rollouts, without declaring a catch."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receiver_probe_pair import compare_receiver_probe_pair
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("baseline", "candidate", "snapshots", "scene", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("baseline-hash", "candidate-hash", "snapshot-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    for path, expected in (
        (a.baseline, a.baseline_hash),
        (a.candidate, a.candidate_hash),
        (a.snapshots, a.snapshot_hash),
    ):
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("pinned completed receipt required")
    base = json.loads(a.baseline.read_text())
    cand = json.loads(a.candidate.read_text())
    snapshot = json.loads(a.snapshots.read_text())
    if (
        base.get("status") != "RELAY_LEG_RESIDUAL_ZERO_OWN_REPLAY_NOT_FOOTBALL_SUCCESS"
        or cand.get("status") != "RELAY_LEG_RESIDUAL_PHYSICAL_PROBE_NOT_FOOTBALL_SUCCESS"
        or base.get("probe_leg_rad") != [0.0] * 12
        or type(cand.get("probe_leg_rad")) is not list
        or len(cand["probe_leg_rad"]) != 12
        or any(
            type(v) not in (int, float) or not np.isfinite(v) or abs(v) > 0.15
            for v in cand["probe_leg_rad"]
        )
        or not any(cand.get("probe_leg_rad", []))
        or a.scene.is_symlink()
        or digest(a.scene) != snapshot.get("scene_hash")
        or any(
            base.get(k) != cand.get(k)
            for k in (
                "source_head",
                "source_hash",
                "audit_hash",
                "snapshot_hash",
                "frontier_hash",
                "role_proof_hash",
            )
        )
        or base.get("snapshot_hash") != a.snapshot_hash
        or any(
            x.get(k) is not False
            for x in (base, cand)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "future_controls_used",
            )
        )
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
    ):
        raise ValueError("same bound source, actual zero and nonzero SIM probes required")
    b = {r["index"]: r for r in base["rows"]}
    c = {r["index"]: r for r in cand["rows"]}
    if (
        len(base["rows"]) != 8
        or len(cand["rows"]) != 8
        or len(b) != 8
        or len(c) != 8
        or set(b) != set(c)
    ):
        raise ValueError("all eight exactly matched development cases required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for index in sorted(b):
        if any(b[index].get(k) != c[index].get(k) for k in ("context", "side")):
            raise ValueError("paired source context required")
        trajectories = []
        for receipt, row in ((a.baseline, b[index]), (a.candidate, c[index])):
            path = receipt.parent / row["physical_trajectory"]
            if (
                path.is_symlink()
                or path.resolve().parent != receipt.parent.resolve()
                or digest(path) != row["physical_trajectory_hash"]
            ):
                raise ValueError("confined hash-matching actual physical artifact required")
            with np.load(path, allow_pickle=False) as x:
                actions = x["executed_receiver_action"]
                observations = x["receiver_observation"]
                if (
                    actions.dtype != np.float32
                    or actions.ndim != 2
                    or actions.shape[1:] != (29,)
                    or observations.dtype != np.float32
                    or observations.shape != (len(actions), 98)
                    or not np.isfinite(actions).all()
                    or not np.isfinite(observations).all()
                    or not np.array_equal(observations[1:, 69:98], actions[:-1])
                    or len(actions) != len(x["qpos"])
                ):
                    raise ValueError("each next observation must carry the actual executed action")
                trajectories.append(
                    {k: x[k].copy() for k in ("qpos", "qvel", "torque", "ball_forces")}
                )
            if digest(path) != row["physical_trajectory_hash"]:
                raise ValueError("physical artifact changed during read")
        compared = compare_receiver_probe_pair(
            *trajectories,
            ball_q=layout.bq,
            ball_v=layout.bv,
            receiver_q=layout.rq,
            receiver_v=layout.rv,
        )
        if (
            b[index].get("consecutive_controls") != compared["baseline_controls"]
            or c[index].get("consecutive_controls") != compared["candidate_controls"]
            or type(b[index].get("body_failure")) is not bool
            or type(c[index].get("body_failure")) is not bool
        ):
            raise ValueError("physical exposure and explicit body failure evidence required")
        rows.append(
            dict(
                index=index,
                context=b[index]["context"],
                side=b[index]["side"],
                baseline_body_failure=b[index]["body_failure"],
                candidate_body_failure=c[index]["body_failure"],
                **compared,
            )
        )
    for path, expected in (
        (a.baseline, a.baseline_hash),
        (a.candidate, a.candidate_hash),
        (a.snapshots, a.snapshot_hash),
    ):
        if digest(path) != expected:
            raise ValueError("input changed during comparison")
    if digest(a.scene) != snapshot["scene_hash"]:
        raise ValueError("physical scene changed during comparison")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="MATCHED_RECEIVER_PHYSICAL_PROBE_NOT_CATCH_CERTIFICATE",
            baseline_hash=a.baseline_hash,
            candidate_hash=a.candidate_hash,
            snapshot_hash=a.snapshot_hash,
            scene_hash=snapshot["scene_hash"],
            source_head=a.expected_head,
            rows=rows,
            independent_fresh=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
