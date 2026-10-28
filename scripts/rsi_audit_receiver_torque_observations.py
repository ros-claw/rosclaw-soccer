"""Compare live actor observation construction with saved causal training data."""

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.shared_native_receiver_torque_actor import SharedNativeReceiverTorqueActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("dataset", "source", "snapshots", "scene", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("dataset-hash", "source-hash", "snapshot-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    pins = ((a.dataset, a.dataset_hash), (a.source, a.source_hash), (a.snapshots, a.snapshot_hash))
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != h for path, h in pins)
    ):
        raise ValueError("exact development source and new external output required")
    data, source, snapshots = (json.loads(path.read_text()) for path, _ in pins)
    if (
        data.get("status") != "RECEIVER_ASSISTED_CAUSAL_TORQUE_DATA_NOT_TRAINED"
        or data.get("candidate_hash") != a.source_hash
        or data.get("snapshot_hash") != a.snapshot_hash
        or source.get("status") != "RELAY_CONTACT_TEACHER_ASSISTED_PHYSICAL_NOT_SUCCESS"
        or snapshots.get("scene_hash") != digest(a.scene)
        or a.scene.is_symlink()
        or any(
            item.get(k) is not False
            for item in (data, source)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "independent_fresh",
                "goal_complete",
            )
        )
    ):
        raise ValueError("exact unpromoted assisted source and dataset required")
    source_rows = {x["index"]: x for x in source["rows"]}
    snapshot_rows = {x["index"]: x for x in snapshots["rows"]}
    if (
        len(data["records"]) != 24
        or len(source_rows) != 24
        or len(snapshot_rows) != 24
        or {r["index"] for r in data["records"]} != set(source_rows)
        or set(source_rows) != set(snapshot_rows)
    ):
        raise ValueError("24 unique matched developmental cases required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    receiver, teammate = [
        NativeVelocityLayout(model, robot_prefix=p, ball_body_name="soccer_ball", player_count=2)
        for p in ("b/", "a/")
    ]
    count = 0
    for row in data["records"]:
        s, snap = source_rows[row["index"]], snapshot_rows[row["index"]]
        archive, raw, init = (
            a.dataset.parent / row["path"],
            a.source.parent / s["physical_trajectory"],
            a.snapshots.parent / snap["snapshot"],
        )
        case_pins = (
            (archive, row["hash"]),
            (raw, s["physical_trajectory_hash"]),
            (init, snap["snapshot_hash"]),
        )
        if (
            any(path.is_symlink() or digest(path) != h for path, h in case_pins)
            or archive.resolve().parent != a.dataset.parent.resolve()
            or raw.resolve().parent != a.source.parent.resolve()
            or init.resolve().parent != a.snapshots.parent.resolve()
        ):
            raise ValueError("exact confined case artifacts required")
        with (
            np.load(archive, allow_pickle=False) as x,
            np.load(raw, allow_pickle=False) as z,
            np.load(init, allow_pickle=False) as initial,
        ):
            d = mujoco.MjData(model)
            mujoco.mj_setState(model, d, initial["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, d)
            q, v = z["qpos"].reshape(-1, model.nq), z["qvel"].reshape(-1, model.nv)
            torque, parent = (
                z["torque"].reshape(-1, model.nu),
                z["teacher_parent_torque"].reshape(-1, model.nu),
            )
            observations = x["observations"].copy()
            if observations.shape != (len(q), 140):
                raise ValueError("complete ordered causal observation tensor required")
            forces = z["ball_forces"].reshape(-1, 2, 3)[:, 1, :]
            if (forces[:10] > 1).any():
                raise ValueError("first control must be pre-receiver contact")
            actor = SharedNativeReceiverTorqueActor.__new__(SharedNativeReceiverTorqueActor)
            actor.world, actor.layout, actor.teammate = SimpleNamespace(data=d), receiver, teammate
            actor.previous_torque = d.actuator_force[receiver.ai].copy()
            actor.previous_contact = np.zeros(3, dtype=bool)
            actor.last_foot_contact_step = None
            for i in range(len(q)):
                actor.next_native_step = 1400 + i
                if i:
                    d.qpos[:], d.qvel[:] = q[i - 1], v[i - 1]
                    actor.previous_torque = torque[i - 1, receiver.ai].copy()
                    actor.previous_contact = forces[i - 1] > 1
                    if actor.previous_contact[:2].any():
                        actor.last_foot_contact_step = 1400 + i - 1
                observed = actor.observation(parent[i, receiver.ai])
                if not np.array_equal(observed, observations[i]):
                    raise ValueError(f"causal observation mismatch case {row['index']} step {i}")
            count += len(q)
        if any(digest(path) != h for path, h in case_pins):
            raise ValueError("case changed during causal reconstruction")
    if count != data["samples"] or any(digest(path) != h for path, h in pins):
        raise ValueError("complete unchanged observation coverage required")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RECEIVER_CAUSAL_OBSERVATION_IDENTITY_NOT_CLOSED_LOOP",
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            source_hash=a.source_hash,
            snapshot_hash=a.snapshot_hash,
            cases=24,
            native_observations=count,
            actor_network_reinferred=False,
            new_physics=False,
            hardware_authorized=False,
            promotion_authorized=False,
            independent_fresh=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
