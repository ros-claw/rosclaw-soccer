"""Export audited before-step teacher examples; not football success or RL."""

import argparse
import json
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receive_teacher_causal_dataset import causal_teacher_samples
from rosclaw_soccer.rsi.receive_teacher_evidence import audit_teacher_torque
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "snapshots", "scene", "replay", "torque-review", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "candidate-hash",
        "snapshot-hash",
        "replay-hash",
        "torque-review-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.candidate, a.candidate_hash),
        (a.snapshots, a.snapshot_hash),
        (a.replay, a.replay_hash),
        (a.torque_review, a.torque_review_hash),
    )
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != expected for path, expected in pins)
    ):
        raise ValueError("exact pinned evidence and unused external dataset required")
    data, snapshots, replay, audit = (json.loads(path.read_text()) for path, _ in pins)
    if (
        data.get("status") != "RELAY_CONTACT_TEACHER_ASSISTED_PHYSICAL_NOT_SUCCESS"
        or data.get("contact_teacher_assisted") is not True
        or data.get("contact_teacher_capture", False) is not False
        or replay.get("status") != "RECEIVER_TEACHER_SAVED_TORQUE_REINTEGRATED_NOT_POLICY"
        or replay.get("candidate_hash") != a.candidate_hash
        or replay.get("neural_policy_reinferred") is not False
        or replay.get("teacher_recomputed") is not False
        or audit.get("status") != "RECEIVER_TEACHER_TORQUE_ACCOUNTING_NOT_SUCCESS"
        or audit.get("candidate_hash") != a.candidate_hash
        or data.get("snapshot_hash") != a.snapshot_hash
        or replay.get("snapshot_hash") != a.snapshot_hash
        or a.scene.is_symlink()
        or digest(a.scene) != snapshots.get("scene_hash")
        or any(
            item.get(k) is not False
            for item in (data, replay, audit)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "independent_fresh",
                "goal_complete",
            )
        )
    ):
        raise ValueError(
            "same teacher physics, torque accounting and own saved-torque replay required"
        )
    cases = {row["index"]: row for row in snapshots["rows"]}
    replay_ids = {row["index"] for row in replay["rows"]}
    audit_ids = {row["index"] for row in audit["rows"]}
    if (
        len(cases) != 24
        or {r["index"] for r in data["rows"]} != set(cases)
        or replay_ids != set(cases)
        or audit_ids != set(cases)
    ):
        raise ValueError("all unique 24 physical cases must be independently accounted")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    receiver, teammate = [
        NativeVelocityLayout(
            model, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=2
        )
        for prefix in ("b/", "a/")
    ]
    a.output.mkdir(parents=True)
    records, total = [], 0
    for row in data["rows"]:
        case = cases[row["index"]]
        snap = a.snapshots.parent / case["snapshot"]
        raw = a.candidate.parent / row["physical_trajectory"]
        if (
            snap.is_symlink()
            or raw.is_symlink()
            or snap.resolve().parent != a.snapshots.parent.resolve()
            or raw.resolve().parent != a.candidate.parent.resolve()
            or digest(snap) != case["snapshot_hash"]
            or digest(raw) != row["physical_trajectory_hash"]
        ):
            raise ValueError("confined exact source trajectory and snapshot required")
        with np.load(snap, allow_pickle=False) as state, np.load(raw, allow_pickle=False) as source:
            d = mujoco.MjData(model)
            mujoco.mj_setState(model, d, state["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, d)
            audit_teacher_torque(
                source["torque"],
                source["teacher_parent_torque"],
                source["teacher_actual_torque_delta"],
                receiver_actuators=np.asarray(receiver.ai),
                commands=row["teacher_native_commands"],
                active_commands=row["teacher_active_native_commands"],
            )
            x, y = causal_teacher_samples(
                initial_qpos=d.qpos.copy(),
                initial_qvel=d.qvel.copy(),
                initial_actuator_torque=d.actuator_force.copy(),
                qpos=source["qpos"],
                qvel=source["qvel"],
                actual_torque=source["torque"],
                parent_torque=source["teacher_parent_torque"],
                ball_forces=source["ball_forces"],
                receiver_layout=receiver,
                teammate_layout=teammate,
            )
        path = a.output / f"case-{row['index']}.npz"
        np.savez_compressed(path, observations=x, applied_residual_nm=y)
        records.append(
            dict(
                index=row["index"],
                context=row["context"],
                split="validation" if row["context"] == 7 else "train",
                path=path.name,
                hash=digest(path),
                source_trajectory_hash=row["physical_trajectory_hash"],
                snapshot_hash=case["snapshot_hash"],
                samples=len(x),
                body_failure=row["body_failure"],
                receiving_ready=row["receiving_ready"],
            )
        )
        total += len(x)
        if digest(snap) != case["snapshot_hash"] or digest(raw) != row["physical_trajectory_hash"]:
            raise ValueError("source case changed during export")
    if (
        total != data["native_steps"]
        or {r["context"] for r in records} != {2, 3, 5, 7}
        or any(digest(path) != h for path, h in pins)
    ):
        raise ValueError("full unchanged development coverage required")
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="RECEIVER_ASSISTED_CAUSAL_TORQUE_DATA_NOT_TRAINED",
            source_head=a.expected_head,
            candidate_hash=a.candidate_hash,
            snapshot_hash=a.snapshot_hash,
            torque_review_hash=a.torque_review_hash,
            replay_hash=a.replay_hash,
            samples=total,
            records=records,
            feature_dim=140,
            train_contexts=[2, 3, 5],
            validation_contexts=[7],
            actor_kinematics_are_before_step=True,
            parent_torque_same_step=True,
            future_actions_used=False,
            teacher_qualified=False,
            training_performed=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
