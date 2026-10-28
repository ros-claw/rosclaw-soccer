"""Recompute actor torques from saved own physical state, not a new rollout."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np
import torch

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receive_teacher_causal_dataset import causal_teacher_samples
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_train_receive_teacher_torque_bc import ResidualActor


def effort_limits() -> np.ndarray:
    return np.asarray(
        [
            139
            if ("hip_roll" in n or "knee" in n)
            else 88
            if ("hip_pitch" in n or "hip_yaw" in n or n == "waist_yaw_joint")
            else 50
            if ("ankle" in n or n in ("waist_roll_joint", "waist_pitch_joint"))
            else 5
            if ("wrist_pitch" in n or "wrist_yaw" in n)
            else 25
            for n in G1_DDS_JOINT_NAMES
        ],
        dtype=np.float64,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "snapshots", "scene", "checkpoint", "training", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "snapshot-hash",
        "checkpoint-hash",
        "training-hash",
        "physical-head",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    a = p.parse_args()
    clean(a.expected_head)
    pins = (
        (a.snapshots, a.snapshot_hash),
        (a.checkpoint, a.checkpoint_hash),
        (a.training, a.training_hash),
    )
    if (
        not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(path.is_symlink() or digest(path) != h for path, h in pins)
    ):
        raise ValueError("exact pinned inputs and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while not a.candidate.is_file():
        state = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                "rosclaw-receiver-neural-torque-b0-20261029.service",
                "-p",
                "ActiveState",
            ],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        if state not in ("ActiveState=active", "ActiveState=activating"):
            raise ValueError("physical worker ended without complete receipt")
        if time.monotonic() >= deadline:
            raise TimeoutError("observer expired; do not restart physical worker")
        time.sleep(15)
    if a.candidate.is_symlink():
        raise ValueError("exact non-symlink physical receipt required")
    candidate_hash = digest(a.candidate)
    source, snaps, train = (
        json.loads(x.read_text()) for x in (a.candidate, a.snapshots, a.training)
    )
    if (
        source.get("status") != "RELAY_NEURAL_TORQUE_RESEARCH_PHYSICAL_NOT_SUCCESS"
        or source.get("neural_torque_actor") is not True
        or source.get("contact_teacher_assisted") is not False
        or source.get("torque_checkpoint_hash") != a.checkpoint_hash
        or source.get("torque_training_hash") != a.training_hash
        or source.get("source_head") != a.physical_head
        or source.get("snapshot_hash") != a.snapshot_hash
        or train.get("checkpoint_hash") != a.checkpoint_hash
        or a.scene.is_symlink()
        or digest(a.scene) != snaps.get("scene_hash")
        or any(
            d.get(k) is not False
            for d in (source, train)
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "independent_fresh",
                "goal_complete",
            )
        )
        or len(source.get("rows", [])) != 24
    ):
        raise ValueError("exact unpromoted learned-torque physical source required")
    cases = {r["index"]: r for r in snaps["rows"]}
    if len(cases) != 24 or {r["index"] for r in source["rows"]} != set(cases):
        raise ValueError("complete 24 unique snapshot cases required")
    state = torch.load(a.checkpoint, map_location="cpu", weights_only=True)
    if state.get("dataset_hash") != train.get("dataset_hash") or any(
        state.get(k) is not False
        for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
    ):
        raise ValueError("exact model lineage and no promotion required")
    torch.set_num_threads(1)
    actor = ResidualActor().eval().requires_grad_(False)
    actor.load_state_dict(state["actor"], strict=True)
    mean, std = state["observation_mean"], state["observation_std"]
    if (
        mean.shape != (140,)
        or std.shape != (140,)
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or (std < 0.01).any()
    ):
        raise ValueError("finite exact actor normalization required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    receiver, teammate = [
        NativeVelocityLayout(model, robot_prefix=p, ball_body_name="soccer_ball", player_count=2)
        for p in ("b/", "a/")
    ]
    effort = effort_limits()
    rows = []
    for row in source["rows"]:
        snap = a.snapshots.parent / cases[row["index"]]["snapshot"]
        raw = a.candidate.parent / row["physical_trajectory"]
        sh, rh = cases[row["index"]]["snapshot_hash"], row["physical_trajectory_hash"]
        if (
            snap.is_symlink()
            or raw.is_symlink()
            or snap.resolve().parent != a.snapshots.parent.resolve()
            or raw.resolve().parent != a.candidate.parent.resolve()
            or digest(snap) != sh
            or digest(raw) != rh
        ):
            raise ValueError("exact confined source snapshot and physics required")
        with np.load(snap, allow_pickle=False) as initial, np.load(raw, allow_pickle=False) as z:
            d = mujoco.MjData(model)
            mujoco.mj_setState(model, d, initial["state"], mujoco.mjtState.mjSTATE_INTEGRATION)
            mujoco.mj_forward(model, d)
            x, _ = causal_teacher_samples(
                initial_qpos=d.qpos.copy(),
                initial_qvel=d.qvel.copy(),
                initial_actuator_torque=d.actuator_force.copy(),
                qpos=z["qpos"],
                qvel=z["qvel"],
                actual_torque=z["torque"],
                parent_torque=z["teacher_parent_torque"],
                ball_forces=z["ball_forces"],
                receiver_layout=receiver,
                teammate_layout=teammate,
            )
            actual = z["torque"].reshape(-1, model.nu)[:, receiver.ai].copy()
            parent = z["teacher_parent_torque"].reshape(-1, model.nu)[:, receiver.ai].copy()
        max_error = 0.0
        with torch.no_grad():
            for i in range(10, len(x)):
                delta = actor((torch.from_numpy(x[i]) - mean) / std).numpy().astype(np.float64)
                predicted = np.clip(parent[i] + delta, -effort, effort)
                max_error = max(max_error, float(np.max(np.abs(predicted - actual[i]))))
        if max_error > 1e-6 or digest(snap) != sh or digest(raw) != rh:
            raise ValueError(
                f"actor inference/actual torque mismatch at case {row['index']}: {max_error}"
            )
        rows.append(
            dict(
                index=row["index"],
                context=row["context"],
                actor_native_inferences=len(x) - 10,
                maximum_abs_error_nm=max_error,
            )
        )
        print(json.dumps(rows[-1]), flush=True)
    if digest(a.candidate) != candidate_hash or any(digest(path) != h for path, h in pins):
        raise ValueError("source inputs changed during actor re-inference")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RECEIVER_NEURAL_TORQUE_REINFERENCE_NOT_NEW_PHYSICS",
            source_head=a.expected_head,
            candidate_hash=candidate_hash,
            checkpoint_hash=a.checkpoint_hash,
            snapshot_hash=a.snapshot_hash,
            rows=rows,
            model_inferences=sum(r["actor_native_inferences"] for r in rows),
            maximum_abs_error_nm=max(r["maximum_abs_error_nm"] for r in rows),
            saved_state_reconstruction=True,
            independent_physics_replay=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
