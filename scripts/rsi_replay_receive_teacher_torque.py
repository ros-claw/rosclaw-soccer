"""Reintegrate teacher-assisted saved torques; does not infer actor or teacher."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.receive_teacher_torque_replay import replay_saved_teacher_torques
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("candidate", "snapshots", "scene", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("snapshot-hash", "physical-head", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--timeout-sec", type=int, default=7200)
    p.add_argument(
        "--unit",
        choices=(
            "rosclaw-receiver-contact-teacher-b0-20261029.service",
            "rosclaw-receiver-contact-capture-b0-20261029.service",
        ),
        default="rosclaw-receiver-contact-teacher-b0-20261029.service",
    )
    a = p.parse_args()
    clean(a.expected_head)
    if (
        not 1 <= a.timeout_sec <= 7200
        or a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or a.snapshots.is_symlink()
        or digest(a.snapshots) != a.snapshot_hash
    ):
        raise ValueError("pinned snapshots and unused external output required")
    deadline = time.monotonic() + a.timeout_sec
    while not a.candidate.is_file():
        status = subprocess.run(
            [
                "systemctl",
                "--user",
                "show",
                a.unit,
                "-p",
                "ActiveState",
            ],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if status not in ("ActiveState=active", "ActiveState=activating"):
            raise ValueError("physics worker ended without complete receipt")
        if time.monotonic() >= deadline:
            raise TimeoutError("observer expired; never restart live work")
        time.sleep(15)
    if a.candidate.is_symlink():
        raise ValueError("no symlink receipt")
    candidate_hash = digest(a.candidate)
    candidate, snapshots = (json.loads(p.read_text()) for p in (a.candidate, a.snapshots))
    if (
        candidate.get("status") != "RELAY_CONTACT_TEACHER_ASSISTED_PHYSICAL_NOT_SUCCESS"
        or candidate.get("contact_teacher_assisted") is not True
        or candidate.get("contact_teacher_capture", False)
        != (a.unit == "rosclaw-receiver-contact-capture-b0-20261029.service")
        or candidate.get("source_head") != a.physical_head
        or candidate.get("snapshot_hash") != a.snapshot_hash
        or any(
            candidate.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "independent_fresh",
                "goal_complete",
            )
        )
        or a.scene.is_symlink()
        or digest(a.scene) != snapshots.get("scene_hash")
        or len(candidate.get("rows", [])) != 24
    ):
        raise ValueError("exact unpromoted assisted source and scene required")
    cases = {r["index"]: r for r in snapshots["rows"]}
    if len(cases) != 24 or {r["index"] for r in candidate["rows"]} != set(cases):
        raise ValueError("complete unique matched cases required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    rows = []
    for row in candidate["rows"]:
        case = cases[row["index"]]
        snap = a.snapshots.parent / case["snapshot"]
        trajectory = a.candidate.parent / row["physical_trajectory"]
        if any(p.is_symlink() for p in (snap, trajectory)) or (
            snap.resolve().parent != a.snapshots.parent.resolve()
            or trajectory.resolve().parent != a.candidate.parent.resolve()
            or digest(snap) != case["snapshot_hash"]
            or digest(trajectory) != row["physical_trajectory_hash"]
        ):
            raise ValueError("exact confined case snapshot and trajectory required")
        with (
            np.load(snap, allow_pickle=False) as initial,
            np.load(trajectory, allow_pickle=False) as raw,
        ):
            steps = replay_saved_teacher_torques(
                model,
                initial["state"].copy(),
                {k: raw[k].copy() for k in ("qpos", "qvel", "torque", "ball_forces")},
            )
        if (
            steps != row["consecutive_controls"] * 10
            or digest(snap) != case["snapshot_hash"]
            or digest(trajectory) != row["physical_trajectory_hash"]
        ):
            raise ValueError("complete unchanged physics reintegration required")
        rows.append(dict(index=row["index"], native_steps=steps))
        print(json.dumps(rows[-1]), flush=True)
    if (
        digest(a.candidate) != candidate_hash
        or digest(a.snapshots) != a.snapshot_hash
        or digest(a.scene) != snapshots["scene_hash"]
    ):
        raise ValueError("source changed during physics replay")
    clean(a.expected_head)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.output,
        dict(
            status="RECEIVER_TEACHER_SAVED_TORQUE_REINTEGRATED_NOT_POLICY",
            source_head=a.expected_head,
            candidate_hash=candidate_hash,
            capture_after_contact=candidate.get("contact_teacher_capture", False),
            snapshot_hash=a.snapshot_hash,
            rows=rows,
            native_steps=sum(r["native_steps"] for r in rows),
            neural_policy_reinferred=False,
            teacher_recomputed=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
