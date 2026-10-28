"""Review pinned assisted two-player physics after its actual worker finishes."""

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
from rosclaw_soccer.rsi.receive_teacher_evidence import audit_teacher_torque
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("baseline", "candidate", "snapshots", "scene", "torque-output", "review", "team"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("baseline-hash", "snapshot-hash", "physical-head", "expected-head"):
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
    unit = a.unit
    pins = ((a.baseline, a.baseline_hash), (a.snapshots, a.snapshot_hash))
    if (
        not 1 <= a.timeout_sec <= 7200
        or any(path.is_symlink() or digest(path) != h for path, h in pins)
        or any(
            x.exists() or x.resolve().is_relative_to(Path.cwd())
            for x in (a.torque_output, a.review, a.team)
        )
    ):
        raise ValueError("exact original inputs and unused external outputs required")
    deadline = time.monotonic() + a.timeout_sec
    while not a.candidate.is_file():
        status = subprocess.run(
            ["systemctl", "--user", "show", unit, "-p", "ActiveState"],
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        if status not in ("ActiveState=active", "ActiveState=activating"):
            raise ValueError("worker ended without complete physics receipt")
        if time.monotonic() >= deadline:
            raise TimeoutError("observer expired; never restart a worker on timeout")
        time.sleep(15)
    if a.candidate.is_symlink():
        raise ValueError("confined physical receipt required")
    candidate_hash = digest(a.candidate)
    data = json.loads(a.candidate.read_text())
    snapshot_data = json.loads(a.snapshots.read_text())
    if (
        data.get("status") != "RELAY_CONTACT_TEACHER_ASSISTED_PHYSICAL_NOT_SUCCESS"
        or data.get("contact_teacher_assisted") is not True
        or data.get("contact_teacher_capture", False)
        != (unit == "rosclaw-receiver-contact-capture-b0-20261029.service")
        or data.get("source_head") != a.physical_head
        or data.get("snapshot_hash") != a.snapshot_hash
        or any(
            data.get(k) is not False
            for k in (
                "independent_fresh",
                "hardware_authorized",
                "promotion_authorized",
                "goal_complete",
                "future_controls_used",
            )
        )
        or data.get("optimizer_updates") != 0
        or len(data.get("rows", [])) != 24
        or len({x["index"] for x in data["rows"]}) != 24
        or a.scene.is_symlink()
        or digest(a.scene) != snapshot_data.get("scene_hash")
    ):
        raise ValueError("exact complete unpromoted assisted physical data required")
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    rows = []
    for row in data["rows"]:
        path = a.candidate.parent / row["physical_trajectory"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.candidate.parent.resolve()
            or digest(path) != row["physical_trajectory_hash"]
        ):
            raise ValueError("confined hashed physical trajectory required")
        with np.load(path, allow_pickle=False) as z:
            result = audit_teacher_torque(
                z["torque"],
                z["teacher_parent_torque"],
                z["teacher_actual_torque_delta"],
                receiver_actuators=np.asarray(layout.ai),
                commands=row["teacher_native_commands"],
                active_commands=row["teacher_active_native_commands"],
            )
            if len(z["torque"]) != row["consecutive_controls"]:
                raise ValueError("physics duration differs from receipt")
        if digest(path) != row["physical_trajectory_hash"]:
            raise ValueError("physical trajectory changed during audit")
        rows.append(dict(index=row["index"], **result))
    if digest(a.candidate) != candidate_hash or any(digest(path) != h for path, h in pins):
        raise ValueError("inputs changed during review")
    clean(a.expected_head)
    a.torque_output.parent.mkdir(parents=True, exist_ok=True)
    write_once(
        a.torque_output,
        dict(
            status="RECEIVER_TEACHER_TORQUE_ACCOUNTING_NOT_SUCCESS",
            source_head=a.expected_head,
            candidate_hash=candidate_hash,
            rows=rows,
            independent_physics_replay=False,
            independent_fresh=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    common = [
        "--baseline",
        str(a.baseline),
        "--baseline-hash",
        a.baseline_hash,
        "--candidate",
        str(a.candidate),
        "--candidate-hash",
        candidate_hash,
        "--snapshots",
        str(a.snapshots),
        "--snapshot-hash",
        a.snapshot_hash,
        "--scene",
        str(a.scene),
        "--expected-head",
        a.expected_head,
    ]
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_compare_long_receiver_probe",
            *common,
            "--output",
            str(a.review),
        ],
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "scripts.rsi_audit_relay_common_successor",
            *common,
            "--review",
            str(a.review),
            "--review-hash",
            digest(a.review),
            "--output",
            str(a.team),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
