"""Replicate the declared nine-body reference/contact pilot, not its neural policy."""

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "motion-csv", "world", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--producer-unit", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-producer-head", required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("pinned verifier and unused output required")
    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                args.producer_unit,
                "-p",
                "MainPID",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
                "-p",
                "SubState",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != {
        "MainPID": "0",
        "Result": "success",
        "ExecMainStatus": "0",
        "SubState": "exited",
    }:
        raise ValueError("durable successful exited producer required")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    admission = json.loads((args.producer / "admission.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    courses = [(0.35, -0.10, 0.0), (0.50, -0.10, 0.0), (0.35, -0.22, 0.0), (0.50, -0.22, 0.0)]
    jobs = [("parked-quarantined", (10.0, 0.0, 0.0), "quarantined")]
    for i, course in enumerate(courses):
        jobs.extend(
            [(f"course{i}-hold", course, "hold"), (f"course{i}-quarantined", course, "quarantined")]
        )
    if (
        declaration["schema"] != "soccer.rsi.wrist_quarantine_foot_contact_pilot.v1"
        or declaration["source_head"] != args.expected_producer_head
        or declaration["courses"] != [list(c) for c in courses]
        or declaration["source_frequency_hz"] != 120
        or declaration["entry_intervals"] != 50
        or complete["actual_native_steps"] != 27000
        or complete["optimizer_steps"] != 0
        or complete["physical_jobs"] != 9
        or [r["job"] for r in complete["results"]] != [j[0] for j in jobs]
        or (args.producer / "failure.json").exists()
    ):
        raise ValueError("exact complete nine-body paired pilot required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer source/assets changed")
    for path in (args.world, args.motion_csv):
        if declaration["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("world/motion differs from producer")
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    }
    pins.update({str(p): digest(p) for p in args.producer.iterdir() if p.is_file()})
    for row in complete["results"]:
        if digest(args.producer / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("trajectory hash differs")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.rsi.source_foot_event import source_foot_event
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_verify_timed_reference import tracking_metrics

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.wrist_reference_contact_recorded_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            planned_additional_native_steps=27000,
            neural_policy_replayed=False,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    try:
        expected = (
            "root_pos_x(m)",
            "root_pos_y(m)",
            "root_pos_z(m)",
            "root_rot_w",
            "root_rot_x",
            "root_rot_y",
            "root_rot_z",
        )
        expected += tuple(f"dof_{n}(rad)" for n in G1_DDS_JOINT_NAMES)
        with args.motion_csv.open(newline="") as stream:
            if tuple(next(csv.reader(stream))) != expected:
                raise ValueError("source channel ABI mismatch")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        defaults = G1SonicRunupController.default_angles.copy()
        entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, defaults]
        held, receipt = quarantine_wrist_reference(poses, entry)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        event = source_foot_event(model, held)
        start = min(max(event.interval - 240, 0), len(held) - 481)
        if (
            hash_json(receipt.__dict__) != hash_json(admission["quarantine"])
            or receipt.contract_hash != admission["quarantine_hash"]
            or hash_json(event.__dict__) != hash_json(admission["event"])
            or start != admission["start_frame"]
            or start + 481 != admission["stop_frame_exclusive"]
        ):
            raise ValueError("explicit source quarantine/event/window differs")
        options = dict(
            source_frequency_hz=120, entry_intervals=50, source_evidence_hash=hash_json(declaration)
        )
        try:
            prepare_sonic_reference(poses[start : start + 481], entry, **options)
        except ValueError as error:
            original_admission = dict(status="REJECTED_UNMODIFIED_WINDOW", error=str(error))
        else:
            original_admission = dict(status="ADMITTED_UNMODIFIED_WINDOW_NOT_EXECUTED")
        if original_admission != admission["original_window_admission"]:
            raise ValueError("original rejected source state differs")
        prepared = prepare_sonic_reference(held[start : start + 481], entry, **options)
        if prepared.contract_hash != admission["preparation_hash"]:
            raise ValueError("prepared reference hash differs")
        np.testing.assert_array_equal(prepared.reference.poses, admission["reference"])
        hold = SonicPoseReference((tuple(float(x) for x in entry),) * 10, hash_json(declaration))
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        results = []
        for row, (name, course, kind) in zip(complete["results"], jobs, strict=True):
            if row["course"] != list(course) or row["all_ball_contacts_recorded"] is not True:
                raise ValueError("course/contact capture differs")
            reference = hold if kind == "hold" else prepared.reference
            with np.load(args.producer / (name + ".npz"), allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            np.testing.assert_array_equal(
                raw["planned_reference"], reference.initialize(entry, frames=410)[:300]
            )
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=course,
                defaults=defaults,
                record_all_ball_contacts=True,
            )
            errors = {}
            for frame in range(300):
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
                    errors[key] = 0.0
            strike = summarize_reference_ball_strike(model, raw)
            metrics = tracking_metrics(raw["canonical_qpos"], raw["planned_reference"], np)
            if strike != row["strike"] or metrics != row["tracking_metrics"]:
                raise ValueError("literal contact/tracking metrics differ")
            if metrics["whole"]["all_rmse_rad"] != row["joint_reference_rmse_rad"]:
                raise ValueError("literal whole tracking RMSE differs")
            if float(raw["pelvis_z_per_substep_m"].min()) != row["minimum_native_pelvis_z_m"]:
                raise ValueError("minimum pelvis differs")
            np.testing.assert_array_equal(body.canonical()[0][:3], row["final_root_position_m"])
            results.append(
                dict(job=name, native=body.review(), maximum_errors=errors, strike=strike)
            )
            write_once(args.output / (name + "-complete.json"), results[-1])
            print("FULL_CONTACT_RECORDED_PHYSICS_EXACT", name, flush=True)
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("source/assets/evidence changed during verification")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_native_steps=27000,
                neural_policy_replayed=False,
                source_phase_is_ball_outcome_label=False,
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json", dict(error=repr(error), promotion_authorized=False)
        )
        raise


if __name__ == "__main__":
    main()
