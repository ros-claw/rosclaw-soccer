"""Explicit wrist-reference ablation plus fixed native foot/ball pilot bank.

Nine H300 own-state neural episodes, zero optimizer updates. Source-only foot
kinematics chooses reference phase; no ball outcomes enter reference selection.
Not M0, not a learned champion, not a license for publicity or hardware.
"""

import argparse
import csv
import hashlib
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("motion-csv", "license", "world", "model-root", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-motion-hash", required=True)
    parser.add_argument("--expected-world-hash", required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if head != args.expected_head or args.output.exists():
        raise ValueError("sealed source and unused output required")
    paths = [
        p
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [
        args.motion_csv,
        args.license,
        args.world,
        args.model_root / "planner_sonic.onnx",
        args.model_root / "low_latency/model_encoder.onnx",
        args.model_root / "low_latency/model_decoder.onnx",
        args.model_root / "low_latency/observation_config.yaml",
    ]
    pins = {str(p): digest(p) for p in paths}
    if (
        digest(args.motion_csv) != args.expected_motion_hash
        or digest(args.world) != args.expected_world_hash
    ):
        raise ValueError("motion/world hash mismatch")

    import mujoco
    import numpy as np

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import (
        WRIST_REFERENCE_NAMES,
        quarantine_wrist_reference,
    )
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.rsi.reference_tracking_episode import run_reference_tracking_episode
    from rosclaw_soccer.rsi.source_foot_event import source_foot_event
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_verify_timed_reference import tracking_metrics

    courses = [(0.35, -0.10, 0.0), (0.50, -0.10, 0.0), (0.35, -0.22, 0.0), (0.50, -0.22, 0.0)]
    args.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.wrist_quarantine_foot_contact_pilot.v1",
        source_head=head,
        input_file_hashes=pins,
        source_frequency_hz=120,
        entry_intervals=50,
        source_phase_selector="first_max_root_local_forward_ankle_speed_all_source_frames",
        window_rule="4seconds_centered_on_source_event_capped_to_source_ends",
        source_reference_ablation="six_wrist_channels_held_at_measured_entry",
        held_channels=list(WRIST_REFERENCE_NAMES),
        remaining_source_joint_channels=23,
        original_source_repaired=False,
        original29joint_imitation_qualified=False,
        courses=[list(c) for c in courses],
        parked_course=[10.0, 0.0, 0.0],
        physical_jobs=9,
        planned_native_steps=27000,
        optimizer_steps=0,
        independent_courses=4,
        source_motion_count=1,
        original_pd_and_limits=True,
        all_robot_ball_contacts_at_500hz=True,
        licensed_source_requires_separate_publicity_review=True,
        official_football_exam=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output / "declaration.json", declaration)
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
                raise ValueError("exact named meter/wxyz/radian source required")
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, G1SonicRunupController.default_angles]
        held, receipt = quarantine_wrist_reference(poses, entry)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        ball_geom = model.geom("ball_geom")
        if (
            int(model.geom_type[ball_geom.id]) != int(mujoco.mjtGeom.mjGEOM_SPHERE)
            or abs(float(ball_geom.size[0]) - 0.11) > 1e-12
            or abs(float(model.body_mass[model.body("ball").id]) - 0.43) > 1e-12
        ):
            raise ValueError("unchanged regulation ball dimensions required")
        event = source_foot_event(model, held)
        start = min(max(event.interval - 240, 0), len(held) - 481)
        if start < 0:
            raise ValueError("at least4seconds source required")
        options = dict(
            source_frequency_hz=120, entry_intervals=50, source_evidence_hash=hash_json(declaration)
        )
        try:
            prepare_sonic_reference(poses[start : start + 481], entry, **options)
        except ValueError as error:
            original_admission = dict(status="REJECTED_UNMODIFIED_WINDOW", error=str(error))
        else:
            original_admission = dict(status="ADMITTED_UNMODIFIED_WINDOW_NOT_EXECUTED")
        prepared = prepare_sonic_reference(held[start : start + 481], entry, **options)
        write_once(
            args.output / "admission.json",
            dict(
                quarantine=receipt.__dict__,
                quarantine_hash=receipt.contract_hash,
                event=event.__dict__,
                start_frame=start,
                stop_frame_exclusive=start + 481,
                preparation_hash=prepared.contract_hash,
                reference=prepared.reference.poses,
                original_window_admission=original_admission,
                before_any_physics=True,
                source_foot_event_not_ball_contact_label=True,
            ),
        )
        hold = SonicPoseReference((tuple(float(x) for x in entry),) * 10, hash_json(declaration))
        jobs = [("parked-quarantined", (10.0, 0.0, 0.0), prepared.reference)]
        for i, course in enumerate(courses):
            jobs.extend(
                [
                    (f"course{i}-hold", course, hold),
                    (f"course{i}-quarantined", course, prepared.reference),
                ]
            )
        results = []
        for name, course, reference in jobs:
            arrays, report = run_reference_tracking_episode(
                args.model_root,
                args.world,
                world_hash=args.expected_world_hash,
                reference=reference,
                course=course,
                record_all_ball_contacts=True,
            )
            target = args.output / (name + ".npz")
            np.savez_compressed(target, **arrays)
            report.update(
                job=name,
                artifact_hash=digest(target),
                strike=summarize_reference_ball_strike(model, arrays),
                tracking_metrics=tracking_metrics(
                    arrays["canonical_qpos"], arrays["planned_reference"], np
                ),
            )
            results.append(report)
            write_once(args.output / (name + "-complete.json"), report)
            print("FOOT_CONTACT_PILOT_CLOSED", name, report["strike"], flush=True)
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise ValueError("source/assets changed during native physics")
        if sum(row["body"]["integrated_steps"] for row in results) != 27000:
            raise ValueError("native budget/accounting differs")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                actual_native_steps=27000,
                physical_jobs=9,
                independent_courses=4,
                source_motion_count=1,
                optimizer_steps=0,
                result="EXPLICIT_REFERENCE_ABLATION_CONTACT_DIAGNOSTIC_NOT_EXAM",
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
