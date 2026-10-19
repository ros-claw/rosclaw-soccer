"""New-process recorded-PD replication of the fixed dynamic reference bank.

This does not rerun neural inference or gradients. Admission, references and
all physical arrays are checked; source rejection cannot be hidden by a pass.
"""

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
    for name in ("producer", "motion-directory", "world", "core-source", "output"):
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
        raise ValueError("durable exited successful producer required")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    admission = json.loads((args.producer / "admission.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    if (
        declaration["schema"] != "soccer.rsi.dynamic_motion_bank16.v1"
        or declaration["source_head"] != args.expected_producer_head
        or declaration["source_names"] != [f"BGI_Shooting_{i:05}.csv" for i in range(1, 17)]
        or declaration["course"] != [10.0, 0.0, 0.0]
        or complete["actual_native_steps"] != 3000 * admission["admitted_count"]
        or complete["optimizer_steps"] != 0
        or [row["source"] for row in admission["rows"]] != declaration["source_names"]
        or complete["admitted_count"] != admission["admitted_count"]
        or complete["rejected_count"] != 16 - admission["admitted_count"]
        or (args.producer / "failure.json").exists()
    ):
        raise ValueError("exact complete dynamic16 protocol required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer source/assets changed")
    if declaration["input_file_hashes"].get(str(args.world)) != digest(args.world):
        raise ValueError("verifier world differs")
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    }
    pins.update({str(p): digest(p) for p in args.producer.iterdir() if p.is_file()})
    for row in complete["results"]:
        if digest(args.producer / (Path(row["source"]).stem + ".npz")) != row["artifact_hash"]:
            raise ValueError("trajectory hash mismatch")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.sonic_motion_window import select_dynamic_leg_window
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_verify_timed_reference import tracking_metrics

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            producer_terminal=terminal,
            schema="soccer.rsi.dynamic_reference_bank_recorded_review.v1",
            planned_additional_native_steps=complete["actual_native_steps"],
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
        defaults = G1SonicRunupController.default_angles.copy()
        entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, defaults]
        references = {}
        for row in admission["rows"]:
            path = args.motion_directory / row["source"]
            if declaration["input_file_hashes"].get(str(path)) != digest(path):
                raise ValueError("motion directory/source binding mismatch")
            try:
                with path.open(newline="") as stream:
                    if tuple(next(csv.reader(stream))) != expected:
                        raise ValueError("exact meter/wxyz/radian canonical29 columns required")
                values = np.loadtxt(path, delimiter=",", skiprows=1, dtype=np.float64)
                window = select_dynamic_leg_window(
                    values, entry, source_evidence_hash=hash_json(declaration)
                )
            except ValueError as error:
                if row["status"] != "REJECTED_SOURCE_WINDOW" or row["error"] != str(error):
                    raise ValueError("rejected source admission differs") from error
                continue
            if (
                row["status"] != "ADMITTED_GEOMETRY_ONLY"
                or row["preparation_hash"] != window.prepared.contract_hash
                or row["start_frame"] != window.source_start_frame
                or row["stop_frame_exclusive"] != window.source_stop_frame_exclusive
                or row["peak_interval"] != window.source_peak_leg_interval
                or row["peak_leg_speed_rad_per_sec"] != window.source_peak_leg_speed_rad_per_sec
            ):
                raise ValueError("selected source admission differs")
            np.testing.assert_array_equal(row["reference"], window.prepared.reference.poses)
            references[path.name] = window.prepared.reference
        if set(references) != {r["source"] for r in complete["results"]}:
            raise ValueError("producer omitted or duplicated admitted sources")
        if len(complete["results"]) != len(references) or len(admission["rows"]) != 16:
            raise ValueError("exact admission and unique complete row count required")
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        results = []
        for row in complete["results"]:
            with np.load(
                args.producer / (Path(row["source"]).stem + ".npz"), allow_pickle=False
            ) as archive:
                raw = {k: archive[k] for k in archive.files}
            np.testing.assert_array_equal(
                raw["planned_reference"],
                references[row["source"]].initialize(entry, frames=410)[:300],
            )
            model = mujoco.MjModel.from_binary_path(str(args.world))
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=(10.0, 0.0, 0.0),
                defaults=defaults,
            )
            errors = {}
            for frame in range(300):
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
                    errors[key] = 0.0
            metrics = tracking_metrics(raw["canonical_qpos"], raw["planned_reference"], np)
            if metrics != row["tracking_metrics"]:
                raise ValueError("literal body group metrics differ")
            if metrics["whole"]["all_rmse_rad"] != row["joint_reference_rmse_rad"]:
                raise ValueError("whole joint RMSE differs")
            if float(raw["pelvis_z_per_substep_m"].min()) != row["minimum_native_pelvis_z_m"]:
                raise ValueError("minimum pelvis height differs")
            np.testing.assert_array_equal(body.canonical()[0][:3], row["final_root_position_m"])
            results.append(dict(source=row["source"], native=body.review(), maximum_errors=errors))
            print("DYNAMIC_BANK_RECORDED_PHYSICS_EXACT", row["source"], flush=True)
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("pinned sources/assets/evidence changed")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                source_admission_rows_verified=16,
                additional_native_steps=3000 * len(results),
                neural_policy_replayed=False,
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
