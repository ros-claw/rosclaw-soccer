"""Independent recorded-target physics replication and fixed tracking diagnostics.

No neural-policy replay or gradient replay is claimed. Runs must use an unused
output and a pinned checkout. Source data remains external and research-only.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def tracking_metrics(actual, reference, np):
    error = actual[:, 7:36] - reference[:, 7:36]
    groups = {"legs": slice(0, 12), "waist": slice(12, 15), "arms": slice(15, 29)}
    windows = {
        "whole": slice(0, 300),
        "entry": slice(0, 50),
        "source": slice(50, 251),
        "terminal_hold": slice(251, 300),
    }
    metrics = {}
    for window, frames in windows.items():
        metrics[window] = {
            group + "_rmse_rad": float(np.sqrt(np.mean(error[frames, channels] ** 2)))
            for group, channels in groups.items()
        }
        metrics[window]["all_rmse_rad"] = float(np.sqrt(np.mean(error[frames] ** 2)))
    metrics["per_joint_rmse_rad"] = np.sqrt(np.mean(error**2, axis=0)).tolist()
    metrics["root_xyz_is_tracking_authority"] = False
    metrics["physical_skill_qualified"] = False
    return metrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "world", "motion-csv", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--producer-unit", required=True)
    parser.add_argument("--expected-head", required=True)
    parser.add_argument("--expected-producer-head", required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
        or args.output.exists()
    ):
        raise ValueError("pinned source and unused verification root required")
    unit = dict(
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
                "SubState",
                "-p",
                "Result",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        ).splitlines()
    )
    if unit != {"MainPID": "0", "Result": "success", "ExecMainStatus": "0", "SubState": "exited"}:
        raise ValueError("completed durable successful producer required")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    if (
        declaration["source_head"] != args.expected_producer_head
        or declaration["arms"] != ["measured_pose_hold", "source_first4seconds"]
        or declaration["course"] != [10.0, 0.0, 0.0]
        or declaration["source_interval_inclusive"] != [0, 480]
        or declaration["source_frequency_hz"] != 120
        or declaration["entry_intervals"] != 50
        or complete["actual_native_steps"] != 6000
        or complete["optimizer_steps"] != 0
        or (args.producer / "failure.json").exists()
    ):
        raise ValueError("exact first4seconds two-arm feasibility protocol required")
    if any(digest(Path(p)) != h for p, h in declaration["input_file_hashes"].items()):
        raise ValueError("producer sources or assets changed")
    for path in (args.world, args.motion_csv):
        if declaration["input_file_hashes"].get(str(path)) != digest(path):
            raise ValueError("verifier world/motion differs from producer")
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    }
    for p in args.producer.iterdir():
        if p.is_file():
            pins[str(p)] = digest(p)
    for row in complete["results"]:
        if pins[str(args.producer / (row["arm"] + ".npz"))] != row["artifact_hash"]:
            raise ValueError("raw producer trajectory hash mismatch")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.timed_reference_recorded_physics_review.v1",
            input_file_hashes=pins,
            producer_terminal_state=unit,
            planned_additional_native_steps=6000,
            neural_policy_replayed=False,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    try:
        defaults = G1SonicRunupController.default_angles.copy()
        entry = np.r_[0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0, defaults]
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        options = dict(
            source_frequency_hz=120, entry_intervals=50, source_evidence_hash=hash_json(declaration)
        )
        try:
            prepare_sonic_reference(poses, entry, **options)
        except ValueError:
            pass
        else:
            raise ValueError("full-source rejection changed")
        prefix = prepare_sonic_reference(poses[:481], entry, **options)
        stored = json.loads((args.producer / "prepared-reference.json").read_text())
        if prefix.contract_hash != stored["contract_hash"]:
            raise ValueError("reference preparation does not match producer")
        hold = SonicPoseReference((tuple(float(x) for x in entry),) * 10, hash_json(declaration))
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        results = []
        for row, reference in zip(complete["results"], (hold, prefix.reference), strict=True):
            arm = row["arm"]
            with np.load(args.producer / (arm + ".npz"), allow_pickle=False) as archive:
                raw = {key: archive[key] for key in archive.files}
            np.testing.assert_array_equal(
                raw["planned_reference"], reference.initialize(entry, frames=410)[:300]
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
                actual = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in actual.items():
                    if key not in raw or value[0].shape != raw[key][frame].shape:
                        raise ValueError("missing or malformed physical field: " + key)
                    difference = float(np.max(abs(value[0] - raw[key][frame])))
                    errors[key] = max(errors.get(key, 0.0), difference)
                    if difference != 0.0:
                        raise ValueError(f"physical mismatch {arm}/{frame}/{key}: {difference}")
            metrics = tracking_metrics(raw["canonical_qpos"], raw["planned_reference"], np)
            if metrics["whole"]["all_rmse_rad"] != row["joint_reference_rmse_rad"]:
                raise ValueError("producer RMSE differs from literal readback")
            if float(raw["pelvis_z_per_substep_m"].min()) != row["minimum_native_pelvis_z_m"]:
                raise ValueError("producer minimum native pelvis height mismatch")
            np.testing.assert_array_equal(body.canonical()[0][:3], row["final_root_position_m"])
            results.append(
                dict(arm=arm, native=body.review(), maximum_errors=errors, tracking_metrics=metrics)
            )
            write_once(args.output / (arm + "-complete.json"), results[-1])
            print("RECORDED_PHYSICS_EXACT", arm, body.integrated_steps, flush=True)
        if any(digest(Path(p)) != h for p, h in pins.items()):
            raise ValueError("verifier sources or producer evidence changed")
        if any(digest(Path(p)) != h for p, h in declaration["input_file_hashes"].items()):
            raise ValueError("producer sources/assets changed during review")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_native_steps=6000,
                neural_policy_replayed=False,
                optimizer_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
                result="EXACT_RECORDED_TARGET_PHYSICS_NOT_POLICY_OR_FOOTBALL_QUALIFICATION",
            ),
        )
    except Exception as error:
        write_once(
            args.output / "failure.json", dict(error=repr(error), promotion_authorized=False)
        )
        raise


if __name__ == "__main__":
    main()
