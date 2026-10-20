"""Reintegrate all four recorded contact-reference controls and rebuild refs.

No neural-policy replay. Shared native body, preparation, adaptation and score
helpers; independent execution/storage, all physical arrays must be bit exact.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "world", "motion-csv", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in (
        "expected-head",
        "expected-producer-head",
        "expected-producer-hash",
        "producer-unit",
    ):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("pinned reader source and unused output required")
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
                "SubState",
                "-p",
                "ExecMainStatus",
            ],
            text=True,
        ).splitlines()
    )
    if terminal != {
        "MainPID": "0",
        "Result": "success",
        "SubState": "exited",
        "ExecMainStatus": "0",
    }:
        raise ValueError("closed successful producer required")
    if digest(args.producer / "complete.json") != args.expected_producer_hash:
        raise ValueError("producer closure differs")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    names = [f"course{i}-{arm}" for i in (1, 2) for arm in ("off", "lower_inward")]
    if (
        declaration["schema"] != "soccer.rsi.contact_reference_offset_pair2.v1"
        or declaration["source_head"] != args.expected_producer_head
        or [r["job"] for r in complete["results"]] != names
        or complete["actual_native_steps"] != 12000
        or complete["optimizer_steps"] != 0
        or (args.producer / "failure.json").exists()
    ):
        raise ValueError("exact whole contact-reference pair study required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer evidence/source/assets changed")
    for p in (args.world, args.motion_csv):
        if declaration["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("reader world/source binding differs")
    paths = [
        p
        for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    ]
    paths += [p for p in args.producer.iterdir() if p.is_file()]
    pins = {str(p): digest(p) for p in paths}
    for row in complete["results"]:
        if digest(args.producer / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("actual raw trajectory hash differs")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.contact_reference_offset import offset_contact_reference
    from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.contact_reference_recorded_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            planned_additional_native_steps=12000,
            neural_policy_replayed=False,
            shared_reference_and_native_helpers=True,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    try:
        model = mujoco.MjModel.from_binary_path(str(args.world))
        poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        results = []
        off_arrays = {}
        for row in complete["results"]:
            with np.load(args.producer / (row["job"] + ".npz"), allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=tuple(row["course"]),
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            reference = None
            for frame in range(300):
                if frame == row["handoff_frame"]:
                    if (
                        integration_hash(model, body.data)
                        != row["handoff_native_state_unchanged_hash"]
                    ):
                        raise ValueError("actual handoff physical state differs")
                    entry = body.canonical()[0][:36]
                    held, _ = quarantine_wrist_reference(poses, entry)
                    prepared = prepare_sonic_reference(
                        held[286:767],
                        entry,
                        source_frequency_hz=120,
                        entry_intervals=50,
                        source_evidence_hash=hash_json(declaration),
                    )
                    adapted, report = offset_contact_reference(
                        model, prepared.reference, offset_root_m=tuple(row["offset_root_m"])
                    )
                    if report != row["contact_reference_adaptation"]:
                        raise ValueError("measured-entry geometry/provenance differs")
                    reference = adapted.initialize(entry, frames=410)
                if reference is not None:
                    local = frame - row["handoff_frame"]
                    np.testing.assert_array_equal(
                        reference[local : local + 10], raw["encoder_reference_poses"][frame]
                    )
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
            if score_full_entry(model, {k: a[20:] for k, a in raw.items()}) != row["score"]:
                raise ValueError("actual score differs")
            if summarize_reference_ball_strike(model, raw) != row["strike"]:
                raise ValueError("whole contact outcome differs")
            context = row["job"].split("-", 1)[0]
            if row["job"].endswith("-off"):
                off_arrays[context] = raw
            else:
                baseline = off_arrays[context]
                deltas = {k: float(np.max(abs(raw[k] - baseline[k]))) for k in raw}
                if deltas != row["maximum_array_deltas_from_paired_off"]:
                    raise ValueError("causal numeric difference summary differs")
            results.append(
                dict(
                    job=row["job"],
                    all_native_arrays_bit_exact=True,
                    handoff_and_modified_reference_reconstructed=True,
                    native=body.review(),
                    score=row["score"],
                    neural_policy_replayed=False,
                )
            )
            write_once(args.output / (row["job"] + "-complete.json"), results[-1])
            print("CONTACT_REFERENCE_RECORDED_PHYSICS_EXACT", row["job"], flush=True)
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("source/evidence changed during reader execution")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_native_steps=12000,
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
