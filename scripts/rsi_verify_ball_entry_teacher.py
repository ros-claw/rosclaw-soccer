"""Re-execute the eight-body entry pilot; verify entry logic and handoff history.

Recorded target physics, not independent neural-policy replay. Pre-handoff
navigation reference generation is NOT independently reexecuted by this reader.
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from types import SimpleNamespace


def digest(path):
    return "sha256:" + hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("producer", "prior", "world", "motion-csv", "core-source", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("producer-unit", "expected-head", "expected-producer-head"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    source = Path.cwd().resolve()
    if (
        args.output.exists()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != args.expected_head
    ):
        raise ValueError("pinned source and unused verification root required")
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
        raise ValueError("durable successfully exited producer required")
    declaration = json.loads((args.producer / "declaration.json").read_text())
    complete = json.loads((args.producer / "complete.json").read_text())
    stored_affordance = json.loads((args.producer / "entry-affordance.json").read_text())
    courses = [(0.35, -0.10, 0.0), (0.50, -0.10, 0.0), (0.35, -0.22, 0.0), (0.50, -0.22, 0.0)]
    jobs = [
        (f"course{i}-{arm}", course, arm)
        for i, course in enumerate(courses)
        for arm in ("fixed_reference", "ball_entry")
    ]
    if (
        declaration["schema"] != "soccer.rsi.measured_ball_entry_teacher4.v1"
        or declaration["source_head"] != args.expected_producer_head
        or declaration["courses"] != [list(c) for c in courses]
        or [r["job"] for r in complete["results"]] != [j[0] for j in jobs]
        or complete["actual_native_steps"] != 24000
        or complete["optimizer_steps"] != 0
        or (args.producer / "failure.json").exists()
    ):
        raise ValueError("exact complete entry pilot required")
    for p, h in declaration["input_file_hashes"].items():
        if digest(Path(p)) != h:
            raise ValueError("producer source/assets/evidence changed")
    for p in (args.world, args.motion_csv, args.prior / "parked-quarantined.npz"):
        if declaration["input_file_hashes"].get(str(p)) != digest(p):
            raise ValueError("reader world/source/prior binding mismatch")
    pins = {
        str(p): digest(p)
        for folder in (source / "src", source / "scripts", args.core_source)
        for p in folder.rglob("*.py")
    }
    pins.update({str(p): digest(p) for p in args.producer.iterdir() if p.is_file()})
    for row in complete["results"]:
        if digest(args.producer / (row["job"] + ".npz")) != row["artifact_hash"]:
            raise ValueError("raw trajectory hash mismatch")

    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.providers.g1.sonic_reference_preparation import prepare_sonic_reference
    from rosclaw_soccer.providers.g1.sonic_runup import (
        G1SonicRunupController,
        _sonic_control_parameters,
    )
    from rosclaw_soccer.providers.g1.sonic_wrist_quarantine import quarantine_wrist_reference
    from rosclaw_soccer.rsi.ball_entry_teacher import ball_entry_decision, derive_entry_affordance
    from rosclaw_soccer.rsi.full_task_branch import integration_hash
    from rosclaw_soccer.rsi.reference_ball_strike_metrics import summarize_reference_ball_strike
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.ball_entry_teacher_recorded_review.v1",
            input_file_hashes=pins,
            producer_terminal=terminal,
            planned_additional_native_steps=24000,
            neural_policy_replayed=False,
            navigation_planner_reexecuted=False,
            optimizer_steps=0,
            hardware_authorized=False,
        ),
    )
    try:
        source_poses = np.loadtxt(args.motion_csv, delimiter=",", skiprows=1, dtype=np.float64)
        model = mujoco.MjModel.from_binary_path(str(args.world))
        with np.load(args.prior / "parked-quarantined.npz", allow_pickle=False) as archive:
            affordance = derive_entry_affordance(
                model,
                archive["canonical_qpos"],
                artifact_hash=digest(args.prior / "parked-quarantined.npz"),
                foot="right",
            )
        if affordance.contract_hash != stored_affordance["contract_hash"]:
            raise ValueError("historical entry affordance differs")
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        defaults = G1SonicRunupController.default_angles.copy()
        normalizer = object.__new__(G1SonicRunupController)
        results = []
        for row, (name, course, arm) in zip(complete["results"], jobs, strict=True):
            with np.load(args.producer / (name + ".npz"), allow_pickle=False) as archive:
                raw = {k: archive[k] for k in archive.files}
            body = G1TargetEpisode(
                model,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=course,
                defaults=defaults,
                record_all_ball_contacts=True,
            )
            state = 1 if arm == "fixed_reference" else 0
            target, error = np.zeros(2), 0.0
            handoff_frame = None
            prepared_values = None
            if arm == "fixed_reference":
                held, _ = quarantine_wrist_reference(source_poses, raw["canonical_qpos"][0, :36])
                ref = prepare_sonic_reference(
                    held[286:767],
                    raw["canonical_qpos"][0, :36],
                    source_frequency_hz=120,
                    entry_intervals=50,
                    source_evidence_hash=hash_json(declaration),
                )
                prepared_values = ref.reference.initialize(
                    raw["canonical_qpos"][0, :36], frames=410
                )
            errors = {}
            for frame in range(300):
                q, v = raw["canonical_qpos"][frame], raw["canonical_qvel"][frame]
                command = (0.0, 0.0, 0.0)
                if state in (0, 2):
                    decision = ball_entry_decision(q, v, frame=frame, affordance=affordance)
                    command, target, error = (
                        decision.command,
                        np.asarray(decision.target_root_xy_m),
                        decision.horizontal_error_m,
                    )
                    if decision.timed_out:
                        state = 2
                    if decision.ready:
                        handoff_frame, state = frame, 1
                        if (
                            integration_hash(model, body.data)
                            != row["handoff_native_state_unchanged_hash"]
                        ):
                            raise ValueError(
                                "handoff native state differs from independent physical replay"
                            )
                        if body.data.time != row["handoff_receipt"]["time_sec"]:
                            raise ValueError("handoff changed global clock")
                        history = [
                            normalizer._history_entry(
                                SimpleNamespace(
                                    qpos=raw["canonical_qpos"][i], qvel=raw["canonical_qvel"][i]
                                ),
                                raw["neural_action_isaac"][i - 1],
                            )
                            for i in range(frame - 9, frame + 1)
                        ]
                        if (
                            hash_json([[x.tolist() for x in h] for h in history])
                            != row["handoff_receipt"]["history_hash"]
                        ):
                            raise ValueError("handoff dropped/duplicated measured neural history")
                        held, _ = quarantine_wrist_reference(source_poses, q[:36])
                        ref = prepare_sonic_reference(
                            held[286:767],
                            q[:36],
                            source_frequency_hz=120,
                            entry_intervals=50,
                            source_evidence_hash=hash_json(declaration),
                        )
                        prepared_values = ref.reference.initialize(q[:36], frames=410)
                np.testing.assert_array_equal(raw["entry_command"][frame], command)
                np.testing.assert_array_equal(raw["entry_target_root_xy_m"][frame], target)
                if raw["entry_error_m"][frame] != error or raw["entry_stage"][frame] != state:
                    raise ValueError("causal entry decision trace differs")
                if prepared_values is not None:
                    local = frame - (handoff_frame if handoff_frame is not None else 0)
                    np.testing.assert_array_equal(
                        raw["encoder_reference_poses"][frame], prepared_values[local : local + 10]
                    )
                physical = body.step(frame, raw["joint_target_rad"][frame].copy())
                for key, value in physical.items():
                    np.testing.assert_array_equal(value[0], raw[key][frame])
                    errors[key] = 0.0
            if handoff_frame != row["handoff_frame"] or row["entry_timed_out"] != (state == 2):
                raise ValueError("handoff/timeout summary differs")
            strike = summarize_reference_ball_strike(model, raw)
            if strike != row["strike"]:
                raise ValueError("literal strike summary differs")
            if float(raw["pelvis_z_per_substep_m"].min()) != row["minimum_native_pelvis_z_m"]:
                raise ValueError("minimum native pelvis differs")
            np.testing.assert_array_equal(body.canonical()[0][:3], row["final_root_position_m"])
            results.append(
                dict(
                    job=name,
                    native=body.review(),
                    maximum_errors=errors,
                    handoff_frame=handoff_frame,
                    strike=strike,
                )
            )
            write_once(args.output / (name + "-complete.json"), results[-1])
            print("ENTRY_TEACHER_RECORDED_PHYSICS_EXACT", name, flush=True)
        for p, h in {**declaration["input_file_hashes"], **pins}.items():
            if digest(Path(p)) != h:
                raise ValueError("source/assets/evidence changed during verification")
        write_once(
            args.output / "complete.json",
            dict(
                results=results,
                additional_native_steps=24000,
                entry_decisions_reconstructed=True,
                handoff_history_and_native_state_checked=True,
                neural_policy_replayed=False,
                navigation_planner_reexecuted=False,
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
