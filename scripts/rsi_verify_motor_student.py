"""Fixed three-seed dispatch or full initial/trained NN+native replay. SIM only."""

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path


def admit(unit, root, expected_head):
    from scripts.rsi_native_context_recovery_common import digest

    terminal = dict(
        line.split("=", 1)
        for line in subprocess.check_output(
            [
                "systemctl",
                "--user",
                "show",
                unit,
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
    if terminal != dict(MainPID="0", SubState="exited", Result="success", ExecMainStatus="0"):
        raise ValueError("evaluation must actually be terminal; never restart a live worker")
    declaration = json.loads((root / "declaration.json").read_text())
    complete = json.loads((root / "complete.json").read_text())
    if (
        declaration["source_head"] != expected_head
        or declaration["schema"] != "soccer.rsi.recurrent_motor_physics40.v1"
        or complete["seed"] != declaration["seed"]
        or complete["seed"] not in (17, 29, 43)
        or len(complete["results"]) != 80
        or complete["optimizer_updates_during_evaluation"] != 0
    ):
        raise ValueError("full80 initial/trained fixed-seed evaluation required")
    pins = dict(declaration["input_file_hashes"])
    bank_proofs = [Path(p) for p, h in pins.items() if h == declaration["bank_complete_hash"]]
    if len(bank_proofs) != 1:
        raise ValueError("one bound curated bank required")
    bank = json.loads((bank_proofs[0].parent / "declaration.json").read_text())
    for row, (variant, index) in zip(
        complete["results"],
        [(v, i) for v in ("initial", "trained") for i in range(40)],
        strict=True,
    ):
        folder = root / f"{variant}-course{index:02d}"
        if (
            row["variant"] != variant
            or row["course"] != bank["rows"][index]["course"]
            or row["split"] != bank["rows"][index]["split"]
            or row != json.loads((folder / "complete.json").read_text())
        ):
            raise ValueError("exact ordered all40 identity for BOTH weights required")
        if row["status"] == "COMPLETE":
            p = (folder / row["artifact"]).resolve()
            if (
                not p.is_relative_to(folder.resolve())
                or digest(p) != row["artifact_hash"]
                or row["native_steps"] != 3000
                or row["student_nn_calls"] != 300
                or row["student_actions_consumed"] != 280
                or row["teacher_actions_consumed_after_boundary"] != 0
            ):
                raise ValueError("complete actual student artifact/counter contract required")
        elif row["status"] != "EXECUTION_FAILED":
            raise ValueError("unknown controller execution status")
    if complete["new_native_steps"] != sum(r["native_steps"] for r in complete["results"]):
        raise ValueError("complete actual physical step sum differs")
    pins.update({str(p): digest(p) for p in root.rglob("*") if p.is_file()})
    return declaration, complete, pins


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("dispatch", "replay"), required=True)
    parser.add_argument(
        "--producer", nargs=2, action="append", required=True, metavar=("UNIT", "ROOT")
    )
    for key in ("world", "core-source", "output"):
        parser.add_argument("--" + key, type=Path, required=True)
    for key in ("expected-head", "expected-producer-head"):
        parser.add_argument("--" + key, required=True)
    parser.add_argument("--expected-producer-hash")
    parser.add_argument("--unit-prefix")
    args = parser.parse_args()
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    if (
        head != args.expected_head
        or args.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain", "--untracked-files=no"])
    ):
        raise ValueError("clean sealed verifier and unused output required")
    if (
        args.mode == "dispatch"
        and (
            len(args.producer) != 3
            or args.unit_prefix is None
            or re.fullmatch(r"[a-z0-9][a-z0-9-]{0,90}", args.unit_prefix) is None
        )
        or args.mode == "replay"
        and (len(args.producer) != 1 or args.expected_producer_hash is None)
    ):
        raise ValueError("exact mode-specific inputs required")
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    pins = {}
    sources = []
    seeds = set()
    units = set()
    for unit, path in args.producer:
        root = Path(path)
        if args.mode == "replay" and digest(root / "complete.json") != args.expected_producer_hash:
            raise ValueError("physical evaluation completion identity differs")
        d, c, p = admit(unit, root, args.expected_producer_head)
        if c["seed"] in seeds or unit in units:
            raise ValueError("unique seed and evaluation unit required")
        seeds.add(c["seed"])
        units.add(unit)
        pins.update(p)
        sources.append((unit, root, d, c))
    if args.mode == "dispatch" and seeds != {17, 29, 43}:
        raise ValueError("all three declared seeds required")
    check_pins(pins)
    if pins.get(str(args.world)) != digest(args.world):
        raise ValueError("original compiled native world required")
    # Model implementation must stay identical to that which trained the tensors.
    neural = Path.cwd() / "src/rosclaw_soccer/rsi/neural_motor_student.py"
    if {h for p, h in pins.items() if Path(p).name == "neural_motor_student.py"} != {
        digest(neural)
    }:
        raise ValueError("motor network implementation changed after training")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", args.core_source):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})
    from scripts.rsi_atomic_artifacts import write_once

    args.output.mkdir(parents=True, exist_ok=False)
    write_once(
        args.output / "declaration.json",
        dict(
            schema="soccer.rsi.motor_neural_native_" + args.mode + ".v1",
            source_head=head,
            input_file_hashes=pins,
            producer_complete_hashes={
                str(r / "complete.json"): digest(r / "complete.json") for _, r, _, _ in sources
            },
            prefix_foundation_neural_math_replayed=False,
            student_neural_math_replayed=args.mode == "replay",
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    launches = []
    results = []
    total = 0
    active = None
    try:
        if args.mode == "dispatch":
            for unit, root, _, c in sorted(sources, key=lambda x: x[3]["seed"]):
                seed = c["seed"]
                name = f"{args.unit_prefix}-seed{seed}"
                output = args.output / f"replay-seed{seed}"
                command = [
                    "systemd-run",
                    "--user",
                    "--no-block",
                    "--unit=" + name,
                    "--property=Type=oneshot",
                    "--property=RemainAfterExit=yes",
                    "--property=MemoryMax=6G",
                    "--property=NoNewPrivileges=yes",
                    "--working-directory=" + str(Path.cwd()),
                    "--setenv=PYTHONPATH=src:.:" + str(args.core_source),
                    "--setenv=OMP_NUM_THREADS=1",
                    "--setenv=OPENBLAS_NUM_THREADS=1",
                    "--setenv=MKL_NUM_THREADS=1",
                    "--setenv=CUDA_VISIBLE_DEVICES=",
                    sys.executable,
                    "scripts/rsi_verify_motor_student.py",
                    "--mode",
                    "replay",
                    "--expected-head",
                    head,
                    "--expected-producer-head",
                    args.expected_producer_head,
                    "--producer",
                    unit,
                    str(root),
                    "--expected-producer-hash",
                    digest(root / "complete.json"),
                    "--world",
                    str(args.world),
                    "--core-source",
                    str(args.core_source),
                    "--output",
                    str(output),
                ]
                subprocess.run(command, check=True)
                item = dict(
                    unit=name + ".service",
                    output=str(output),
                    producer_complete_hash=digest(root / "complete.json"),
                )
                launches.append(item)
                write_once(args.output / f"launch-seed{seed}.json", item)
            check_pins(pins)
            write_once(
                args.output / "complete.json",
                dict(
                    status="THREE_SEED_NN_NATIVE_REPLAY_DISPATCHED_NOT_VERIFIED",
                    launched_reviews=launches,
                    new_native_steps=0,
                    goal_complete=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            return
        _, root, d, c = sources[0]
        if any(row["status"] != "COMPLETE" for row in c["results"]):
            write_once(
                args.output / "complete.json",
                dict(
                    status="FULL_POLICY_QUALIFICATION_REJECTED_EXECUTION_FAILURE",
                    seed=c["seed"],
                    failed_conditions=[
                        dict(course=r["course"], variant=r["variant"])
                        for r in c["results"]
                        if r["status"] != "COMPLETE"
                    ],
                    additional_native_steps=0,
                    all80_verified=False,
                    promotion_authorized=False,
                    hardware_authorized=False,
                ),
            )
            return
        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
        from rosclaw_soccer.providers.g1.sonic_runup import (
            G1SonicRunupController,
            _sonic_control_parameters,
        )
        from rosclaw_soccer.rsi.bilateral_entry import observed_contact_foot
        from rosclaw_soccer.rsi.entry_prefix_diagnostics import diagnose_entry_prefix
        from rosclaw_soccer.rsi.full_entry_forecast import score_full_entry
        from rosclaw_soccer.rsi.neural_motor_student import create_motor_student
        from scripts.rsi_motor_student_rollout import replay_student, validate_student_recording

        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        training_paths = [Path(p) for p, h in pins.items() if h == d["training_complete_hash"]]
        if len(training_paths) != 1:
            raise ValueError("one bound trained motor receipt required")
        tc = json.loads(training_paths[0].read_text())
        from rosclaw_soccer.rsi.motor_training_protocol import checkpoint_step_pair

        protocol = d.get("training_protocol", "bc3000")
        initial_steps, final_steps = checkpoint_step_pair(protocol)
        if (
            tc.get("training_protocol", "bc3000") != protocol
            or tc["optimizer_steps"] != final_steps
        ):
            raise ValueError("actual fixed training protocol differs")
        models = {}
        weight_hashes = {}
        for variant, h in (
            ("initial", tc["initial_checkpoint_hash"]),
            ("trained", tc["checkpoint_hash"]),
        ):
            paths = [Path(p) for p, value in pins.items() if value == h]
            if len(paths) != 1 or digest(paths[0]) != h:
                raise ValueError("unique unchanged motor checkpoint required")
            payload = torch.load(paths[0], map_location="cpu", weights_only=True)
            if (
                payload["optimizer_steps"]
                != (initial_steps if variant == "initial" else final_steps)
                or payload["seed"] != c["seed"]
                or payload.get("training_protocol", "bc3000") != protocol
            ):
                raise ValueError("seed/update checkpoint provenance differs")
            model = create_motor_student(hidden_size=128, action_limit=8.0)
            model.load_state_dict(payload["state_dict"], strict=True)
            model.eval()
            models[variant] = model
            weight_hashes[variant] = h
        world = mujoco.MjModel.from_binary_path(str(args.world))
        kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
        for row in c["results"]:
            variant = row["variant"]
            index = row["course"]["course_index"]
            folder = root / f"{variant}-course{index:02d}"
            if row["weights_hash"] != weight_hashes[variant]:
                raise ValueError("actual controller weights differ")
            with np.load(folder / row["artifact"], allow_pickle=False) as z:
                trace = {k: z[k].copy() for k in z.files}
            validate_student_recording(trace, nbody=world.nbody)
            active = G1TargetEpisode(
                world,
                world_hash=digest(args.world),
                kp=kp,
                kd=kd,
                course=tuple(row["course"]["course"]),
                defaults=G1SonicRunupController.default_angles.copy(),
                record_all_ball_contacts=True,
            )
            replay = replay_student(models[variant], active, trace)
            if active.review() != row["body"]:
                raise ValueError("body accounting differs")
            pd = diagnose_entry_prefix(world, {k: v[:20] for k, v in trace.items()})
            score = score_full_entry(world, {k: v[20:] for k, v in trace.items()})
            score["original_suffix_qualified"] = score["qualified"]
            score["common_prefix_physical_safe"] = pd["prefix_physical_safe"]
            if not pd["prefix_physical_safe"]:
                score["qualified"], score["cost"], score["physical_prefix_and_suffix_safe"] = (
                    False,
                    None,
                    False,
                )
            if (
                score != row["score"]
                or pd != row["prefix_diagnostics"]
                or observed_contact_foot(world, trace, global_start_frame=0)
                != row["observed_contact_foot"]
                or row["causal_student_first_touch"]
                != bool(score["qualified"] and pd["post_boundary_control_can_cause_first_touch"])
            ):
                raise ValueError("actual score, safety or NN touch attribution differs")
            result = dict(
                course=row["course"],
                variant=variant,
                split=row["split"],
                score=score,
                causal_student_first_touch=row["causal_student_first_touch"],
                **replay,
            )
            results.append(result)
            total += active.integrated_steps
            active = None
            write_once(args.output / f"{variant}-course{index:02d}.json", result)
            print(
                "MOTOR_NN_NATIVE_REPLAY_EXACT",
                c["seed"],
                variant,
                index,
                score["qualified"],
                flush=True,
            )
        summaries = dict(
            initial_qualified=sum(
                r["variant"] == "initial" and r["score"]["qualified"] for r in results
            ),
            trained_qualified=sum(
                r["variant"] == "trained" and r["score"]["qualified"] for r in results
            ),
            initial_causal_qualified=sum(
                r["variant"] == "initial" and r["causal_student_first_touch"] for r in results
            ),
            trained_causal_qualified=sum(
                r["variant"] == "trained" and r["causal_student_first_touch"] for r in results
            ),
        )
        if total != c["new_native_steps"] or any(c[k] != v for k, v in summaries.items()):
            raise ValueError("all80 outcomes/accounting differ")
        check_pins(pins)
        write_once(
            args.output / "complete.json",
            dict(
                status="ALL80_STUDENT_NN_AND_NATIVE_REPLAY_VERIFIED",
                seed=c["seed"],
                results=results,
                training_protocol=protocol,
                initial_checkpoint_optimizer_steps=initial_steps,
                trained_checkpoint_optimizer_steps=final_steps,
                additional_native_steps=total,
                optimizer_updates=0,
                promotion_authorized=False,
                hardware_authorized=False,
                **summaries,
            ),
        )
    except Exception as error:
        if active is not None:
            import numpy as np

            np.savez_compressed(
                args.output / "last-native-state.npz",
                qpos=active.data.qpos.copy(),
                qvel=active.data.qvel.copy(),
                time_sec=np.array([active.data.time]),
            )
        write_once(
            args.output / "failure.json",
            dict(
                error=repr(error),
                completed_cases=len(results),
                completed_native_steps=total,
                current_attempted_steps=active.attempted_steps if active is not None else 0,
                current_integrated_steps=active.integrated_steps if active is not None else 0,
                launched_reviews=launches,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
