"""All16 SIM kick/standing lifecycles using frozen learned motor weights.

First kick NN prefix MUST match its qualified source; subsequent states are
new continuing physics. No old formal exam/reference horizon is extended.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--source-audit-root", type=Path, required=True)
    p.add_argument("--source-audit-hash", required=True)
    p.add_argument("--variant", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--skills", type=int, choices=(1, 3), default=3)
    p.add_argument("--course-bank", choices=("train", "dev"), default="train")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean pinned lifecycle source required")
    from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof, unique_proof_path

    cases = json.loads((a.source_root / "declaration.json").read_text())["cases"]
    sd, pins = recovery_proof(
        a.source_root,
        a.source_hash,
        a.source_audit_root,
        a.source_audit_hash,
        cases=cases,
        course_bank=a.course_bank,
    )
    source = json.loads((a.source_root / "complete.json").read_text())
    if len(cases) != 16:
        raise ValueError("complete frozen sixteen-condition bank required")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    names = ("fixed-target_R2", "neural-stand-carry_R2", "neural-stand-carry_R6-successor")
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_root=str(a.source_root),
            source_hash=a.source_hash,
            source_audit_root=str(a.source_audit_root),
            source_audit_hash=a.source_audit_hash,
            course_bank=a.course_bank,
            cases=cases,
            variant=a.variant,
            variant_name=names[a.variant],
            skills=a.skills,
            standing_goal="last_admitted_reference_joint_pose_with_zero_desired_joint_and_angular_velocity",
            standing_controls=100,
            settle_max_controls=50,
            no_motion_padding=True,
            standing_memory="carry_own_same_R2_backbone" if a.variant else "none",
            standing_actor="R2_base_of_typed_contact_actor" if a.variant else "fixed_last_qtarget",
            actor_checkpoint_hash=sd["checkpoint_hash"],
            actor_training_receipt_hash=sd["training_receipt_hash"],
            first_only_qualified_reference_prefix=True,
            independent_world_repeats=2,
            body_resets=0,
            motion_reference_still_required_for_kicks=True,
            one_shared_MjData_per_world=True,
            optimizer_updates=0,
            old_exam_result=False,
            fresh_exam=False,
            team_game=False,
            hardware_authorized=False,
            promotion_authorized=False,
            commercial_publicity_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_balance_recovery import transfer_reference_prefix
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, cases)
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, sd["checkpoint_hash"]),
            checkpoint_hash=sd["checkpoint_hash"],
            training_receipt=unique_proof_path(pins, sd["training_receipt_hash"]),
            training_receipt_hash=sd["training_receipt_hash"],
            scene_hash=scene,
        )

        def read(row, root=a.source_root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined raw trajectory required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def same(x, y):
            if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
                raise ValueError("qualified first prefix or full independent repeat differs")

        worlds = []
        for i, case in enumerate(cases):
            reference = read(source["results"]["learned"]["worlds"][i]["skills"][0]["result"])
            prefix = transfer_reference_prefix(
                reference, horizon=bench.clips[case["source"]].length
            )
            replicas = []
            for repeat in range(2):
                folder = a.output / f"case{i}-repeat{repeat}"
                folder.mkdir()
                artifacts = []

                def store(
                    key, raw, folder=folder, artifacts=artifacts, prefix=prefix, reference=reference
                ):
                    if key == "kick0":
                        same(raw, prefix)
                    if a.variant == 0 and key == "stand0":
                        n = len(reference["NN_action"])
                        if any(
                            not np.array_equal(raw[k][:60], v[n:])
                            for k, v in reference.items()
                            if not k.startswith(("NN_", "initial_"))
                        ):
                            raise ValueError(
                                "zero standing changed the original first fixed-target tail"
                            )
                    path = folder / (key + ".npz")
                    if path.exists():
                        raise ValueError("trajectory overwrite forbidden")
                    np.savez_compressed(path, **raw)
                    row = dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))
                    artifacts.append(row)
                    return row

                world = run_standing_lifecycle(
                    bench, case, actor, variant=a.variant, skills=a.skills, store=store
                )
                world.update(
                    case=i,
                    artifacts=artifacts,
                    quality_by_kick=[
                        bool(contact_rank([x["result"], x["result"]])[1] > 0)
                        for x in world["kicks"]
                    ],
                )
                write_once(folder / "complete.json", world)
                replicas.append(world)
            if replicas[0]["status"] != replicas[1]["status"] or len(
                replicas[0]["artifacts"]
            ) != len(replicas[1]["artifacts"]):
                raise ValueError("complete lifecycle branches differ")
            for x, y in zip(replicas[0]["artifacts"], replicas[1]["artifacts"], strict=True):
                same(read(x, a.output), read(y, a.output))
            worlds.append(dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"]))
            print(
                json.dumps(
                    dict(
                        case=i,
                        status=world["status"],
                        quality_by_kick=world["quality_by_kick"],
                        stands_healthy=[x["result"]["healthy"] for x in world["stands"]],
                    )
                ),
                flush=True,
            )
        summary = dict(
            stable=sum(w["stable"] for w in worlds),
            quality_by_kick=[
                sum(len(w["quality_by_kick"]) > k and w["quality_by_kick"][k] for w in worlds)
                for k in range(a.skills)
            ],
            stand_healthy=[
                sum(len(w["stands"]) > k and w["stands"][k]["result"]["healthy"] for w in worlds)
                for k in range(a.skills)
            ],
            stand_certificates=[
                sum(
                    len(w["stands"]) > k and w["stands"][k]["result"]["stability_observed"]
                    for w in worlds
                )
                for k in range(a.skills)
            ],
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_NEURAL_STANDING_LIFECYCLE_CLOSED_PENDING_AUDIT",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration_hash=digest(a.output / "declaration.json"),
                worlds=worlds,
                summary=summary,
                course_bank=a.course_bank,
                variant=a.variant,
                skills=a.skills,
                all_first_prefix_and_full_repeat_arrays_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                optimizer_updates=0,
                goal_complete=False,
                old_exam_result=False,
                hardware_authorized=False,
                promotion_authorized=False,
                team_game=False,
                fresh_exam=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                native_steps=0 if bench is None else bench.native_steps,
                neural_rows=0 if bench is None else bench.neural_rows,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
