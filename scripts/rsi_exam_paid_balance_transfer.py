"""Transfer ONE frozen balance feedback model to all TRAIN or DEV16 worlds.

Both arms execute the same learned kick actors. Only the recovery gains differ
(zero vs one frozen teacher), not a per-case oracle or new model search.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "teacher-root",
        "teacher-audit-root",
        "source-root",
        "source-audit-root",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "teacher-hash",
        "teacher-audit-hash",
        "source-hash",
        "source-audit-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--course-bank", choices=("train", "dev"), default="train")
    for name in ("development-root", "development-audit-root"):
        p.add_argument("--" + name, type=Path)
    for name in ("development-hash", "development-audit-hash"):
        p.add_argument("--" + name)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean pinned transfer source required")
    from rosclaw_soccer.rsi.paid_balance_teacher_proof import balance_teacher_proof
    from rosclaw_soccer.rsi.paid_recovery_proof import recovery_proof, unique_proof_path

    teacher, td, pins, latent = balance_teacher_proof(
        a.teacher_root, a.teacher_hash, a.teacher_audit_root, a.teacher_audit_hash
    )
    train_cases = json.loads((a.source_root / "declaration.json").read_text())["cases"]
    train_sd, sp = recovery_proof(
        a.source_root, a.source_hash, a.source_audit_root, a.source_audit_hash, cases=train_cases
    )
    if (
        train_cases != td["source_cases"]
        or teacher["input_file_hashes"].get(str(a.source_root / "complete.json")) != a.source_hash
        or teacher["input_file_hashes"].get(str(a.source_audit_root / "complete.json"))
        != a.source_audit_hash
    ):
        raise ValueError("exact teacher training source and motor lineage required")
    target_root, target_hash, target_audit_root, target_audit_hash = (
        a.source_root,
        a.source_hash,
        a.source_audit_root,
        a.source_audit_hash,
    )
    sd, cases = train_sd, train_cases
    dev_options = (
        a.development_root,
        a.development_hash,
        a.development_audit_root,
        a.development_audit_hash,
    )
    if a.course_bank == "dev":
        if not all(x is not None for x in dev_options):
            raise ValueError("complete separate DEV source proofs required")
        target_root, target_hash, target_audit_root, target_audit_hash = dev_options
        cases = json.loads((target_root / "declaration.json").read_text())["cases"]
        sd, dev_pins = recovery_proof(
            target_root,
            target_hash,
            target_audit_root,
            target_audit_hash,
            cases=cases,
            course_bank="dev",
        )
        sp.update(dev_pins)
    elif any(x is not None for x in dev_options):
        raise ValueError("development proofs cannot masquerade as TRAIN")
    source = json.loads((target_root / "complete.json").read_text())
    for path, h in sp.items():
        if path in pins and pins[path] != h:
            raise ValueError("teacher/source dependency conflict")
    pins.update(sp)
    if len(cases) != 16 or sd["checkpoint_hash"] != train_sd["checkpoint_hash"]:
        raise ValueError("same pinned trained motor and all16 source cases required")
    # The teacher's selected recovery uses exactly this kick model lineage.
    if sd["checkpoint_hash"] not in td["input_file_hashes"].values():
        raise ValueError("recovery teacher was not trained on this source motor")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_bank=a.course_bank,
            cases=cases,
            source_root=str(target_root),
            source_hash=target_hash,
            teacher_root=str(a.teacher_root),
            teacher_hash=a.teacher_hash,
            teacher_audit_root=str(a.teacher_audit_root),
            teacher_audit_hash=a.teacher_audit_hash,
            source_audit_root=str(target_audit_root),
            source_audit_hash=target_audit_hash,
            training_source_root=str(a.source_root),
            training_source_hash=a.source_hash,
            training_source_audit_root=str(a.source_audit_root),
            training_source_audit_hash=a.source_audit_hash,
            common_gain=latent.tolist(),
            arms=["zero", "feedback"],
            independent_world_repeats=2,
            all_cases_retained=True,
            second_is_reference_prefix=True,
            new_recovery_phase_not_old_exam=True,
            actor_checkpoint_hash=sd["checkpoint_hash"],
            actor_training_receipt_hash=sd["training_receipt_hash"],
            body_resets=0,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_balance_recovery import transfer_reference_prefix
        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_recovery_trial import run_recovery_trial
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
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

        def read(row, root=target_root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("confined exact source trajectory required")
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        def same(x, y):
            if set(x) != set(y) or any(not np.array_equal(x[k], y[k]) for k in x):
                raise ValueError("qualified prefix or full independent repeat differs")

        results = {}
        for arm, gain in (("zero", np.zeros((9, 4))), ("feedback", latent)):
            worlds = []
            for i, case in enumerate(cases):
                reference = source["results"]["learned"]["worlds"][i]
                source_artifacts = {Path(x["raw_path"]).stem: x for x in reference["artifacts"]}
                references = {
                    k: read(v)
                    for k, v in source_artifacts.items()
                    if k in ("skill0", "terminal0") or k.startswith("between0-1-")
                }
                if len(reference["skills"]) > 1:
                    references["skill1"] = transfer_reference_prefix(
                        read(reference["skills"][1]["result"]),
                        horizon=bench.clips[case["source"]].length,
                    )
                replicas = []
                for repeat in range(2):
                    folder = a.output / arm / f"case{i}-repeat{repeat}"
                    folder.mkdir(parents=True)
                    artifacts = []

                    def store(key, raw, folder=folder, artifacts=artifacts, references=references):
                        if key in references:
                            same(raw, references[key])
                        path = folder / (key + ".npz")
                        if path.exists():
                            raise ValueError("physical evidence overwrite forbidden")
                        np.savez_compressed(path, **raw)
                        row = dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))
                        artifacts.append(row)
                        return row

                    world = run_recovery_trial(bench, case, actor.base, actor, gain, store=store)
                    world.update(
                        case=i,
                        artifacts=artifacts,
                        quality_by_skill=[
                            bool(contact_rank([s["result"], s["result"]])[1] > 0)
                            for s in world["skills"]
                        ],
                    )
                    write_once(folder / "complete.json", world)
                    replicas.append(world)
                if replicas[0]["status"] != replicas[1]["status"] or len(
                    replicas[0]["artifacts"]
                ) != len(replicas[1]["artifacts"]):
                    raise ValueError("complete world branch repeats differ")
                for x, y in zip(replicas[0]["artifacts"], replicas[1]["artifacts"], strict=True):
                    same(read(x, a.output), read(y, a.output))
                worlds.append(dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"]))
                print(
                    json.dumps(
                        dict(
                            arm=arm,
                            case=i,
                            status=world["status"],
                            quality_by_skill=world["quality_by_skill"],
                        )
                    ),
                    flush=True,
                )
            results[arm] = dict(
                worlds=worlds,
                stable=sum(w["stable"] for w in worlds),
                recovery_healthy=sum(
                    w["recovery"] is not None and w["recovery"]["healthy"] for w in worlds
                ),
                recovery_certificates=sum(
                    w["recovery"] is not None and w["recovery"]["recovery_stability_observed"]
                    for w in worlds
                ),
                quality_by_skill=[
                    sum(len(w["quality_by_skill"]) > k and w["quality_by_skill"][k] for w in worlds)
                    for k in range(3)
                ],
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_BALANCE_FULL16_TRANSFER_CLOSED_PENDING_AUDIT",
                input_file_hashes=pins,
                source_head=a.expected_head,
                course_bank=a.course_bank,
                results=results,
                all_prefix_and_independent_world_arrays_exact=True,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                neural_optimizer_updates=0,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
                fresh_exam=False,
                team_game=False,
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
