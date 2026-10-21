"""Actual continuous three-kick/standing worlds for an admitted neural student."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("student-root", "source-root", "source-audit-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--student-hash", required=True)
    p.add_argument("--bank", choices=("train", "dev"), required=True)
    p.add_argument("--expected-head", required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or digest(a.student_root / "complete.json") != a.student_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and clean exact student/runner required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    standing_gate(a.source_root, a.source_audit_root, variant=1, bank=a.bank)
    source, sd, review, training = [
        json.loads((root / name).read_text())
        for root, name in (
            (a.source_root, "complete.json"),
            (a.source_root, "declaration.json"),
            (a.source_audit_root, "complete.json"),
            (a.student_root, "complete.json"),
        )
    ]
    pins = dict(training["input_file_hashes"])
    for path, h in review["input_file_hashes"].items():
        if path in pins and pins[path] != h:
            raise ValueError("source proof lineage conflicts")
        pins[path] = h
    rp = a.student_root / "complete.json"
    cp = Path(training["checkpoint_path"])
    pins.update(
        {
            str(rp): a.student_hash,
            str(cp): training["checkpoint_hash"],
            str(a.source_audit_root / "complete.json"): digest(
                a.source_audit_root / "complete.json"
            ),
        }
    )
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            cases=sd["cases"],
            bank=a.bank,
            student_hash=a.student_hash,
            student_root=str(a.student_root),
            source_root=str(a.source_root),
            source_audit_root=str(a.source_audit_root),
            all16=True,
            repeats=2,
            skills=3,
            standing_controls=100,
            first_motor="unchanged_R2",
            successor_motor="own_neural_student_no_IK_no_case_id",
            standing_motor="frozen_R2_with_own_base_memory",
            body_resets=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_continuity_audit import identical
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_standing_lifecycle import run_standing_lifecycle
        from rosclaw_soccer.rsi.paid_successor_student_actor import load_successor_student
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        bench, policy, scene = open_paid_bench(pins, sd["cases"])
        actor, _ = load_successor_student(
            policy,
            checkpoint=cp,
            checkpoint_hash=training["checkpoint_hash"],
            training_receipt=rp,
            training_receipt_hash=a.student_hash,
            scene_hash=scene,
        )

        def read(row, root=a.source_root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined physical reference required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        worlds = []
        for i, case in enumerate(sd["cases"]):
            first = read(source["worlds"][i]["kicks"][0]["result"])
            replicas, raw_replicas = [], []
            for repeat in range(2):
                folder = a.output / f"case{i}-repeat{repeat}"
                folder.mkdir()
                artifacts, raws = [], {}

                def store(key, raw, folder=folder, artifacts=artifacts, raws=raws, first=first):
                    if key == "kick0":
                        identical(raw, first)
                    path = folder / (key + ".npz")
                    if path.exists():
                        raise ValueError("trajectory overwrite forbidden")
                    np.savez_compressed(path, **raw)
                    raws[key] = raw
                    link = dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))
                    artifacts.append(link)
                    return link

                world = run_standing_lifecycle(bench, case, actor, variant=2, skills=3, store=store)
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
                raw_replicas.append(raws)
            if set(raw_replicas[0]) != set(raw_replicas[1]):
                raise ValueError("whole-world phase set differs across repeats")
            for key in raw_replicas[0]:
                identical(raw_replicas[0][key], raw_replicas[1][key])
            worlds.append(dict(replicas[0], repeat_artifacts=replicas[1]["artifacts"]))
            print(
                json.dumps(dict(case=i, status=world["status"], quality=world["quality_by_kick"])),
                flush=True,
            )
        summary = dict(
            stable=sum(w["stable"] for w in worlds),
            quality_by_kick=[
                sum(len(w["quality_by_kick"]) > k and w["quality_by_kick"][k] for w in worlds)
                for k in range(3)
            ],
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT",
                input_file_hashes=pins,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                summary=summary,
                worlds=worlds,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                all_first_and_full_repeat_arrays_exact=True,
                bank=a.bank,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                actual_native_steps=bench.native_steps if bench is not None else 0,
                actual_neural_rows=bench.neural_rows if bench is not None else 0,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
