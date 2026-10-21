"""Frozen seed0 local teacher transfer on all TRAIN16 and DEV16 successor entries.

Only two trained source identities receive their common Cartesian intent; the
other six retain the original actor. No DEV optimization or winner selection.
These are explicitly restored branches, not additional continuous team games.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "teacher-root",
        "teacher-audit-root",
        "train-root",
        "train-audit-root",
        "dev-root",
        "dev-audit-root",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--wait-for-audit", action="store_true")
    a = p.parse_args()
    if (
        a.output.exists()
        or digest(a.teacher_root / "complete.json") != a.teacher_hash
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused transfer output and frozen exact teacher/source required")
    deadline = time.monotonic() + 7200
    while not (a.teacher_audit_root / "complete.json").exists():
        if (a.teacher_audit_root / "failed.json").exists():
            raise ValueError("teacher audit failed; no transfer/automatic retry")
        if not a.wait_for_audit or time.monotonic() > deadline:
            raise TimeoutError("teacher audit wait expired; no physics retry")
        time.sleep(10)
    tr, td, ta = [
        json.loads((root / name).read_text())
        for root, name in (
            (a.teacher_root, "complete.json"),
            (a.teacher_root, "declaration.json"),
            (a.teacher_audit_root, "complete.json"),
        )
    ]
    if (
        tr["status"] != "PAID_SUCCESSOR_CONTACT_TEACHER_CLOSED_PENDING_AUDIT"
        or ta["status"] != "PAID_SUCCESSOR_CONTACT_ALL_CANDIDATES_NATIVE_NEURAL_IK_CEM_AUDIT_PASS"
        or ta["input_file_hashes"].get(str(a.teacher_root / "complete.json")) != a.teacher_hash
        or td["random_seed"] != 20261022
        or td["pairs"] != [[2, 3], [10, 11]]
        or tr["declaration_hash"] != digest(a.teacher_root / "declaration.json")
        or ta["counts"]["own_neural"] != tr["actual_neural_rows"]
        or ta["own_native_steps"] != tr["actual_native_steps"]
        or any(s["selected_rank"][:5] != [1, 1, 1, 2, 2] for s in tr["searches"])
        or any((root / "failed.json").exists() for root in (a.teacher_root, a.teacher_audit_root))
    ):
        raise ValueError("predeclared seed0 audited 4/4 clean stable teacher required")
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate

    banks = {"train": (a.train_root, a.train_audit_root), "dev": (a.dev_root, a.dev_audit_root)}
    gates = {
        bank: standing_gate(root, audit, variant=1, bank=bank)
        for bank, (root, audit) in banks.items()
    }
    pins = dict(ta["input_file_hashes"])
    pins[str(a.teacher_audit_root / "complete.json")] = digest(
        a.teacher_audit_root / "complete.json"
    )
    bank_data = {}
    for bank, (root, audit) in banks.items():
        r, d, proof = [
            json.loads((folder / name).read_text())
            for folder, name in (
                (root, "complete.json"),
                (root, "declaration.json"),
                (audit, "complete.json"),
            )
        ]
        for path, h in proof["input_file_hashes"].items():
            if path in pins and pins[path] != h:
                raise ValueError("source pin conflict")
            pins[path] = h
        pins[str(audit / "complete.json")] = digest(audit / "complete.json")
        bank_data[bank] = (r, d)
    if bank_data["train"][1]["cases"] != td["source_cases"]:
        raise ValueError("teacher must derive from exact declared TRAIN bank")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            baseline_gates=gates,
            teacher_root=str(a.teacher_root),
            teacher_hash=a.teacher_hash,
            teacher_audit_root=str(a.teacher_audit_root),
            teacher_seed=0,
            bank_roots={bank: str(roots[0]) for bank, roots in banks.items()},
            bank_audit_roots={bank: str(roots[1]) for bank, roots in banks.items()},
            all16_per_bank=True,
            repeats=2,
            no_DEV_search=True,
            explicit_restored_successor_branches=True,
            uninterrupted_game=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    bench = None
    try:
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.paid_contact_residual_actor import load_contact_residual_actor
        from rosclaw_soccer.rsi.paid_continuity_audit import identical, snapshot_fields
        from rosclaw_soccer.rsi.paid_foot_taskspace import FootTaskSpaceIntent
        from rosclaw_soccer.rsi.paid_recovery_proof import unique_proof_path
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench
        from rosclaw_soccer.rsi.paid_successor_trial import run_successor_trial
        from scripts.rsi_search_paid_foot_contact import contact_rank

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        cases = [case for _, d in bank_data.values() for case in d["cases"]]
        bench, policy, scene = open_paid_bench(pins, cases)
        sd = bank_data["train"][1]
        if any(
            d["actor_checkpoint_hash"] != sd["actor_checkpoint_hash"] for _, d in bank_data.values()
        ):
            raise ValueError("same frozen underlying R2 motor required")
        actor, _ = load_contact_residual_actor(
            policy,
            checkpoint=unique_proof_path(pins, sd["actor_checkpoint_hash"]),
            checkpoint_hash=sd["actor_checkpoint_hash"],
            training_receipt=unique_proof_path(pins, sd["actor_training_receipt_hash"]),
            training_receipt_hash=sd["actor_training_receipt_hash"],
            scene_hash=scene,
        )
        intents = {
            td["source_cases"][s["pair"][0]]["source"]: (s["center"], np.array(s["latent"]))
            for s in tr["searches"]
        }

        def read(row, root):
            path = root / row["raw_path"]
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(root.resolve())
                or digest(path) != row["raw_hash"]
            ):
                raise ValueError("exact confined branch evidence required")
            pins[str(path)] = row["raw_hash"]
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        results = {}
        for bank, (source, d) in bank_data.items():
            root = banks[bank][0]
            records = []
            for i, case in enumerate(d["cases"]):
                world = source["worlds"][i]
                boundary = read(world["boundaries"][1]["boundary"], root)
                original = {
                    "kick": read(world["kicks"][2]["result"], root),
                    "kick-terminal": read(world["kicks"][2]["terminal"], root),
                    "stand": read(world["stands"][2]["result"], root),
                }
                replicas, all_raw = [], []
                for repeat in range(2):
                    folder = a.output / bank / f"case{i}-repeat{repeat}"
                    folder.mkdir(parents=True)
                    raw_parts = {}

                    def store(
                        key, raw, case=case, original=original, raw_parts=raw_parts, folder=folder
                    ):
                        if case["source"] not in intents:
                            identical(raw, original[key])
                        raw_parts[key] = raw
                        path = folder / (key + ".npz")
                        if path.exists():
                            raise ValueError("phase trajectory overwrite forbidden")
                        np.savez_compressed(path, **raw)
                        return dict(raw_path=str(path.relative_to(a.output)), raw_hash=digest(path))

                    transform = None
                    if case["source"] in intents:
                        center, latent = intents[case["source"]]
                        transform = FootTaskSpaceIntent(
                            bench.template.model,
                            foot="left" if "left" in case["source"] else "right",
                            center=center,
                            latent=latent,
                        )
                    row = run_successor_trial(
                        bench,
                        case,
                        actor.base,
                        entry=snapshot_fields(boundary, "after_"),
                        previous=boundary["previous_actual_action"],
                        transform=transform,
                        store=store,
                    )
                    row.update(
                        case=i,
                        bank=bank,
                        teacher_applied=transform is not None,
                        quality=bool(contact_rank([row["kick"], row["kick"]])[1] > 0),
                    )
                    write_once(folder / "complete.json", row)
                    replicas.append(row)
                    all_raw.append(raw_parts)
                if set(all_raw[0]) != set(all_raw[1]):
                    raise ValueError("whole branch phase set differs across repeats")
                for key in all_raw[0]:
                    identical(all_raw[0][key], all_raw[1][key])
                records.append(dict(replicas[0], repeat=replicas[1]))
                print(
                    json.dumps(
                        dict(
                            bank=bank,
                            case=i,
                            quality=row["quality"],
                            healthy=bool(row["stand"] and row["stand"]["healthy"]),
                        )
                    ),
                    flush=True,
                )
            results[bank] = dict(
                rows=records,
                quality=sum(x["quality"] for x in records),
                body_healthy=sum(bool(x["stand"] and x["stand"]["healthy"]) for x in records),
                stand_stable=sum(
                    bool(x["stand"] and x["stand"]["stability_observed"]) for x in records
                ),
                baseline_quality=source["summary"]["quality_by_kick"][2],
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_FROZEN_SUCCESSOR_TEACHER_TRAIN_DEV_TRANSFER_PENDING_AUDIT",
                input_file_hashes=pins,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                results=results,
                actual_native_steps=bench.native_steps,
                actual_neural_rows=bench.neural_rows,
                all_branch_repeat_arrays_exact=True,
                optimizer_updates=0,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
                uninterrupted_game=False,
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
