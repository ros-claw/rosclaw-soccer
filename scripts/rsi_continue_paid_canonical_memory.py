"""Finite fit/exam/audit/diagnosis chain; not an autonomous scientific successor.

Close every declared lane even when another lane fails. Never retry a charged
attempt or treat a timer/heartbeat as evidence of policy improvement.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data-root", "train20-plan", "train16-plan", "dev16-plan", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("data-source-head", "train20-hash", "train16-hash", "dev16-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen canonical pipeline source required")

    clean()
    if a.output.exists() or os.environ.get("CUDA_VISIBLE_DEVICES") != "":
        raise ValueError("unused output and explicit CPU-only pipeline required")
    if any(
        os.environ.get(k) != "1"
        for k in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")
    ):
        raise ValueError("single-thread numerical children required")
    plans = {
        name: (getattr(a, name + "_plan"), getattr(a, name + "_hash"))
        for name in ("train20", "train16", "dev16")
    }
    for root, h in plans.values():
        if (root / "complete.json").is_symlink() or digest(root / "complete.json") != h:
            raise ValueError("exact predeclared physical banks required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            schema="paid.canonical_memory.finite_queue.v1",
            source_head=a.expected_head,
            data_source_head=a.data_source_head,
            data_root=str(a.data_root),
            plans={n: dict(root=str(root), hash=h) for n, (root, h) in plans.items()},
            learned_and_zero_coefficient_control=True,
            parallel_exam_lanes=4,
            automatic_retry=False,
            scientific_successor_automated=False,
            agent_wakeup_provided_by_this_script=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )

    def stage(name, module, args, status):
        clean()
        root = a.output / name
        if root.exists():
            raise ValueError("charged stage cannot be retried")
        write_once(a.output / (name + "-started.json"), dict(module=module, output=str(root)))
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts." + module,
                *args,
                "--expected-head",
                a.expected_head,
                "--output",
                str(root),
            ],
            check=True,
        )
        path = root / "complete.json"
        receipt = json.loads(path.read_text())
        if (
            path.is_symlink()
            or (root / "failed.json").exists()
            or receipt.get("status") != status
            or any(
                receipt.get(k) is not False for k in ("hardware_authorized", "promotion_authorized")
            )
        ):
            raise ValueError("explicit closed SIM-only stage receipt required")
        result = dict(root=str(root), hash=digest(path), status=status)
        write_once(a.output / (name + "-closed.json"), result)
        return result

    try:
        deadline = time.monotonic() + 7200
        while not (a.data_root / "complete.json").exists():
            clean()
            if (a.data_root / "failed.json").exists():
                raise ValueError("curation failed; no automatic retry")
            if time.monotonic() >= deadline:
                raise TimeoutError("bounded curation wait; no physics retry")
            time.sleep(5)
        rp = a.data_root / "complete.json"
        data = json.loads(rp.read_text())
        if (
            rp.is_symlink()
            or data.get("source_head") != a.data_source_head
            or data.get("status") != "PAID_CANONICAL_SUCCESSFUL_BEHAVIOR_TRAIN_CAUSALLY_RECOMPUTED"
        ):
            raise ValueError("exact canonical data producer required")
        dh = digest(rp)
        fits = {}
        for name in ("learned", "zero-control"):
            fits[name] = stage(
                name + "-trained",
                "rsi_fit_paid_cardinal_memory",
                [
                    "--data-root",
                    str(a.data_root),
                    "--data-hash",
                    dh,
                    "--canonical-behavior",
                    *(["--zero-control"] if name == "zero-control" else []),
                ],
                "PAID_CARDINAL_MEMORY_ZERO_COEFFICIENT_CONTROL_NOT_PHYSICALLY_QUALIFIED"
                if name == "zero-control"
                else "PAID_CARDINAL_MOTOR_MEMORY_FIT_NOT_PHYSICALLY_QUALIFIED",
            )

        def lane(model, bank):
            name = model + "-" + bank
            root, h = plans[bank]
            original = bank != "train20"
            exam = stage(
                name + "-exam",
                "rsi_exam_paid_ancestor_router",
                [
                    "--plan-root",
                    str(root),
                    "--plan-hash",
                    h,
                    "--student-root",
                    fits[model]["root"],
                    "--first-enabled",
                ],
                "PAID_FULL_ENTRY_ORIGINAL16_WORLDS_CLOSED_PENDING_AUDIT"
                if original
                else "PAID_ROUTER_TRAIN20_WORLDS_CLOSED_PENDING_AUDIT",
            )
            audit = stage(
                name + "-audit",
                "rsi_audit_paid_ancestor_router",
                ["--root", exam["root"], "--producer-head", a.expected_head, "--first-enabled"],
                "PAID_FULL_ENTRY_ORIGINAL16_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS"
                if original
                else "PAID_ROUTER_TRAIN20_OWN_NN_AND_ALL_PHASE_PHYSICS_AUDIT_PASS",
            )
            result = dict(exam=exam, audit=audit)
            if model == "learned" and bank == "train20":
                result["diagnosis"] = stage(
                    "learned-own-diagnosis",
                    "rsi_diagnose_paid_compact_policy",
                    [
                        "--cardinal-memory",
                        "--exam-root",
                        exam["root"],
                        "--audit-root",
                        audit["root"],
                        "--data-root",
                        str(a.data_root),
                        "--data-hash",
                        dh,
                        "--producer-head",
                        a.expected_head,
                    ],
                    "PAID_TRAIN_COMPACT_OWN_POLICY_SUPPORT_AND_CONTACT_DIAGNOSIS",
                )
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            names = [("learned", b) for b in plans] + [("zero-control", "train20")]
            jobs = [(m + "-" + b, pool.submit(lane, m, b)) for m, b in names]
            results = {name: job.result() for name, job in jobs}
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CANONICAL_MEMORY_ALL_DECLARED_LANES_CLOSED_NOT_PROMOTED",
                source_head=a.expected_head,
                data_hash=dh,
                fits=fits,
                lanes=results,
                scientific_successor_automated=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                preserve_charged_stages=True,
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
