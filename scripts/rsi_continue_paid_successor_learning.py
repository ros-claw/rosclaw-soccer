"""One finite per-seed SIM pipeline: causal data -> CUDA fit -> both bank exams/audits.

Fixed allowlisted modules only. No retry of incomplete physics, no promotion,
and no claim that this finite scheduler replaces subsequent scientific analysis.
"""

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    p = argparse.ArgumentParser(description=__doc__)
    for name in (
        "data-root",
        "train-root",
        "train-audit-root",
        "dev-root",
        "dev-audit-root",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--data-source-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--gpu-uuid", required=True)
    p.add_argument("--decoupled", action="store_true")
    a = p.parse_args()

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("immutable clean pipeline source required")

    clean()
    if not a.gpu_uuid.startswith("GPU-") or "," in a.gpu_uuid:
        p.error("one explicitly isolated GPU UUID required")
    a.output.mkdir(exist_ok=True)
    with (a.output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        declaration = dict(
            schema="soccer.successor_learning_finite_queue.v1",
            seed=a.seed,
            decoupled=a.decoupled,
            source_head=a.expected_head,
            data_source_head=a.data_source_head,
            data_root=str(a.data_root),
            gpu_uuid=a.gpu_uuid,
            bank_roots={"train": str(a.train_root), "dev": str(a.dev_root)},
            bank_audits={"train": str(a.train_audit_root), "dev": str(a.dev_audit_root)},
            automatic_retry=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        path = a.output / "declaration.json"
        if path.exists():
            if json.loads(path.read_text()) != declaration:
                raise ValueError("existing queue commitment differs")
        else:
            write_once(path, declaration)
        deadline = time.monotonic() + 7200
        while not (a.data_root / "complete.json").exists():
            clean()
            if (a.data_root / "failed.json").exists():
                raise ValueError("curation failed; no implicit training or retry")
            if time.monotonic() > deadline:
                raise TimeoutError("curation wait expired without physics retry")
            time.sleep(10)
        data = json.loads((a.data_root / "complete.json").read_text())
        if (
            data["source_head"] != a.data_source_head
            or data["status"] != "PAID_SUCCESSOR_TRAIN_FEATURES_CAUSALLY_RECOMPUTED"
            or data["curriculum_protocol"] != "successor-six-teacher-v2"
            or (a.data_root / "failed.json").exists()
        ):
            raise ValueError("declared expanded TRAIN-only feature curation required")
        data_hash = digest(a.data_root / "complete.json")
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        stages = {}

        def stage(name, module, argv, *, expected_status, gpu=False):
            root = a.output / name
            clean()
            if digest(a.data_root / "complete.json") != data_hash:
                raise ValueError("committed curation changed")
            if (root / "failed.json").exists():
                raise ValueError("failed stage must be diagnosed, not automatically retried")
            if not (root / "complete.json").exists():
                started = a.output / (name + "-started.json")
                if root.exists() or started.exists():
                    raise ValueError(
                        "incomplete/ambiguous launch preserved; no duplicate allocation"
                    )
                args = [*argv, "--expected-head", a.expected_head, "--output", str(root)]
                write_once(
                    started,
                    dict(
                        module=module,
                        argv=args,
                        source_head=a.expected_head,
                        hardware_authorized=False,
                    ),
                )
                env["CUDA_VISIBLE_DEVICES"] = a.gpu_uuid if gpu else ""
                with (a.output / (name + ".log")).open("xb") as log:
                    try:
                        subprocess.run(
                            [sys.executable, "-u", "-m", "scripts." + module, *args],
                            env=env,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            check=True,
                            timeout=7200,
                        )
                    except (subprocess.CalledProcessError, subprocess.TimeoutExpired) as error:
                        write_once(
                            a.output / (name + "-failed.json"),
                            dict(
                                error_type=type(error).__name__,
                                error=str(error),
                                automatic_retry=False,
                                raw_data_preserved=True,
                                hardware_authorized=False,
                            ),
                        )
                        raise
            clean()
            result = json.loads((root / "complete.json").read_text())
            if (
                result["status"] != expected_status
                or result["source_head"] != a.expected_head
                or result["hardware_authorized"] is not False
                or result["promotion_authorized"] is not False
                or (root / "failed.json").exists()
            ):
                raise ValueError("child exit/marker is not sufficient qualified stage evidence")
            stages[name] = dict(root=str(root), hash=digest(root / "complete.json"))
            return root, result

        student, training = stage(
            "trained",
            "rsi_train_paid_successor_student",
            [
                "--data-root",
                str(a.data_root),
                "--data-hash",
                data_hash,
                "--seed",
                str(a.seed),
                "--staged",
                *(["--decoupled"] if a.decoupled else []),
            ],
            expected_status="PAID_SUCCESSOR_NEURAL_STUDENT_TRAINED_NOT_PHYSICALLY_QUALIFIED",
            gpu=True,
        )
        if training["data_hash"] != data_hash or training["config"]["seed_index"] != a.seed:
            raise ValueError("wrong trained student identity")
        for bank, source, audit in (
            ("train", a.train_root, a.train_audit_root),
            ("dev", a.dev_root, a.dev_audit_root),
        ):
            exam, _ = stage(
                bank + "-exam",
                "rsi_exam_paid_successor_student",
                [
                    "--student-root",
                    str(student),
                    "--student-hash",
                    stages["trained"]["hash"],
                    "--source-root",
                    str(source),
                    "--source-audit-root",
                    str(audit),
                    "--bank",
                    bank,
                ],
                expected_status="PAID_SUCCESSOR_NEURAL_STUDENT_LIFECYCLE_CLOSED_PENDING_AUDIT",
            )
            stage(
                bank + "-audit",
                "rsi_audit_paid_successor_student",
                [
                    "--root",
                    str(exam),
                    "--producer-head",
                    a.expected_head,
                ],
                expected_status="PAID_SUCCESSOR_STUDENT_ALL_NATIVE_PRIMITIVES_AND_OWN_NEURAL_AUDIT_PASS",
            )
        completed = dict(
            status="FINITE_SUCCESSOR_LEARNING_QUEUE_CLOSED_REQUIRES_AGENT_REVIEW",
            stages=stages,
            source_head=a.expected_head,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        final = a.output / "complete.json"
        if final.exists():
            if json.loads(final.read_text()) != completed:
                raise ValueError("closed queue evidence changed")
        else:
            write_once(final, completed)


if __name__ == "__main__":
    main()
