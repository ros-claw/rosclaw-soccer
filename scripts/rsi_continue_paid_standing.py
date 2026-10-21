"""Finite SIM queue: audited TRAIN -> all16 DEV physics -> own-neural DEV audit.

Not an autonomous coding agent or indefinite scientific successor generator.
No arbitrary shell commands, hidden retries, promotion, or hardware actions.
An interrupted incomplete stage is retained and requires agent diagnosis.
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
    from rosclaw_soccer.rsi.paid_standing_continuation import standing_gate
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--train", nargs=2, action="append", required=True, metavar=("EXAM_ROOT", "AUDIT_ROOT")
    )
    for name in (
        "physics-source",
        "audit-source",
        "dev-source-root",
        "dev-source-audit-root",
        "output",
    ):
        p.add_argument("--" + name, type=Path, required=True)
    p.add_argument("--dev-source-hash", required=True)
    p.add_argument("--dev-source-audit-hash", required=True)
    a = p.parse_args()
    if len(a.train) != 3:
        p.error("exact fixed-target, R2-standing, R6-successor TRAIN prerequisites required")
    sources = [a.physics_source, a.audit_source, Path.cwd()]
    heads = {
        str(root): subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True
        ).strip()
        for root in sources
    }

    def clean_sources():
        for root in sources:
            if subprocess.check_output(["git", "status", "--porcelain"], cwd=root).strip() or (
                subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
                != heads[str(root)]
            ):
                raise ValueError("source changed; do not launch/retry a stage")

    clean_sources()
    pins = {
        str(a.dev_source_root / "complete.json"): a.dev_source_hash,
        str(a.dev_source_audit_root / "complete.json"): a.dev_source_audit_hash,
    }
    check_pins(pins)
    a.output.mkdir(exist_ok=True)
    with (a.output / "queue.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        plan = dict(
            schema="soccer.rsi.paid_standing_continuation.v1",
            sources=heads,
            train=a.train,
            dev_source_root=str(a.dev_source_root),
            dev_source_audit_root=str(a.dev_source_audit_root),
            input_file_hashes=pins,
            variants=[0, 1, 2],
            courses_per_variant=16,
            whole_world_repeats=2,
            allow_automatic_retry=False,
            agent_reasoning_automated=False,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        declaration = a.output / "declaration.json"
        if declaration.exists():
            if json.loads(declaration.read_text()) != plan:
                raise ValueError("queue commitment differs")
        else:
            write_once(declaration, plan)
        deadline = time.monotonic() + 7200
        while True:
            clean_sources()
            for exam, audit in a.train:
                if any((Path(root) / "failed.json").exists() for root in (exam, audit)):
                    raise ValueError("TRAIN failed; retain data and request agent diagnosis")
            if all((Path(audit) / "complete.json").exists() for _, audit in a.train):
                break
            if time.monotonic() > deadline:
                raise TimeoutError("TRAIN wait expired; no experiment retry")
            time.sleep(10)
        gates = [
            standing_gate(Path(e), Path(v), variant=i, bank="train")
            for i, (e, v) in enumerate(a.train)
        ]
        if any(g["summary"]["stable"] != 16 for g in gates[1:]):
            raise ValueError("neural standing TRAIN stability inadequate; diagnose before DEV")
        admission = a.output / "train-admitted.json"
        if admission.exists():
            if json.loads(admission.read_text()) != {"gates": gates}:
                raise ValueError("TRAIN gate evidence changed")
        else:
            write_once(admission, dict(gates=gates))
        env = dict(
            os.environ,
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )

        def stage(name, source, module, root, argv):
            clean_sources()
            check_pins(pins)
            if (root / "failed.json").exists():
                raise ValueError("failed stage requires new diagnosed plan, not automatic retry")
            if (root / "complete.json").exists():
                return  # Accepted only by the explicit gate below, never by existence alone.
            if root.exists():
                raise ValueError("incomplete stage preserved; no duplicate physics allocation")
            start = a.output / (name + "-started.json")
            if start.exists():
                raise ValueError("ambiguous interrupted launch requires agent diagnosis")
            write_once(
                start,
                dict(
                    module=module,
                    source=str(source),
                    argv=argv,
                    source_head=heads[str(source)],
                    hardware_authorized=False,
                ),
            )
            env["PYTHONPATH"] = (
                str(source / "src") + ":" + str(source) + ":" + os.environ.get("PYTHONPATH", "")
            )
            with (a.output / (name + ".log")).open("xb") as log:
                try:
                    subprocess.run(
                        [sys.executable, "-u", "-m", module, *argv],
                        cwd=source,
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
            clean_sources()
            if not (root / "complete.json").is_file() or (root / "failed.json").exists():
                raise ValueError("child exit zero is not a closed evidence result")

        for variant in range(3):
            exam, audit = (a.output / f"dev-v{variant}-{kind}" for kind in ("exam", "audit"))
            stage(
                f"dev-v{variant}-physics",
                a.physics_source,
                "scripts.rsi_exam_paid_standing_lifecycle",
                exam,
                [
                    "--source-root",
                    str(a.dev_source_root),
                    "--source-hash",
                    a.dev_source_hash,
                    "--source-audit-root",
                    str(a.dev_source_audit_root),
                    "--source-audit-hash",
                    a.dev_source_audit_hash,
                    "--variant",
                    str(variant),
                    "--skills",
                    "3",
                    "--course-bank",
                    "dev",
                    "--expected-head",
                    heads[str(a.physics_source)],
                    "--output",
                    str(exam),
                ],
            )
            stage(
                f"dev-v{variant}-audit",
                a.audit_source,
                "scripts.rsi_audit_paid_standing_lifecycle",
                audit,
                [
                    "--root",
                    str(exam),
                    "--hash",
                    digest(exam / "complete.json"),
                    "--expected-head",
                    heads[str(a.audit_source)],
                    "--output",
                    str(audit),
                ],
            )
            result = standing_gate(exam, audit, variant=variant, bank="dev")
            path = a.output / f"dev-v{variant}-accepted.json"
            if path.exists():
                if json.loads(path.read_text()) != result:
                    raise ValueError("DEV gate evidence changed")
            else:
                write_once(path, result)
        completed = dict(
            status="FINITE_STANDING_DEV_QUEUE_CLOSED_REQUIRES_AGENT_ANALYSIS",
            goal_complete=False,
            team_game=False,
            promotion_authorized=False,
            hardware_authorized=False,
            next="analyse all DEV outcomes before next learning plan",
        )
        path = a.output / "complete.json"
        if path.exists():
            if json.loads(path.read_text()) != completed:
                raise ValueError("closed queue record changed")
        else:
            write_once(path, completed)


if __name__ == "__main__":
    main()
