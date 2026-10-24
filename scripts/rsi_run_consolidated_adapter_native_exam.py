"""Once-only declared native exam after a successful independent CPU audit.

This is a bounded successor, not an autonomous research decision. The existing
frozen native protocols still own physics, course admission and scoring.
Systemd After/Requires may queue this behind an already declared CPU unit.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

from scripts.rsi_native_context_recovery_common import digest


def native_command(
    *,
    training_root: Path,
    training_hash: str,
    cpu_root: Path,
    expected_head: str,
    output: Path,
    exam: str,
    rehearsal_root: Path | None = None,
    rehearsal_hash: str | None = None,
    native_root: Path | None = None,
    producer_head: str | None = None,
) -> list[str]:
    if exam not in ("pass80", "retention44", "pass80-own", "retention44-own"):
        raise ValueError("one explicitly declared original native protocol required")
    own = exam.endswith("-own")
    if own and (native_root is None or producer_head is None):
        raise ValueError("exact closed physical predecessor and producer head required")
    tp, cp = training_root / "complete.json", cpu_root / "complete.json"
    if (
        output.exists()
        or any(p.is_symlink() for p in (tp, cp))
        or digest(tp) != training_hash
        or any((r / "failed.json").exists() for r in (training_root, cpu_root))
    ):
        raise ValueError("closed exact predecessor and unused once-only physical output required")
    t, c = [json.loads(p.read_text()) for p in (tp, cp)]
    from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import training_data_hash

    kind = t["checkpoint_kind"]
    if (
        kind
        not in (
            "consolidated-adapter-v1",
            "consolidated-refinement-v1",
            "support-masked-refinement-v1",
        )
        or c["status"] != "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS"
        or c["checkpoint_kind"] != t["checkpoint_kind"]
        or c["source_hash"] != training_hash
        or c["source_head"] != (producer_head if own else expected_head)
        or c["checkpoint_hash"] != t["checkpoint_hash"]
        or c["data_hash"] != training_data_hash(t, kind)
        or c["physical_steps_executed"] != 0
        or c["old_retention_proven"] is not False
        or any(
            d[k] is not False
            for d in (t, c)
            for k in ("hardware_authorized", "promotion_authorized", "goal_complete")
        )
    ):
        raise ValueError("exact independently CPU-audited SIM-only new head required")
    from rosclaw_soccer.rsi.consolidated_adapter_checkpoint import expected_cpu_audit_decisions

    if c["own_actor_decisions"] != expected_cpu_audit_decisions(t, kind):
        raise ValueError("incomplete CPU own-history scope cannot launch physics")
    if own:
        if native_root is None:
            raise ValueError("exact physical source required")
        rp, dp = native_root / "complete.json", native_root / "declaration.json"
        if (
            native_root.is_symlink()
            or any(p.is_symlink() for p in (rp, dp))
            or (native_root / "failed.json").exists()
        ):
            raise ValueError("complete ordinary physical source without a failure required")
        report, declared = [json.loads(p.read_text()) for p in (rp, dp)]
        status, count = (
            ("TEAM_PASS_CANDIDATE80_COLLECTED_PENDING_OWN_NATIVE_AND_RETENTION", 80)
            if exam == "pass80-own"
            else ("DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT", 44)
        )
        if (
            report["status"] != status
            or len(report["records"]) != count
            or report["declaration_hash"] != digest(dp)
            or declared["checkpoint_kind"] != kind
            or any(
                v["training_hash"] != training_hash
                or v["source_head"] != producer_head
                or v["hardware_authorized"] is not False
                or v["promotion_authorized"] is not False
                for v in (report, declared)
            )
        ):
            raise ValueError("whole source/declared model and original producer must match")
        args = [
            sys.executable,
            "-u",
            "-m",
            "scripts.rsi_audit_team_pass_motor"
            if exam == "pass80-own"
            else "scripts.rsi_audit_dual_motor_native_exam",
            "--root",
            str(native_root),
            "--source-hash",
            digest(rp),
            "--expected-head",
            expected_head,
            "--output",
            str(output),
        ]
        if exam == "pass80-own":
            args += [
                "--checkpoint-kind",
                kind,
                "--producer-head",
                str(producer_head),
            ]
        return args
    args = [
        sys.executable,
        "-u",
        "-m",
        "scripts.rsi_exam_team_pass_motor"
        if exam == "pass80"
        else "scripts.rsi_exam_trained_dual_motor",
        "--checkpoint-kind",
        kind,
        "--training-root",
        str(training_root),
        "--training-hash",
        training_hash,
        "--cpu-audit-root",
        str(cpu_root),
        "--expected-head",
        expected_head,
        "--output",
        str(output),
    ]
    if exam == "pass80":
        args += ["--cpu-audit-hash", digest(cp)]
    else:
        if rehearsal_root is None or rehearsal_hash is None:
            raise ValueError("original fixed rehearsal witness required for retention exam")
        if digest(rehearsal_root / "complete.json") != rehearsal_hash:
            raise ValueError("original fixed rehearsal hash differs")
        args += [
            "--cpu-auditor-head",
            expected_head,
            "--rehearsal-root",
            str(rehearsal_root),
            "--rehearsal-hash",
            rehearsal_hash,
        ]
    return args


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("training-root", "cpu-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("training-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument(
        "--exam", choices=("pass80", "retention44", "pass80-own", "retention44-own"), required=True
    )
    p.add_argument("--rehearsal-root", type=Path)
    p.add_argument("--rehearsal-hash")
    p.add_argument("--native-root", type=Path)
    p.add_argument("--producer-head")
    a = p.parse_args()
    from scripts.rsi_native_fresh_common import clean

    clean(a.expected_head)
    command = native_command(**vars(a))
    print(
        json.dumps(
            dict(
                phase="STARTING_DECLARED_NATIVE_EXAM",
                exam=a.exam,
                automatic_retry=False,
                scientific_decision_automated=False,
                hardware_authorized=False,
            )
        ),
        flush=True,
    )
    subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
