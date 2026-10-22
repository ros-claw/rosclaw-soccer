"""Bounded scientific successor chain, not an idle observer or success retry loop.

One admitted dataset -> actual updates -> independent CPU audit -> unassisted
native44 exam -> independent native44 audit. No threshold relaxation, winner
cherry-pick, automatic retry or promotion. Host active Goal supplies reasoning
after this fixed sequence; this program does not claim to wake an LLM.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("data-root", "warm-start-root", "rehearsal-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in (
        "data-producer-head",
        "warm-start-hash",
        "rehearsal-hash",
        "expected-head",
        "gpu-uuid",
    ):
        p.add_argument("--" + n, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("whole stage must retain exact frozen source")

    clean()
    if a.output.exists() or not a.gpu_uuid.startswith("GPU-"):
        raise ValueError("unused bounded stage output and explicit GPU UUID required")
    dp = a.data_root / "complete.json"
    data = json.loads(dp.read_text())
    if (
        data["source_head"] != a.data_producer_head
        or data["status"] != "DUAL_MOTOR_ACTUAL_DAGGER_SEQUENCES_READY_NOT_TRAINED"
        or data["new_corrections"] != 2
        or len(data["records"]) != 24
        or data["hardware_authorized"] is not False
        or data["promotion_authorized"] is not False
        or (a.data_root / "failed.json").exists()
    ):
        raise ValueError("fixed completed two-correction actual DAgger data required")
    data_hash = digest(dp)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.bounded_actual_dagger_successor_chain.v1",
            source_head=a.expected_head,
            data_hash=data_hash,
            data_producer_head=a.data_producer_head,
            warm_start_hash=a.warm_start_hash,
            rehearsal_hash=a.rehearsal_hash,
            stages=["train256", "own_cpu24", "unassisted_native44", "own_native44"],
            attempts_per_stage=1,
            automatic_retry=False,
            threshold_changes_allowed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    completed = []

    def run(module, output, options, status, *, gpu=False):
        clean()
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES=a.gpu_uuid if gpu else "",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUBLAS_WORKSPACE_CONFIG=":4096:8",
        )
        argv = [
            sys.executable,
            "-m",
            "scripts." + module,
            "--expected-head",
            a.expected_head,
            "--output",
            str(output),
        ] + options
        write_once(
            a.output / (output.name + "-started.json"),
            dict(argv=argv, gpu=gpu, hardware_authorized=False, promotion_authorized=False),
        )
        subprocess.run(argv, env=env, check=True)
        rp = output / "complete.json"
        receipt = json.loads(rp.read_text())
        if (
            receipt["source_head"] != a.expected_head
            or receipt["status"] != status
            or receipt["hardware_authorized"] is not False
            or receipt["promotion_authorized"] is not False
            or (output / "failed.json").exists()
        ):
            raise ValueError("exact terminal stage evidence required")
        h = digest(rp)
        completed.append(dict(stage=output.name, hash=h))
        write_once(a.output / (output.name + "-closed.json"), completed[-1])
        return receipt, h

    try:
        train, cpu, exam, audit = [
            a.output / n for n in ("train", "cpu-audit", "native-exam", "native-audit")
        ]
        _, th = run(
            "rsi_train_dual_motor",
            train,
            [
                "--data-root",
                str(a.data_root),
                "--data-hash",
                data_hash,
                "--warm-start-root",
                str(a.warm_start_root),
                "--warm-start-hash",
                a.warm_start_hash,
                "--seed-index",
                "0",
            ],
            "DUAL_RECURRENT_MOTOR_ACTUALLY_TRAINED_NOT_PHYSICALLY_QUALIFIED",
            gpu=True,
        )
        run(
            "rsi_audit_trained_dual_motor",
            cpu,
            ["--root", str(train), "--source-hash", th],
            "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
        )
        _, eh = run(
            "rsi_exam_trained_dual_motor",
            exam,
            [
                "--training-root",
                str(train),
                "--training-hash",
                th,
                "--cpu-audit-root",
                str(cpu),
                "--cpu-auditor-head",
                a.expected_head,
                "--rehearsal-root",
                str(a.rehearsal_root),
                "--rehearsal-hash",
                a.rehearsal_hash,
            ],
            "DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
        )
        result, _ = run(
            "rsi_audit_dual_motor_native_exam",
            audit,
            ["--root", str(exam), "--source-hash", eh],
            "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS",
        )
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="BOUNDED_DAGGER_STAGE_CLOSED_NOT_GOAL_COMPLETE",
                source_head=a.expected_head,
                completed=completed,
                old_protected_retained=result["old_protected_retained"],
                consumed_run_successes=result["consumed_run_successes"],
                next_agent_action="REASON_FROM_AUDITED_OUTCOME_NOT_BLIND_RETRAIN",
                reasoning_wakeup_provided_by_this_program=False,
                automatic_retry=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed=completed,
                next_agent_action="DIAGNOSE_AND_DECLARE_NEW_STAGE_NOT_RETRY",
                automatic_retry=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
