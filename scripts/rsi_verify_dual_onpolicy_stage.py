"""Automatic CPU -> unassisted native44 -> own native44 examination of PPO motor.

This does not retry training or promote a candidate. The exact producer head
is bound before a queued training job closes; receipt hashes bind each next step.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("training-root", "rehearsal-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("producer-head", "rehearsal-hash", "expected-head"):
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
            raise ValueError("exact clean frozen on-policy exam chain source required")

    clean()
    tp = a.training_root / "complete.json"
    t = json.loads(tp.read_text())
    if (
        a.output.exists()
        or tp.is_symlink()
        or (a.training_root / "failed.json").exists()
        or t["source_head"] != a.producer_head
        or t["status"] != "DUAL_AR1_ACTOR_CRITIC_ACTUALLY_UPDATED_PENDING_PHYSICAL_EXAM"
        or t["online_RL_claim"] is not True
        or not 1 <= t["accepted_optimizer_steps"] <= 4
        or t["hardware_authorized"] is not False
        or t["promotion_authorized"] is not False
    ):
        raise ValueError("exact closed bounded actor-critic training source required")
    a.output.mkdir()
    th = digest(tp)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="dual256.actual_onpolicy_exam_chain.v1",
            source_head=a.expected_head,
            training_hash=th,
            producer_head=a.producer_head,
            checkpoint_kind="ar1-actor-critic",
            stages=["cpu-audit", "native-exam", "native-audit"],
            attempts_per_stage=1,
            automatic_retry=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    completed = []

    def run(module, name, options, status):
        clean()
        output = a.output / name
        argv = [
            sys.executable,
            "-m",
            "scripts." + module,
            "--expected-head",
            a.expected_head,
            "--output",
            str(output),
        ] + options
        write_once(a.output / (name + "-started.json"), dict(argv=argv))
        env = dict(
            os.environ,
            CUDA_VISIBLE_DEVICES="",
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
        )
        subprocess.run(argv, env=env, check=True)
        receipt = output / "complete.json"
        r = json.loads(receipt.read_text())
        if (
            r["source_head"] != a.expected_head
            or r["status"] != status
            or r["hardware_authorized"] is not False
            or r["promotion_authorized"] is not False
        ):
            raise ValueError("exact independently closed stage evidence required")
        h = digest(receipt)
        completed.append(dict(stage=name, hash=h))
        write_once(a.output / (name + "-closed.json"), completed[-1])
        return r, h

    try:
        run(
            "rsi_audit_trained_dual_motor",
            "cpu-audit",
            [
                "--root",
                str(a.training_root),
                "--source-hash",
                th,
                "--checkpoint-kind",
                "ar1-actor-critic",
            ],
            "DUAL_RECURRENT_TRAINED_CHECKPOINT_CPU_OWN_HISTORY_AUDIT_PASS_NOT_PHYSICS",
        )
        _, eh = run(
            "rsi_exam_trained_dual_motor",
            "native-exam",
            [
                "--training-root",
                str(a.training_root),
                "--training-hash",
                th,
                "--cpu-audit-root",
                str(a.output / "cpu-audit"),
                "--cpu-auditor-head",
                a.expected_head,
                "--rehearsal-root",
                str(a.rehearsal_root),
                "--rehearsal-hash",
                a.rehearsal_hash,
                "--checkpoint-kind",
                "ar1-actor-critic",
            ],
            "DUAL_RECURRENT_NATIVE_TRAIN_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT",
        )
        result, _ = run(
            "rsi_audit_dual_motor_native_exam",
            "native-audit",
            ["--root", str(a.output / "native-exam"), "--source-hash", eh],
            "DUAL_RECURRENT_ALL44_WHOLE_NATIVE_OWN_HISTORY_AUDIT_PASS",
        )
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="AR1_ACTOR_CRITIC_EXAM_CHAIN_CLOSED_NOT_GOAL_COMPLETE",
                source_head=a.expected_head,
                completed=completed,
                training_hash=th,
                old_protected_retained=result["old_protected_retained"],
                old_success_cases=result["old_success_cases"],
                consumed_run_successes=result["consumed_run_successes"],
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
                automatic_retry=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
