"""Automatically chain a qualified SIM parent exam to practice and its audit.

This finite, journaled pipeline is not an autonomous scientist. It never retries
failed physics, promotes a policy, or declares the overall soccer goal complete.
"""

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path


def checked_receipt(root: Path, status: str) -> tuple[dict, str]:
    path = root / "complete.json"
    if root.is_symlink() or path.is_symlink() or (root / "failed.json").exists():
        raise ValueError("ordinary completed evidence without a failure required")
    raw = path.read_bytes()
    report = json.loads(raw)
    if report.get("status") != status:
        raise ValueError("stage receipt status does not match its declared contract")
    return report, "sha256:" + hashlib.sha256(raw).hexdigest()


def main():
    from scripts.rsi_atomic_artifacts import write_once

    p = argparse.ArgumentParser(description=__doc__)
    for name in ("exam", "primitive", "neural"):
        p.add_argument("--" + name + "-root", type=Path, required=True)
        p.add_argument("--" + name + "-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--course-bank", choices=("train", "dev"), default="train")
    p.add_argument("--recovery-chain-root", type=Path)
    p.add_argument("--recovery-chain-hash")
    p.add_argument("--recovery-audit-root", type=Path)
    p.add_argument("--recovery-audit-hash")
    p.add_argument("--recovery-cutover", type=int, choices=(140, 160, 190))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    recovery_options = (
        "recovery-chain-root",
        "recovery-chain-hash",
        "recovery-audit-root",
        "recovery-audit-hash",
        "recovery-cutover",
    )
    use_teacher = any(getattr(a, k.replace("-", "_")) is not None for k in recovery_options)
    if use_teacher and (
        a.course_bank != "train"
        or any(getattr(a, k.replace("-", "_")) is None for k in recovery_options)
    ):
        raise ValueError("complete TRAIN-only recovery teacher commitment required")
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused pipeline output and clean pinned source required")
    statuses = (
        "PAID_LIVE_DISTILLED_CPU_FULL_WORLD_EXAM_CLOSED",
        "PAID_LIVE_DISTILLED_ALL_PRIMITIVE_AUDIT_PASS",
        "PAID_GATED_CONTACT_ALL_NEURAL_PROPOSALS_EXACT",
    )
    parents = []
    for name, status in zip(("exam", "primitive", "neural"), statuses, strict=True):
        root, expected = getattr(a, name + "_root"), getattr(a, name + "_hash")
        report, actual = checked_receipt(root, status)
        if actual != expected:
            raise ValueError("parent commitment changed before pipeline allocation")
        parents.append(report)
    exam, primitive, neural = parents
    exam_path, primitive_path = a.exam_root / "complete.json", a.primitive_root / "complete.json"
    if (
        primitive["input_file_hashes"].get(str(exam_path)) != a.exam_hash
        or neural["input_file_hashes"].get(str(exam_path)) != a.exam_hash
        or neural["input_file_hashes"].get(str(primitive_path)) != a.primitive_hash
        or neural["own_neural_rows"] != exam["actual_neural_rows"]
        or len(exam["candidates"]) != 1
        or exam["candidates"][0]["full_continuous_skill_gate"] is not True
    ):
        raise ValueError("exact independently qualified parent dependency chain required")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            course_bank=a.course_bank,
            recovery_teacher={k: str(getattr(a, k.replace("-", "_"))) for k in recovery_options}
            if use_teacher
            else None,
            parent_exam_hash=a.exam_hash,
            parent_primitive_hash=a.primitive_hash,
            parent_neural_hash=a.neural_hash,
            stages=["three_skill_physics", "native_and_neural_audit"],
            automatic_retries=0,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )
    stage = "three_skill_physics"
    try:
        chain, audit = a.output / "chain", a.output / "audit"
        command = [sys.executable, "-u", "-m", "scripts.rsi_exam_paid_practice_chain"]
        for name in ("exam", "primitive", "neural"):
            command += [
                "--" + name + "-root",
                str(getattr(a, name + "_root")),
                "--" + name + "-hash",
                getattr(a, name + "_hash"),
            ]
        if use_teacher:
            for key in recovery_options:
                command += ["--" + key, str(getattr(a, key.replace("-", "_")))]
        subprocess.run(
            command
            + [
                "--skills",
                "3",
                "--course-bank",
                a.course_bank,
                "--expected-head",
                a.expected_head,
                "--output",
                str(chain),
            ],
            check=True,
        )
        _, chain_hash = checked_receipt(
            chain, "PAID_CONTINUING_CHAIN_CPU_EXAM_CLOSED_PENDING_INDEPENDENT_AUDIT"
        )
        write_once(a.output / "physics_closed.json", dict(chain_hash=chain_hash))
        stage = "native_and_neural_audit"
        subprocess.run(
            [
                sys.executable,
                "-u",
                "-m",
                "scripts.rsi_audit_paid_practice_chain",
                "--root",
                str(chain),
                "--hash",
                chain_hash,
                "--parent-root",
                str(a.exam_root),
                "--parent-hash",
                a.exam_hash,
                "--expected-head",
                a.expected_head,
                "--output",
                str(audit),
            ],
            check=True,
        )
        _, audit_hash = checked_receipt(audit, "PAID_CONTINUING_CHAIN_NATIVE_AND_NEURAL_AUDIT_PASS")
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_CONTACT_CONTINUATION_PIPELINE_AUDITED",
                chain_hash=chain_hash,
                audit_hash=audit_hash,
                goal_complete=False,
                hardware_authorized=False,
                next_agent_action="ANALYZE_NEW_SUCCESSOR_STATES_AND_IMPLEMENT_NEXT_SCIENTIFIC_STAGE",
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                stage=stage,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
