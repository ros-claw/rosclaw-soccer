"""Await one owned SIM collection, audit ALL208, learn, test a fixed failure.

No sampling retries, fresh exam, promotion, hardware, or changed reward. The
next independent optimizer starts only after the complete sealed collection.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_continue_memory_validation import run_stage
from scripts.rsi_train_protected_online_motor_v308 import _head


def checked_collection(summary: dict[str, Any], declared: dict[str, Any]) -> None:
    if (
        summary.get("schema") != "soccer.rsi.smooth_memory_failure_exploration.v1"
        or summary.get("commitment") != declared
        or summary.get("independent_contexts") != 13
        or summary.get("exploration_executions") != 208
        or summary.get("physical_executions") != 234
        or declared.get("samples_per_course") != 16
        or declared.get("exploration_stream") != 2
        or declared.get("exploration_stream_namespace") != "STREAM_STRIDE_20000000"
        or declared.get("partition") != "TRAIN_CONSUMED"
        or declared.get("sampling_construction") != "EXACT_CACHED_COMPLETE_MEAN_V1"
        or len(declared.get("courses", [])) != 13
        or len(declared.get("sampling_view_hashes", [])) != 208
        or len(summary.get("rows", [])) != 13
        or len({tuple(v) for v in declared.get("courses", [])}) != 13
        or len(set(declared.get("sampling_view_hashes", []))) != 208
        or any(
            type(summary.get(k)) is not int
            for k in ("independent_contexts", "exploration_executions", "physical_executions")
        )
        or any(
            obj.get(k) is not False
            for obj in (summary, declared)
            for k in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("complete sealed independent 13x16 collection required")
    for i, row in enumerate(summary["rows"]):
        samples = row.get("samples", [])
        if (
            row.get("index") != i
            or [row.get("seed"), row.get("lane")] != declared["courses"][i]
            or len(samples) != 16
            or any(
                sample.get("sample") != s
                or sample.get("view_hash") != declared["sampling_view_hashes"][i * 16 + s]
                for s, sample in enumerate(samples)
            )
        ):
            raise ValueError("all ordered success AND failure trajectories required")


def process_start(pid: int) -> str | None:
    try:
        # /proc field 22; comm itself may contain spaces or parentheses.
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[19]
    except FileNotFoundError:
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "source-root",
        "core-root",
        "exploration-root",
        "output-root",
        "behavior-model",
        "parent-bank-root",
        "consolidated-memory",
        "consolidation-manifest",
        "reference-root",
        "reference-source",
        "reference-core",
        "qualified-parent-model",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--collector-pid", type=int, required=True)
    parser.add_argument("--collector-start", required=True)
    args = parser.parse_args()
    if not 1 <= args.collector_pid < 2**31:
        parser.error("one explicit owned collector required")
    declared = load_json_artifact(args.exploration_root / "commitment.json")
    files = [
        args.exploration_root / "commitment.json",
        args.behavior_model,
        args.consolidated_memory,
        args.consolidation_manifest,
        args.qualified_parent_model,
        args.g1_usd,
        args.late_swing_policy,
        args.parent_bank_root / "validation_summary.json",
        args.parent_bank_root / "independent_review.json",
        Path(__file__),
        Path(__file__).with_name("rsi_continue_memory_validation.py"),
    ]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in files}
    source_head, core_head = _head(args.source_root), _head(args.core_root)

    def stable() -> None:
        for root, head in ((args.source_root, source_head), (args.core_root, core_head)):
            if (
                _head(root) != head
                or subprocess.check_output(
                    ["git", "status", "--porcelain"], cwd=root, text=True
                ).strip()
            ):
                raise ValueError("immutable clean numerical/audit sources required")
        if any(hash_bytes(Path(p).read_bytes()) != h for p, h in pins.items()):
            raise ValueError("continuation input changed; stop without implicit retry")

    stable()
    if declared.get("source_commit") != source_head or declared.get("core_commit") != core_head:
        raise ValueError("continue the actual frozen collection source, not another checkout")
    start = process_start(args.collector_pid)
    if start is not None and (
        start != args.collector_start
        or Path(f"/proc/{args.collector_pid}/cwd").resolve() != args.source_root.resolve()
        or str(args.exploration_root).encode()
        not in Path(f"/proc/{args.collector_pid}/cmdline").read_bytes().split(b"\0")
    ):
        raise ValueError("owned collector identity changed; never follow a reused PID")
    args.output_root.mkdir(exist_ok=False)
    declaration = dict(
        schema="soccer.rsi.independent_motor_learning_continuation.v1",
        input_hashes=pins,
        collection_commitment_hash=hash_json(declared),
        source_commit=source_head,
        core_commit=core_head,
        collector_pid=args.collector_pid,
        collector_start=args.collector_start,
        learning_kind="advantage-regression",
        physical_rollouts_required=208,
        frame_samples_required=56160,
        protected_current_parent_contexts=39,
        partition="TRAIN_CONSUMED",
        fresh_exam_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output_root / "commitment.json", declaration)
    began = time.monotonic()
    while (start := process_start(args.collector_pid)) is not None:
        if start != args.collector_start or time.monotonic() - began > 28800:
            raise ValueError("owned collection wait expired or PID reused; no automatic retry")
        time.sleep(10)
    stable()
    summary = _sealed(args.exploration_root / "training_summary.json")
    checked_collection(summary, declared)
    write_once(
        args.output_root / "collection-finished.json",
        dict(
            summary_hash=summary["report_hash"],
            new_physical_executions=234,
            exploration_rollouts=208,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    env = [
        "env",
        "OPENBLAS_NUM_THREADS=1",
        "PYTHONPATH="
        + ":".join(
            map(
                str,
                (
                    args.source_root,
                    args.source_root / "src",
                    args.source_root / "scripts",
                    args.core_root / "src",
                ),
            )
        ),
    ]

    def command(script: str, **values: Any) -> list[str]:
        return (
            env
            + [sys.executable, "-u", str(args.source_root / "scripts" / script)]
            + [v for k, val in values.items() for v in ("--" + k.replace("_", "-"), str(val))]
        )

    audited, learned = args.output_root / "audited-rollouts", args.output_root / "learned"
    run_stage(
        args.output_root,
        "audit-all-208",
        command(
            "rsi_fit_parallel_memory_motor.py",
            exploration_root=args.exploration_root,
            parent_bank_root=args.parent_bank_root,
            model=args.behavior_model,
            output_root=audited,
            behavior_kind="smooth-memory",
            audit_workers=4,
        )
        + ["--audit-only"],
    )
    stable()
    run_stage(
        args.output_root,
        "learn-208",
        command(
            "rsi_fit_current_memory_gradient.py",
            learning_root=audited,
            exploration_root=args.exploration_root,
            behavior_model=args.behavior_model,
            parent_bank_root=args.parent_bank_root,
            consolidated_memory=args.consolidated_memory,
            consolidation_manifest=args.consolidation_manifest,
            audit_source_root=args.source_root,
            core_root=args.core_root,
            system_reserve_path=args.system_reserve_path,
            learning_kind="advantage-regression",
            samples_per_course=16,
            output_root=learned,
        ),
    )
    stable()
    learning = _sealed(learned / "learning.json")
    run_stage(
        args.output_root,
        "fixed-counterexample",
        ["xvfb-run", "-a", "--server-args=-screen 0 1280x720x24 -nolisten tcp"]
        + command(
            "rsi_retest_current_memory_counterexample.py",
            reference_root=args.reference_root,
            reference_source=args.reference_source,
            reference_core=args.reference_core,
            qualified_parent_bank=args.parent_bank_root,
            qualified_parent_model=args.qualified_parent_model,
            candidate=learned / "model.json.gz",
            learning_report=learned / "learning.json",
            expected_model_hash=learning["model_hash"],
            expected_learning_hash=learning["report_hash"],
            output_root=args.output_root / "counterexample",
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            late_swing_policy=args.late_swing_policy,
            core_root=args.core_root,
            system_reserve_path=args.system_reserve_path,
            gpu=1,
        ),
    )
    stable()
    review = _sealed(args.output_root / "counterexample/independent_review.json")
    result = dict(
        schema="soccer.rsi.independent_motor_learning_result.v1",
        commitment_hash=hash_json(declaration),
        collection_hash=summary["report_hash"],
        learning_hash=learning["report_hash"],
        counterexample_hash=review["report_hash"],
        model_hash=learning["model_hash"],
        qualification="COUNTEREXAMPLE_PASSED_NEEDS_FULL_RETENTION_CPU_FRESH"
        if review["known_counterexample_resolved"] is True
        else "REJECTED_FIXED_COUNTEREXAMPLE",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
