"""Seal a COMPLETE explicitly recovered consumed collection, not a clean run.

No sampler, optimizer, retry or promotion runs here. Original failure records
remain intact. A downstream learner must still independently audit ALL samples.
"""

import argparse
import re
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_continue_independent_motor_learning import checked_collection


def recovered_summary(
    declared: dict[str, Any],
    rows: list[dict[str, Any]],
    started: dict[str, Any],
    finished: dict[str, Any],
    terminal: dict[str, Any],
) -> dict[str, Any]:
    for obj in (started, finished):
        if obj.get("report_hash") != hash_json(
            {k: v for k, v in obj.items() if k != "report_hash"}
        ):
            raise ValueError("sealed explicit recovery required")
    if (
        declared.get("schema") != "soccer.rsi.smooth_memory_exploration_commitment.v1"
        or declared.get("behavior_kind") != "OUTPUT_MEMORY_CURRENT_PARENT_AR1"
        or declared.get("sampling_rho") != 0.9
        or declared.get("std_raw") != 0.1
        or declared.get("execution_timeout_s") != 600
        or started.get("schema") != "soccer.rsi.explicit_sampling_shard_recovery.v1"
        or finished.get("schema") != "soccer.rsi.explicit_sampling_shard_recovery_result.v1"
        or started.get("original_commitment_hash") != hash_json(declared)
        or finished.get("source_commitment_hash") != started["report_hash"]
        or started.get("actual_worker_commit") != declared.get("source_commit")
        or started.get("actual_core_commit") != declared.get("core_commit")
        or type(started.get("gpu")) is not int
        or started["gpu"] != 0
        or started.get("course_indices") != [0, 4, 8, 12]
        or finished.get("completed_course_indices") != [0, 4, 8, 12]
        or started.get("automatic_retry") is not False
        or started.get("reused_reports_are_not_new_physics") is not True
        or started.get("original_failed_run_must_be_preserved") is not True
        or started.get("completed_reports_require_full_reaudit") is not True
        or finished.get("original_failed_log_preserved") is not True
        or finished.get("original_existing_bytes_preserved") is not True
        or finished.get("entire_collection_sealed") is not False
        or finished.get("learning_authorized") is not False
        or terminal.get("schema") != "soccer.rsi.interrupted_independent_sampling_status.v1"
        or terminal.get("collector_source_commit") != declared.get("source_commit")
        or terminal.get("core_commit") != declared.get("core_commit")
        or type(terminal.get("collector_exit_code")) is not int
        or terminal["collector_exit_code"] != 1
        or terminal.get("completed_exploration_samples") != 179
        or type(terminal.get("completed_exploration_samples")) is not int
        or terminal.get("required_exploration_samples") != 208
        or terminal.get("missing_exploration_samples") != 29
        or terminal.get("complete_course_rows") != [0, 1, 2, 3, 4, 5, 6, 7, 9, 10, 11]
        or terminal.get("failed_native_course") != declared["courses"][8]
        or terminal.get("failed_native_sample") != 3
        or any(
            obj.get(key) is not False
            for obj in (started, finished, terminal)
            for key in ("promotion_authorized", "hardware_authorized")
        )
    ):
        raise ValueError("same original failed collection and explicit complete recovery required")
    summary = dict(
        schema="soccer.rsi.smooth_memory_failure_exploration.v1",
        commitment=declared,
        rows=rows,
        independent_contexts=13,
        exploration_executions=208,
        physical_executions=234,
        qualification="TRAIN_CONSUMED_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    checked_collection(summary, declared)
    hashes = []
    for row in rows:
        hashes.extend([row.get("parent_report_hash"), row.get("greedy", {}).get("report_hash")])
        for sample in row["samples"]:
            if type(sample.get("high_quality")) is not bool:
                raise ValueError("explicit unchanged sample outcome labels required")
            hashes.append(sample.get("report_hash"))
    if any(type(h) is not str or not re.fullmatch(r"sha256:[0-9a-f]{64}", h) for h in hashes):
        raise ValueError("all 234 physical report identities required")
    if len(set(hashes)) != 234:
        raise ValueError("reused evidence cannot masquerade as different executions")
    summary.update(
        high_quality_samples=sum(s["high_quality"] for r in rows for s in r["samples"]),
        recovery_provenance=dict(
            original_collector_exit_code=terminal["collector_exit_code"],
            original_terminal_status_hash=hash_json(terminal),
            recovery_started_hash=started["report_hash"],
            recovery_finished_hash=finished["report_hash"],
            original_complete_exploration_samples=179,
            newly_completed_exploration_samples=29,
            original_complete_control_executions=24,
            newly_completed_control_executions=2,
            new_physical_executions_during_recovery=31,
            reused_physical_executions=203,
            independent_audit_required=True,
            original_failure_is_not_erased=True,
            clean_original_collector_success=False,
            fresh_exam_authorized=False,
        ),
    )
    summary["report_hash"] = hash_json(summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--collection-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.collection_root
    if root.is_symlink() or not root.is_dir():
        raise ValueError("original regular local collection required")
    files = [
        root / n
        for n in (
            "commitment.json",
            "explicit-shard0-recovery1-started.json",
            "explicit-shard0-recovery1-finished.json",
            "interrupted-collection-status-v403.json",
            *(f"row-{i}.json" for i in range(13)),
        )
    ]
    if any(p.is_symlink() or not p.is_file() for p in files):
        raise ValueError("ALL original/recovered rows and failure ledgers required")
    pins = {str(p): hash_bytes(p.read_bytes()) for p in files}
    declared, started, finished, terminal, *rows = [load_json_artifact(p) for p in files]
    summary = recovered_summary(declared, rows, started, finished, terminal)
    preserved = {**started["input_file_hashes"], **started["existing_physical_file_hashes"]}
    if any(hash_bytes(Path(p).read_bytes()) != h for p, h in preserved.items()):
        raise ValueError("original source/input/physical bytes changed")
    failed = started["original_failed_attempt"]
    archived = (
        root / "failed-native-attempt-20261003-course8-sample3" / Path(failed["log_path"]).name
    )
    if archived.is_symlink() or hash_bytes(archived.read_bytes()) != failed["log_sha256"]:
        raise ValueError("original failed native attempt was not preserved")
    for path, expected in terminal["input_file_hashes"].items():
        original = archived if path == failed["log_path"] else Path(path)
        if hash_bytes(original.read_bytes()) != expected:
            raise ValueError("original interrupted run log evidence changed")
    for i, (seed, lane) in enumerate(declared["courses"]):
        for arm, kind in [("reproduction", "parent"), ("greedy", "actor")] + [
            (f"sample-{s}", "actor") for s in range(16)
        ]:
            folder = root / f"seed{seed}-lane{lane}-{arm}-{kind}"
            names = ["trace.npz", "body_trace.npz"]
            if kind == "actor":
                names += ["contact_motor_trace.npz", "late_swing_action_trace.npz"]
            paths = [resolve_physical_report(folder / "report.json"), *(folder / n for n in names)]
            if folder.is_symlink() or any(p.is_symlink() or not p.is_file() for p in paths):
                raise ValueError(f"complete regular physical artifacts required for course {i}")
            pins.update({str(p): hash_bytes(p.read_bytes()) for p in paths})
    if any(hash_bytes(Path(p).read_bytes()) != h for p, h in pins.items()):
        raise ValueError("collection changed during complete recovery sealing")
    receipt = dict(
        schema="soccer.rsi.explicit_collection_recovery_seal.v1",
        summary_hash=summary["report_hash"],
        file_hashes=pins,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        independent_audit_required=True,
        learning_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    receipt["report_hash"] = hash_json(receipt)
    write_once(root / "explicit-collection-recovery-seal.json", receipt)
    write_once(root / "training_summary.json", summary)
    print("COMPLETE_RECOVERED_COLLECTION_SEALED", summary["report_hash"], flush=True)


if __name__ == "__main__":
    main()
