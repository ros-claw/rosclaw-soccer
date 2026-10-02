"""Explicit recovery: preserve failed evidence and copy complete owned controls.

Never overwrite the failed run or retry inside an execution. Original frozen
runner/Core are reused; only missing executions are newly run on GPUs 1/2.
"""

import argparse
import os
import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_memory_learning_rollouts import ordered_audits
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_collect_protected_phase_bank_validation import score
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_prepare_smooth_memory_round_two import folder_bytes
from scripts.rsi_train_protected_online_motor_v308 import _head
from scripts.rsi_validate_consolidated_bank_delta import DECLARED_INDICES, selected_indices


def native_failure_kind(log: str) -> str:
    if "Unable to allocate memory of size 671088640" in log:
        return "NATIVE_CONTACT_BUFFER_ALLOCATION_FAILED"
    if "[Fatal]" in log and "libX11.so.6!XOpenDisplay" in log:
        return "NATIVE_XDISPLAY_STARTUP_CRASH"
    raise ValueError("explicit observed native allocation/display failure required")


def copy_complete_control(source: Path, destination: Path, *, link: bool) -> None:
    if (
        type(link) is not bool
        or source.is_symlink()
        or any(p.is_symlink() for p in source.rglob("*"))
    ):
        raise ValueError("explicit local non-symlink control reuse required")
    if link:
        if source.stat().st_dev != destination.parent.stat().st_dev:
            raise ValueError("hard-link reuse requires the same local evidence volume")
        shutil.copytree(source, destination, copy_function=os.link)
    else:
        shutil.copytree(source, destination)


def execute_recovery_course(job: dict[str, Any]) -> dict[str, Any]:
    args, row = argparse.Namespace(**job["args"]), job["row"]
    seed, lane = row["seed"], row["lane"]
    common = dict(
        root=args.output_root,
        runner=args.execution_source / "scripts/rsi_isaac_vector_first_touch.py",
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        seed=seed,
        lane=lane,
        gpu=job["gpu"],
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
        execution_timeout_s=600,
        compressed_report=True,
        resume=True,
    )
    parent, _ = _run(**common, arm="reproduction", kind="parent")
    stem = f"seed{seed}-lane{lane}"
    historical = _sealed(args.parent_bank_root / f"{stem}-reproduction-parent/report.json")
    if any(parent[k] != historical[k] for k in ("body_trace_hash", "trace_hash")):
        raise ValueError("recovery changed physical parent")
    parent_path = args.output_root / f"{stem}-reproduction-parent/report.json"
    result = dict(
        index=row["index"], seed=seed, lane=lane, parent_report_hash=parent["report_hash"]
    )
    for arm, model in (("warm", args.parent_model), ("candidate", args.model)):
        raw, outcome = _run(
            **common, arm=arm, kind="actor", motor_step=model, parent_report_override=parent_path
        )
        if arm == "warm":
            historical = _sealed(args.parent_bank_root / f"{stem}-candidate-actor/report.json")
            if any(raw[k] != historical[k] for k in ("body_trace_hash", "trace_hash")):
                raise ValueError("recovery changed exact qualified NN parent physics")
        result[arm] = dict(
            report_hash=raw["report_hash"], high_quality=high_quality(outcome), **outcome
        )
    write_once(args.output_root / f"row-{row['index']}.json", result)
    print(f"CONSOLIDATED_DELTA_RECOVERY_COMPLETED index={row['index']}", flush=True)
    return result


def check_original_binding(commitment: dict[str, Any], runner: Path, core_root: Path) -> None:
    if (
        commitment["schema"] != "soccer.rsi.consolidated_bank_delta_commitment.v1"
        or commitment["indices"] != DECLARED_INDICES
        or commitment["physical_report_representation"] != "lossless_gzip_json"
        or commitment["source_commit"] != _head(runner.parent.parent)
        or commitment["runner_hash"] != hash_bytes(runner.read_bytes())
        or commitment["core_commit"] != _head(core_root)
        or any(commitment[k] is not False for k in ("promotion_authorized", "hardware_authorized"))
        or any(hash_bytes(Path(p).read_bytes()) != h for p, h in commitment["input_hashes"].items())
    ):
        raise ValueError("original exact frozen physical bindings required for recovery")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "failed-root",
        "failure-log",
        "execution-source",
        "model",
        "parent-model",
        "parent-bank-root",
        "rejected-bank-root",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--expected-complete-count", type=int, choices=range(1, 21), default=11)
    parser.add_argument("--workers", type=int, choices=(1, 2), default=2)
    parser.add_argument("--link-complete-controls", action="store_true")
    args = parser.parse_args()
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

    original = load_json_artifact(args.failed_root / "commitment.json")
    runner = args.execution_source / "scripts/rsi_isaac_vector_first_touch.py"
    check_original_binding(original, runner, args.core_root)
    if (
        not args.failure_log.is_file()
        or args.failure_log.is_symlink()
        or args.failure_log.resolve().parent != (args.failed_root / "logs").resolve()
        or (args.failed_root / args.failure_log.stem / "report.json.gz").exists()
        or (args.failed_root / "validation_summary.json").exists()
        or args.output_root.exists()
    ):
        raise ValueError("preserved incomplete native allocation failure and new root required")
    failure_kind = native_failure_kind(args.failure_log.read_text())
    model, parent = load_json_artifact(args.model), load_json_artifact(args.parent_model)
    if (
        model["model_hash"] != original["model_hash"]
        or parent["model_hash"] != original["warm_model_hash"]
    ):
        raise ValueError("recovery cannot change policy weights")
    bank = _sealed(args.parent_bank_root / "validation_summary.json")
    rejected = _sealed(args.rejected_bank_root / "validation_summary.json")
    indices = selected_indices(bank, rejected)
    if (
        bank["report_hash"] != original["parent_bank_hash"]
        or _sealed(args.rejected_bank_root / "independent_review.json")["report_hash"]
        != original["rejected_review_hash"]
        or hash_bytes(args.g1_usd.read_bytes()) != original["asset_hash"]
    ):
        raise ValueError("original immutable seven-course references required")
    complete = []
    for i in indices:
        row = bank["rows"][i]
        for arm, kind in (("reproduction", "parent"), ("warm", "actor"), ("candidate", "actor")):
            stem = f"seed{row['seed']}-lane{row['lane']}-{arm}-{kind}"
            folder = args.failed_root / stem
            if not (folder / "report.json.gz").is_file():
                continue
            if any(p.is_symlink() for p in folder.rglob("*")) or folder.is_symlink():
                raise ValueError("complete original controls must be local non-symlink evidence")
            raw = _sealed(folder / "report.json")
            if (
                raw["source_hash"] != original["runner_hash"]
                or (raw["training_course_seed"], raw["single_course_lane"])
                != (row["seed"], row["lane"])
                or raw["asset_hash"] != original["asset_hash"]
                or any(
                    raw[k] != hash_bytes((folder / name).read_bytes())
                    for k, name in (
                        ("trace_hash", "trace.npz"),
                        ("body_trace_hash", "body_trace.npz"),
                    )
                )
                or (
                    kind == "actor"
                    and raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
                    != original["warm_model_hash" if arm == "warm" else "model_hash"]
                )
            ):
                raise ValueError("original complete control identity changed")
            complete.append(folder)
    if len(complete) != args.expected_complete_count:
        raise ValueError("exact preregistered complete-control count required")
    new_count = 21 - len(complete)
    prior_failures = original.get("recovery", {}).get("original_native_failures_preserved", 0) + 1
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    if (
        args.link_complete_controls
        and args.failed_root.stat().st_dev != args.output_root.parent.stat().st_dev
    ):
        raise ValueError("hard-link reuse requires the same local evidence volume")
    largest = max(folder_bytes(f) for f in complete)
    reused_storage = 0 if args.link_complete_controls else sum(folder_bytes(f) for f in complete)
    budget = int(1.15 * (reused_storage + new_count * largest)) + 512 * 1024**2
    complete_file_pins = {
        str(p.relative_to(args.failed_root)): hash_bytes(p.read_bytes())
        for f in complete
        for p in f.rglob("*")
        if p.is_file()
    }
    capacity = capacity_check(args.output_root.parent, args.system_reserve_path, budget)
    args.output_root.mkdir()
    (args.output_root / "logs").mkdir()
    shutil.copy2(args.failure_log, args.output_root / "prior-native-failure.log")
    if "recovery" in original:
        inherited_log = args.failed_root / "prior-native-failure.log"
        if (
            hash_bytes(inherited_log.read_bytes())
            != original["recovery"]["preserved_failure_log_hash"]
        ):
            raise ValueError("ancestor native failure log changed")
        shutil.copy2(inherited_log, args.output_root / "ancestor-native-failure.log")
    for folder in complete:
        copy_complete_control(
            folder, args.output_root / folder.name, link=args.link_complete_controls
        )
        shutil.copy2(
            args.failed_root / "logs" / f"{folder.name}.log",
            args.output_root / "logs" / f"{folder.name}.log",
        )
        if _sealed(args.output_root / folder.name / "report.json") != _sealed(
            folder / "report.json"
        ):
            raise ValueError("recovery copy changed complete control")
    commitment = dict(
        original,
        capacity=capacity,
        recovery=dict(
            original_commitment_hash=hash_json(original),
            original_root=str(args.failed_root.resolve()),
            preserved_failure_log_hash=hash_bytes(args.failure_log.read_bytes()),
            orchestrator_source_hash=hash_bytes(Path(__file__).read_bytes()),
            reused_complete_physical_reports=len(complete),
            new_physical_executions_planned=new_count,
            original_native_failures_preserved=prior_failures,
            failure_kind=failure_kind,
            ancestor_recovery=original.get("recovery"),
            complete_control_file_hashes=complete_file_pins,
            complete_control_representation=(
                "same_volume_hard_links"
                if args.link_complete_controls
                else "independent_byte_copies"
            ),
            owned_native_gpu_schedule=list(range(1, args.workers + 1)),
            no_old_evidence_overwritten=True,
        ),
    )
    write_once(args.output_root / "commitment.json", commitment)
    rows: list[dict[str, Any]] = []
    jobs = [
        dict(row=bank["rows"][i], args=vars(args), gpu=1 + n % args.workers)
        for n, i in enumerate(indices)
    ]
    for start in range(0, len(jobs), args.workers):
        rows.extend(
            ordered_audits(
                execute_recovery_course, jobs[start : start + args.workers], args.workers
            )
        )
    check_original_binding(original, runner, args.core_root)
    if any(
        hash_bytes((args.failed_root / p).read_bytes()) != h
        or hash_bytes((args.output_root / p).read_bytes()) != h
        for p, h in complete_file_pins.items()
    ):
        raise ValueError("reused control files changed during recovery")
    result = dict(
        schema="soccer.rsi.consolidated_bank_delta_physics.v1",
        commitment=commitment,
        rows=rows,
        physical_executions=21,
        independent_contexts=7,
        new_physical_executions=new_count,
        reused_complete_physical_reports=len(complete),
        prior_native_failures=prior_failures,
        **score(rows),
        qualification="ALL_AFFECTED_CONSUMED_CASES_ONLY_NOT_FULL_BANK_OR_FRESH",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "validation_summary.json", result)
    print({k: v for k, v in result.items() if k not in ("rows", "commitment")}, flush=True)


if __name__ == "__main__":
    main()
