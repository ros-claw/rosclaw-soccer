"""One declared SIM curriculum: qualify parent, explore all failures, learn, verify.

No retries, automatic promotion, fresh exam or changes to motor mathematics.
Each child process exits successfully before its sealed evidence is consumed.
"""

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.smooth_memory_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_continue_memory_validation import run_stage
from scripts.rsi_prepare_retained_parent_smooth_curriculum import qualify_parent_bank
from scripts.rsi_prepare_smooth_memory_round_two import folder_bytes
from scripts.rsi_train_protected_online_motor_v308 import _head


def required_storage(bank_root: Path, rows: list[dict[str, Any]], failures: int) -> int:
    """Budget all collection, TWO pilots, fit/CPU and next full bank in advance."""
    actor = parent = 0
    for row in rows:
        stem = f"seed{row['seed']}-lane{row['lane']}"
        actor = max(actor, folder_bytes(bank_root / f"{stem}-candidate-actor"))
        parent = max(parent, folder_bytes(bank_root / f"{stem}-reproduction-parent"))
    # Sampling files can exceed the old zero-head compressed file; reserve
    # 100 MiB each, not its optimistic measured 37 MiB size.
    collection = failures * (9 * actor + parent + 8 * 100 * 1024**2)
    validation = (52 + 8) * (2 * actor + parent)
    return int(1.15 * (collection + validation)) + 2 * 1024**3


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "model",
        "parent-model",
        "parent-bank-root",
        "parent-pilot",
        "transport-review",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "scene",
        "bank",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--parent-cpu-root", type=Path, action="append", required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if len(args.parent_cpu_root) != 4:
        parser.error("four ordered independently reviewed actual-parent CPU cases required")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    ).strip():
        raise ValueError("clean frozen execution source required")
    plan = json.loads(args.plan.read_text())
    declared = json.loads(
        (
            source / "docs/rsi/protocols/current-retained-parent-smooth-curriculum-v367.json"
        ).read_text()
    )
    if plan != declared:
        raise ValueError("exact preregistered retained-parent protocol required")
    model = json.loads(args.model.read_text())
    parent = json.loads(args.parent_model.read_text())
    validate_model(model)
    bank = _sealed(args.parent_bank_root / "validation_summary.json")
    review = _sealed(args.parent_bank_root / "independent_review.json")
    failures = qualify_parent_bank(model, parent, bank, review)
    if parent["model_hash"] != plan["retained_parent_model_hash"]:
        raise ValueError("declared actual retained parent required")
    budget = required_storage(args.parent_bank_root, bank["rows"], len(failures))
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(args.output_root.parent).free
    if free < plan["system_disk_reserve_bytes"] + budget:
        raise ValueError(
            "entire two-pilot/collection/fit/CPU/full-bank budget plus reserve required"
        )
    root = args.output_root
    root.mkdir(exist_ok=False)
    # Bind every externally supplied input, including all four CPU references.
    files = [
        args.plan,
        args.model,
        args.parent_model,
        args.parent_pilot,
        args.transport_review,
        args.g1_usd,
        args.scene,
        args.bank,
        args.parent_bank_root / "validation_summary.json",
        args.parent_bank_root / "independent_review.json",
    ]
    files += [p / "independent_review.json" for p in args.parent_cpu_root]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in files}
    commitment = dict(
        schema="soccer.rsi.retained_parent_curriculum_continuation.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        input_file_hashes=pins,
        parent_review_hash=review["report_hash"],
        all_current_failure_courses=len(failures),
        planned_exploration_rollouts=8 * len(failures),
        planned_physical_executions=12 + 10 * len(failures) + 172,
        storage_budget_bytes=budget,
        available_storage_bytes=free,
        system_disk_reserve_bytes=plan["system_disk_reserve_bytes"],
        activation_ceiling="SIM_ONLY",
        partition="TRAIN_CONSUMED",
        fresh_holdout_open_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(root / "commitment.json", commitment)
    python = sys.executable

    def script(name: str) -> list[str]:
        return [python, "-u", str(source / "scripts" / name)]

    def options(**values: Any) -> list[str]:
        return [
            item
            for key, value in values.items()
            for item in ("--" + key.replace("_", "-"), str(value))
        ]

    common = options(
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        late_swing_policy=args.late_swing_policy,
        core_root=args.core_root,
    )
    display = ["xvfb-run", "-a", "--server-args=-screen 0 1280x720x24 -nolisten tcp"]
    pilot = root / "zero-pilot"
    run_stage(
        root,
        "zero-pilot",
        display
        + script("rsi_collect_online_step_validation.py")
        + common
        + options(
            output_root=pilot,
            step_model=args.parent_model,
            online_model=args.model,
            pilot_summary=args.parent_pilot,
        ),
    )
    run_stage(
        root,
        "zero-pilot-review",
        script("rsi_review_step_motor_physics.py")
        + options(root=pilot, output=pilot / "independent_review.json"),
    )
    run_stage(
        root,
        "curriculum-preflight",
        script("rsi_prepare_retained_parent_smooth_curriculum.py")
        + options(
            plan=args.plan,
            model=args.model,
            parent_model=args.parent_model,
            bank_root=args.parent_bank_root,
            pilot_root=pilot,
            transport_review=args.transport_review,
            storage_root=root,
            output=root / "preflight.json",
        ),
    )
    exploration, learned = root / "exploration", root / "learned"
    run_stage(
        root,
        "all-failure-exploration",
        display
        + script("rsi_collect_failed_step_courses.py")
        + common
        + options(
            output_root=exploration,
            bank_physics_root=args.parent_bank_root,
            warm_model=args.model,
            pilot_root=pilot,
            behavior_kind="smooth-memory",
            samples_per_course=8,
            exploration_stream=1,
        )
        + ["--compressed-sampling-models"],
    )
    run_stage(
        root,
        "audit-and-learn",
        script("rsi_fit_parallel_memory_motor.py")
        + options(
            behavior_kind="smooth-memory",
            audit_workers=4,
            exploration_root=exploration,
            parent_bank_root=args.parent_bank_root,
            model=args.model,
            output_root=learned,
        ),
    )
    command = (
        script("rsi_continue_memory_validation.py")
        + common
        + options(
            candidate=learned / "model.json",
            baseline=args.parent_model,
            baseline_pilot=args.parent_pilot,
            output_root=root / "validation",
            scene=args.scene,
            bank=args.bank,
            baseline_cpu_review_name="independent_review.json",
        )
    )
    for reference in args.parent_cpu_root:
        command += options(baseline_cpu_root=reference)
    run_stage(root, "candidate-validation", command)
    if any(hash_bytes(Path(p).read_bytes()) != h for p, h in pins.items()) or (
        _head(source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
    ):
        raise ValueError("curriculum input or source drift")
    result = _sealed(root / "validation/result.json")
    report = dict(
        schema="soccer.rsi.retained_parent_curriculum_continuation_result.v1",
        commitment_hash=hash_json(commitment),
        candidate_result_hash=result["report_hash"],
        consumed_gain=result["consumed_gain"],
        current_parent_retained=result["current_parent_retained"],
        qualification="COMPLETE_CONSUMED_ITERATION_NOT_FRESH_OR_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    report["report_hash"] = hash_json(report)
    write_once(root / "result.json", report)
    print(report, flush=True)


if __name__ == "__main__":
    main()
