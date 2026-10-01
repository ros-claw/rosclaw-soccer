"""Bind the preregistered second iteration to complete actual parent evidence.

Read-only qualification plus one immutable plan receipt. No simulator is run,
no fresh partition is opened and no model is promoted by this entry point.
"""

import argparse
import json
import shutil
from pathlib import Path

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.smooth_memory_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows


def folder_bytes(path: Path) -> int:
    files = [p for p in path.rglob("*") if p.is_file()]
    if not files:
        raise ValueError("complete physical report folder required for storage measurement")
    return sum(p.stat().st_size for p in files)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("plan", "parent-model", "bank-root", "transport-review", "storage-root", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    model = json.loads(args.parent_model.read_text())
    validate_model(model)
    bank = _sealed(args.bank_root / "validation_summary.json")
    review = _sealed(args.bank_root / "independent_review.json")
    transport = _sealed(args.transport_review)
    failures = qualified_memory_failure_rows(bank, review)
    if (
        plan["schema"] != "soccer.rsi.smooth_memory_round_two_plan.v1"
        or plan["partition"] != "TRAIN_CONSUMED"
        or plan["activation_ceiling"] != "SIM_ONLY"
        or plan["behavior_model_hash"] != model["model_hash"]
        or plan["behavior_generation"] != model["generation"]
        or model["generation"] != 1
        or bank["commitment"]["model_hash"] != model["model_hash"]
        or plan["course_selection"] != "ALL_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER"
        or plan["samples_per_failure_course"] != 8
        or plan["std_raw"] != 0.1
        or plan["sampling_rho"] != 0.9
        or plan["sampling_generation"] != 1
        or plan["sampling_seed_namespace"] != "GENERATION_STRIDE_100000"
        or plan["sampling_model_storage"] != "ATOMIC_GZIP_JSON_V1"
        or plan["compressed_transport_physical_equivalence_required"] is not True
        or review["candidate_high_quality"] - review["warm_high_quality"] < 1
        or transport["schema"] != "soccer.rsi.compressed_sampling_transport_independent_review.v1"
        or transport["complete_payload_equal"] is not True
        or transport["body_and_ball_trace_hashes_equal"] is not True
        or transport["actual_motor_actions_reconstructed"] != 300
        or transport["physical_executions_reviewed"] != 2
        or any(
            plan[k] is not False
            for k in (
                "fresh_holdout_open_authorized",
                "promotion_authorized",
                "hardware_authorized",
            )
        )
        or any(transport[k] is not False for k in ("promotion_authorized", "hardware_authorized"))
        or plan["system_disk_reserve_bytes"] != 100 * 1024**3
    ):
        raise ValueError("complete qualified actual first-generation parent and transport required")
    # Measure archived parent/current-actor output sizes, not an invented fixed
    # number of bytes per episode. The 10% margin is for wrapper/receipt growth.
    actor_sizes, parent_sizes = [], []
    for row in bank["rows"]:
        stem = f"seed{row['seed']}-lane{row['lane']}"
        actor_sizes.append(
            folder_bytes(args.bank_root / f"{stem}-candidate-actor")
            + (args.bank_root / f"logs/{stem}-candidate-actor.log").stat().st_size
        )
        parent_sizes.append(
            folder_bytes(args.bank_root / f"{stem}-reproduction-parent")
            + (args.bank_root / f"logs/{stem}-reproduction-parent.log").stat().st_size
        )
    actor_bytes, parent_bytes = max(actor_sizes), max(parent_sizes)
    samples = 8
    sampling_bytes = int(transport["compressed_bytes"] * 1.1)
    collection = int(
        1.1
        * len(failures)
        * ((samples + 1) * actor_bytes + parent_bytes + samples * sampling_bytes)
    )
    full_bank = int(1.1 * 52 * (2 * actor_bytes + parent_bytes))
    pilot = int(1.1 * 4 * (2 * actor_bytes + parent_bytes))
    other = 2 * 1024**3  # Fit NPZ/model, four CPU reports and logs.
    reserve = plan["system_disk_reserve_bytes"]
    free = shutil.disk_usage(args.storage_root).free
    if free < reserve + collection + full_bank + pilot + other:
        raise ValueError(
            f"complete round requires {reserve + collection + full_bank + pilot + other} "
            f"bytes including reserve; only {free} available"
        )
    result = dict(
        schema="soccer.rsi.smooth_memory_round_two_preflight.v1",
        plan_hash=hash_json(plan),
        parent_model_hash=model["model_hash"],
        parent_bank_hash=bank["report_hash"],
        parent_review_hash=review["report_hash"],
        transport_review_hash=transport["report_hash"],
        parent_high_quality=review["candidate_high_quality"],
        courses=[[r["seed"], r["lane"]] for r in failures],
        independent_failure_contexts=len(failures),
        samples_per_failure_course=8,
        planned_exploration_executions=len(failures) * 8,
        planned_collection_executions=len(failures) * 10,
        estimated_collection_bytes=collection,
        estimated_next_full_bank_bytes=full_bank,
        estimated_pilot_bytes=pilot,
        additional_fit_and_cpu_bytes=other,
        system_reserve_bytes=reserve,
        available_storage_bytes=free,
        storage_root=str(args.storage_root.resolve()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="CONSUMED_ITERATION_PREFLIGHT_ONLY_NOT_LEARNING_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
