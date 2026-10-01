"""Select all failures of a complete retained parent, never a rejected candidate."""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.smooth_memory_motor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows
from scripts.rsi_preflight_memory_bank_validation import check_pilot
from scripts.rsi_prepare_smooth_memory_round_two import folder_bytes


def qualify_parent_bank(
    model: dict[str, Any],
    parent: dict[str, Any],
    bank: dict[str, Any],
    review: dict[str, Any],
) -> list[dict[str, Any]]:
    failures = qualified_memory_failure_rows(bank, review)
    if (
        model["generation"] != 0
        or model["frozen_parent"] != parent
        or bank["commitment"]["model_hash"] != parent["model_hash"]
        or review["candidate_high_quality"] - review["warm_high_quality"] < 1
    ):
        raise ValueError("zero child of the actual complete retained improved parent required")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "model",
        "parent-model",
        "bank-root",
        "pilot-root",
        "transport-review",
        "storage-root",
        "output",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    model = json.loads(args.model.read_text())
    parent = json.loads(args.parent_model.read_text())
    validate_model(model)
    bank = _sealed(args.bank_root / "validation_summary.json")
    review = _sealed(args.bank_root / "independent_review.json")
    transport = _sealed(args.transport_review)
    pilot = _sealed(args.pilot_root / "validation_summary.json")
    pilot_review = _sealed(args.pilot_root / "independent_review.json")
    failures = qualify_parent_bank(model, parent, bank, review)
    check_pilot(pilot, pilot_review, model["model_hash"], baseline_hash=parent["model_hash"])
    if (
        plan["schema"] != "soccer.rsi.current_retained_parent_smooth_curriculum_plan.v1"
        or plan["activation_ceiling"] != "SIM_ONLY"
        or plan["partition"] != "TRAIN_CONSUMED"
        or plan["retained_parent_model_hash"] != parent["model_hash"]
        or plan["retained_parent_generation"] != parent["generation"]
        or model["generation"] != 0
        or model["frozen_parent"] != parent
        or bank["commitment"]["model_hash"] != parent["model_hash"]
        or review["candidate_high_quality"] - review["warm_high_quality"] < 1
        or plan["course_selection"] != "ALL_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER"
        or plan["samples_per_failure_course"] != 8
        or plan["std_raw"] != 0.1
        or plan["sampling_rho"] != 0.9
        or plan["exploration_stream"] != 1
        or plan["exploration_stream_namespace"] != "STREAM_STRIDE_20000000"
        or plan["sampling_workers"] != "SPAWN_ONE_CPU_AUDITOR_PER_GPU"
        or plan["sampling_model_storage"] != "ATOMIC_GZIP_JSON_V1"
        or transport["schema"] != "soccer.rsi.compressed_sampling_transport_independent_review.v1"
        or transport["complete_payload_equal"] is not True
        or transport["body_and_ball_trace_hashes_equal"] is not True
        or transport["actual_motor_actions_reconstructed"] != 300
        or transport["physical_executions_reviewed"] != 2
        or any(
            obj.get(k) is not False
            for obj in (plan, transport)
            for k in ("promotion_authorized", "hardware_authorized")
        )
        or plan["fresh_holdout_open_authorized"] is not False
        or plan["system_disk_reserve_bytes"] != 100 * 1024**3
    ):
        raise ValueError("complete retained actual parent, zero-child pilot and transport required")
    # Zero-head inheritance is proved on actual controls, not just constructor
    # arithmetic. Every warm/online body and ball trajectory must be identical.
    for row in pilot["rows"]:
        stem = f"seed{row['seed']}-lane{row['lane']}"
        warm = _sealed(args.pilot_root / f"{stem}-warm-actor/report.json")
        zero = _sealed(args.pilot_root / f"{stem}-online-actor/report.json")
        if any(warm[k] != zero[k] for k in ("body_trace_hash", "trace_hash")):
            raise ValueError("zero added residual changed the actual parent physics")
    actors, parents = [], []
    for row in bank["rows"]:
        stem = f"seed{row['seed']}-lane{row['lane']}"
        actors.append(folder_bytes(args.bank_root / f"{stem}-candidate-actor"))
        parents.append(folder_bytes(args.bank_root / f"{stem}-reproduction-parent"))
    actor_bytes, parent_bytes = max(actors), max(parents)
    collection = int(
        1.15
        * len(failures)
        * (9 * actor_bytes + parent_bytes + 8 * transport["compressed_bytes"] * 1.15)
    )
    bank_bytes = int(1.15 * 52 * (2 * actor_bytes + parent_bytes))
    pilot_bytes = int(1.15 * 4 * (2 * actor_bytes + parent_bytes))
    other = 2 * 1024**3
    free = shutil.disk_usage(args.storage_root).free
    reserve = plan["system_disk_reserve_bytes"]
    if free < reserve + collection + bank_bytes + pilot_bytes + other:
        raise ValueError("entire new curriculum/fit/CPU/full-bank budget plus reserve required")
    result = dict(
        schema="soccer.rsi.retained_parent_smooth_curriculum_preflight.v1",
        plan_hash=hash_json(plan),
        parent_model_hash=parent["model_hash"],
        behavior_model_hash=model["model_hash"],
        parent_bank_hash=bank["report_hash"],
        parent_review_hash=review["report_hash"],
        zero_child_pilot_hash=pilot_review["report_hash"],
        transport_review_hash=transport["report_hash"],
        parent_high_quality=review["candidate_high_quality"],
        courses=[[r["seed"], r["lane"]] for r in failures],
        exploration_stream=1,
        planned_collection_executions=10 * len(failures),
        planned_exploration_executions=8 * len(failures),
        estimated_collection_bytes=collection,
        estimated_full_bank_bytes=bank_bytes,
        estimated_pilot_bytes=pilot_bytes,
        additional_fit_and_cpu_bytes=other,
        storage_root=str(args.storage_root.resolve()),
        available_storage_bytes=free,
        system_reserve_bytes=reserve,
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        qualification="CONSUMED_PREFLIGHT_ONLY_NOT_PROMOTION_OR_LEARNING_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output, result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
