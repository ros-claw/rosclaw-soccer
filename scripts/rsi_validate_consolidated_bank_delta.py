"""SIM-only prescreen of ALL parent gains and ALL rejected-child bank deltas.

Includes regressions, gains and new out-of-play cases, never a favorable subset.
This 7-course diagnostic cannot qualify a parent, fresh exam or whole bank.
"""

import argparse
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.consolidated_smooth_motor import validate_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_memory_learning_rollouts import ordered_audits
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_collect_protected_phase_bank_validation import review_course, score
from scripts.rsi_compressed_bank_storage import capacity_check
from scripts.rsi_train_protected_online_motor_v308 import _head

DECLARED_INDICES = [6, 21, 23, 25, 35, 42, 49]


def selected_indices(parent: dict[str, Any], rejected: dict[str, Any]) -> list[int]:
    if len(parent["rows"]) != 52 or len(rejected["rows"]) != 52:
        raise ValueError("complete parent and rejected-child banks required")
    indices = []
    for i, (before, after) in enumerate(zip(parent["rows"], rejected["rows"], strict=True)):
        if (
            before["index"] != i
            or after["index"] != i
            or (before["seed"], before["lane"]) != (after["seed"], after["lane"])
            or before["candidate"]["high_quality"] != after["warm"]["high_quality"]
        ):
            raise ValueError("unaligned current-parent full-bank rows")
        new_parent = not before["warm"]["high_quality"] and before["candidate"]["high_quality"]
        changed = (
            any(
                after["warm"][key] != after["candidate"][key]
                for key in ("high_quality", "clean_foot_only")
            )
            or after["warm"]["maximum_lateral_excursion_m"]
            <= 4
            < after["candidate"]["maximum_lateral_excursion_m"]
        )
        if new_parent or changed:
            indices.append(i)
    if indices != DECLARED_INDICES:
        raise ValueError("exact complete preregistered seven-course delta required; no truncation")
    return indices


def execute_course(job: dict[str, Any]) -> dict[str, Any]:
    args, row = argparse.Namespace(**job["args"]), job["row"]
    seed, lane = row["seed"], row["lane"]
    common = dict(
        root=args.output_root,
        runner=Path(job["runner"]),
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
    )
    parent, _ = _run(**common, arm="reproduction", kind="parent")
    stem = f"seed{seed}-lane{lane}"
    old_parent = _sealed(args.parent_bank_root / f"{stem}-reproduction-parent/report.json")
    if any(parent[k] != old_parent[k] for k in ("body_trace_hash", "trace_hash")):
        raise ValueError("delta source changed frozen physical parent")
    parent_path = args.output_root / f"{stem}-reproduction-parent/report.json"
    result = dict(
        index=row["index"], seed=seed, lane=lane, parent_report_hash=parent["report_hash"]
    )
    for arm, path in (("warm", args.parent_model), ("candidate", args.model)):
        raw, outcome = _run(
            **common, arm=arm, kind="actor", motor_step=path, parent_report_override=parent_path
        )
        if arm == "warm":
            old = _sealed(args.parent_bank_root / f"{stem}-candidate-actor/report.json")
            if any(raw[k] != old[k] for k in ("body_trace_hash", "trace_hash")):
                raise ValueError("delta neural parent did not reproduce qualified physics")
        result[arm] = dict(
            report_hash=raw["report_hash"], high_quality=high_quality(outcome), **outcome
        )
    write_once(args.output_root / f"row-{row['index']}.json", result)
    print(f"CONSOLIDATED_BANK_DELTA_EXECUTED index={row['index']}", flush=True)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "model",
        "parent-model",
        "parent-bank-root",
        "rejected-bank-root",
        "transport-review",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    model, parent = load_json_artifact(args.model), load_json_artifact(args.parent_model)
    validate_model(model)
    if model["base_model"]["frozen_parent"] != parent:
        raise ValueError("delta must compare exact qualified NN parent")
    bank, review = (
        _sealed(args.parent_bank_root / n)
        for n in ("validation_summary.json", "independent_review.json")
    )
    qualified_memory_failure_rows(bank, review)
    rejected, rejected_review = (
        _sealed(args.rejected_bank_root / n)
        for n in ("validation_summary.json", "independent_review.json")
    )
    if (
        bank["commitment"]["model_hash"] != parent["model_hash"]
        or rejected["commitment"]["warm_model_hash"] != parent["model_hash"]
        or rejected["commitment"]["model_hash"] != model["base_model"]["model_hash"]
        or rejected_review["source_summary_hash"] != rejected["report_hash"]
        or rejected_review["physical_reports_reviewed"] != 156
    ):
        raise ValueError("delta loses independently reviewed parent/rejected-child bindings")
    indices = selected_indices(bank, rejected)
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    input_paths = [
        args.model,
        args.parent_model,
        args.transport_review,
        args.g1_usd,
        args.late_swing_policy,
        *[
            root / name
            for root in (args.parent_bank_root, args.rejected_bank_root)
            for name in ("validation_summary.json", "independent_review.json")
        ],
    ]
    pins = {str(p.resolve()): hash_bytes(p.read_bytes()) for p in input_paths}
    if args.review_only:
        summary = _sealed(args.output_root / "validation_summary.json")
        commitment = load_json_artifact(args.output_root / "commitment.json")
        if (
            summary["schema"] != "soccer.rsi.consolidated_bank_delta_physics.v1"
            or summary["physical_executions"] != 21
            or summary["independent_contexts"] != 7
            or commitment["indices"] != indices
            or summary["commitment"] != commitment
            or commitment["model_hash"] != model["model_hash"]
            or commitment["warm_model_hash"] != parent["model_hash"]
            or commitment["parent_bank_hash"] != bank["report_hash"]
            or commitment["rejected_review_hash"] != rejected_review["report_hash"]
            or [r["index"] for r in summary["rows"]] != indices
            or commitment["input_hashes"] != pins
            or any(
                obj.get(k) is not False
                for obj in (summary, commitment)
                for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("complete seven-course diagnostic required")
        jobs = [
            dict(
                root=str(args.output_root),
                index=i,
                course={
                    "seed": bank["rows"][i]["seed"],
                    "lane": bank["rows"][i]["lane"],
                    "parent_report": str(
                        args.parent_bank_root
                        / (
                            f"seed{bank['rows'][i]['seed']}-lane{bank['rows'][i]['lane']}-reproduction-parent/report.json"
                        )
                    ),
                },
                recorded=row,
                commitment=commitment,
            )
            for i, row in zip(indices, summary["rows"], strict=True)
        ]
        rows = list(ordered_audits(review_course, jobs, 4))
        if (
            any(summary[k] != v for k, v in score(rows).items())
            or _sealed(args.output_root / "validation_summary.json") != summary
            or load_json_artifact(args.output_root / "commitment.json") != commitment
            or any(hash_bytes(p.read_bytes()) != pins[str(p.resolve())] for p in input_paths)
        ):
            raise ValueError("delta measurements do not reconstruct")
        result = dict(
            schema="soccer.rsi.consolidated_bank_delta_review.v1",
            source_summary_hash=summary["report_hash"],
            physical_reports_reviewed=21,
            motor_frames_reconstructed=4200,
            **score(rows),
            qualification="ALL_AFFECTED_CONSUMED_CASES_ONLY_NOT_FULL_BANK_OR_FRESH",
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output_root / "independent_review.json", result)
        print(result, flush=True)
        return
    transport = _sealed(args.transport_review)
    if (
        transport["schema"] != "soccer.rsi.physical_report_transport_review.v1"
        or transport["model_hash"] != model["model_hash"]
        or transport["complete_payload_equal"] is not True
        or transport["physical_executions_added"] != 3
    ):
        raise ValueError("completed actual compressed transport proof required")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    budget = int(1.2 * 21 * transport["compressed_report_bytes"]) + 512 * 1024**2
    capacity = capacity_check(args.output_root.parent, args.system_reserve_path, budget)
    commitment = dict(
        schema="soccer.rsi.consolidated_bank_delta_commitment.v1",
        indices=indices,
        source_commit=_head(source),
        input_hashes=pins,
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        model_hash=model["model_hash"],
        warm_model_hash=parent["model_hash"],
        parent_bank_hash=bank["report_hash"],
        rejected_review_hash=rejected_review["report_hash"],
        transport_review_hash=transport["report_hash"],
        capacity=capacity,
        physical_report_representation="lossless_gzip_json",
        partition="TRAIN_CONSUMED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / "logs").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    jobs = [
        dict(row=bank["rows"][i], args=vars(args), runner=str(runner), gpu=n % 4)
        for n, i in enumerate(indices)
    ]
    # Round-sized groups guarantee at most one new owned native job per GPU.
    rows = []
    for start in range(0, len(jobs), 4):
        rows.extend(ordered_audits(execute_course, jobs[start : start + 4], 4))
    if (
        _head(source) != commitment["source_commit"]
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
        or _head(args.core_root) != commitment["core_commit"]
        or any(hash_bytes(p.read_bytes()) != pins[str(p.resolve())] for p in input_paths)
    ):
        raise ValueError("delta execution source drift")
    result = dict(
        schema="soccer.rsi.consolidated_bank_delta_physics.v1",
        commitment=commitment,
        rows=rows,
        physical_executions=21,
        independent_contexts=7,
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
