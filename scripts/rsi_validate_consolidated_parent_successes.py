"""Actual counterfactual on ALL newly gained parent courses; no full-bank claim.

Reproduce parent and rejected child, then test consolidation. Never promote the
rejected child or select just its failures. Every command is independently
reconstructed; no fresh, hardware, retries or hidden replacement courses.
"""

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.consolidated_smooth_motor import validate_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_audit_memory_learning_rollouts import ordered_audits
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_failed_step_courses import qualified_memory_failure_rows
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_prepare_smooth_memory_round_two import folder_bytes
from scripts.rsi_train_protected_online_motor_v308 import _head


def newly_successful_rows(bank: dict[str, Any], review: dict[str, Any]) -> list[dict[str, Any]]:
    qualified_memory_failure_rows(bank, review)
    rows = [
        r for r in bank["rows"] if not r["warm"]["high_quality"] and r["candidate"]["high_quality"]
    ]
    if not 1 <= len(rows) <= 4:
        raise ValueError("one to four complete newly gained parent courses; never truncate")
    return rows


def execute_course(job: dict[str, Any]) -> dict[str, Any]:
    row, args = job["row"], argparse.Namespace(**job["args"])
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
    )
    parent, _ = _run(**common, arm="reproduction", kind="parent")
    historical_parent = _sealed(
        args.parent_bank_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
    )
    if any(parent[k] != historical_parent[k] for k in ("body_trace_hash", "trace_hash")):
        raise ValueError("new runner changed frozen parent physics")
    parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
    record = dict(seed=seed, lane=lane, parent_report_hash=parent["report_hash"])
    for arm, model_path, old_root in (
        ("qualified", args.parent_model, args.parent_bank_root),
        ("rejected", args.rejected_model, args.rejected_bank_root),
        ("consolidated", args.model, None),
    ):
        raw, outcome = _run(
            **common,
            arm=arm,
            kind="actor",
            motor_step=model_path,
            parent_report_override=parent_path,
        )
        if old_root is not None:
            old = _sealed(old_root / f"seed{seed}-lane{lane}-candidate-actor/report.json")
            if any(raw[k] != old[k] for k in ("body_trace_hash", "trace_hash")):
                raise ValueError("counterfactual actor did not reproduce historical physics")
        record[arm] = dict(
            report_hash=raw["report_hash"], high_quality=high_quality(outcome), **outcome
        )
    write_once(args.output_root / f"row-{seed}-{lane}.json", record)
    print(f"CONSOLIDATION_COUNTERFACTUAL_EXECUTED seed={seed} lane={lane}", flush=True)
    return record


def review_course(job: dict[str, Any]) -> None:
    root, row, commitment = Path(job["root"]), job["row"], job["commitment"]
    stem = f"seed{row['seed']}-lane{row['lane']}"
    parent_folder = root / f"{stem}-reproduction-parent"
    parent = _sealed(parent_folder / "report.json")
    audit_lateral_approach(parent_folder)
    if parent["report_hash"] != row["parent_report_hash"]:
        raise ValueError("counterfactual parent changed")
    if (parent["training_course_seed"], parent["single_course_lane"]) != (row["seed"], row["lane"]):
        raise ValueError("counterfactual course identity changed")
    for arm in ("qualified", "rejected", "consolidated"):
        folder = root / f"{stem}-{arm}-actor"
        raw = _sealed(folder / "report.json")
        measured = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
        if (
            raw["report_hash"] != row[arm]["report_hash"]
            or raw["parent_report_hash"] != parent["report_hash"]
            or (raw["training_course_seed"], raw["single_course_lane"])
            != (row["seed"], row["lane"])
            or raw["environments"][0]["course"] != parent["environments"][0]["course"]
            or raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
            != commitment[arm + "_model_hash"]
            or any(measured[k] != row[arm][k] for k in measured)
        ):
            raise ValueError("counterfactual model or measured outcome changed")


def counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        qualified_high_quality=sum(r["qualified"]["high_quality"] for r in rows),
        rejected_high_quality=sum(r["rejected"]["high_quality"] for r in rows),
        consolidated_high_quality=sum(r["consolidated"]["high_quality"] for r in rows),
        old_high_quality_loss=sum(
            r["qualified"]["high_quality"] and not r["consolidated"]["high_quality"] for r in rows
        ),
        old_clean_foot_loss=sum(
            r["qualified"]["clean_foot_only"] and not r["consolidated"]["clean_foot_only"]
            for r in rows
        ),
        new_out_of_play=sum(
            r["qualified"]["maximum_lateral_excursion_m"]
            <= 4
            < r["consolidated"]["maximum_lateral_excursion_m"]
            for r in rows
        ),
        safe_pelvis=all(r["consolidated"]["minimum_pelvis_z_m"] >= 0.65 for r in rows),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "plan",
        "model",
        "parent-model",
        "rejected-model",
        "parent-bank-root",
        "rejected-bank-root",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--review-only", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    plan = json.loads(args.plan.read_text())
    declared = json.loads(
        (
            source / "docs/rsi/protocols/current-parent-consolidation-counterfactual-v371.json"
        ).read_text()
    )
    if plan != declared:
        raise ValueError("exact preregistered consolidation counterfactual required")
    if args.review_only:
        summary = _sealed(args.output_root / "validation_summary.json")
        commitment = json.loads((args.output_root / "commitment.json").read_text())
        parent_bank = _sealed(args.parent_bank_root / "validation_summary.json")
        parent_review = _sealed(args.parent_bank_root / "independent_review.json")
        selected = newly_successful_rows(parent_bank, parent_review)
        if (
            summary["schema"] != "soccer.rsi.consolidation_counterfactual_physics.v1"
            or commitment["courses"] != [[r["seed"], r["lane"]] for r in selected]
            or commitment["parent_bank_hash"] != parent_bank["report_hash"]
            or commitment["parent_review_hash"] != parent_review["report_hash"]
            or commitment["plan_hash"] != hash_json(plan)
            or any(
                obj.get(k) is not False
                for obj in (summary, commitment)
                for k in ("promotion_authorized", "hardware_authorized")
            )
            or commitment["fresh_holdout_open_authorized"] is not False
        ):
            raise ValueError("declared complete new parent capabilities and SIM boundary required")
        if summary["commitment"] != commitment or summary["courses"] != commitment["courses"]:
            raise ValueError("unchanged complete counterfactual required")
        rows = summary["rows"]
        if [[r["seed"], r["lane"]] for r in rows] != commitment["courses"] or summary[
            "physical_executions"
        ] != 4 * len(rows):
            raise ValueError("all declared counterfactual courses required")
        list(
            ordered_audits(
                review_course,
                [dict(root=str(args.output_root), row=r, commitment=commitment) for r in rows],
                4,
            )
        )
        measured = counts(rows)
        if (
            _sealed(args.output_root / "validation_summary.json")["report_hash"]
            != summary["report_hash"]
        ):
            raise ValueError("counterfactual summary changed during reconstruction")
        if any(summary[k] != v for k, v in measured.items()):
            raise ValueError("counterfactual scoring changed")
        result = dict(
            schema="soccer.rsi.consolidation_counterfactual_review.v1",
            source_summary_hash=summary["report_hash"],
            existing_physical_reports_reviewed=4 * len(rows),
            motor_frames_reconstructed=3 * 300 * len(rows),
            **measured,
            qualification="CONSUMED_NEW_PARENT_CAPABILITY_COUNTERFACTUAL_NOT_FULL_BANK_OR_FRESH",
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(args.output_root / "independent_review.json", result)
        print(result, flush=True)
        return
    model = json.loads(args.model.read_text())
    validate_model(model)
    parent, rejected = [json.loads(p.read_text()) for p in (args.parent_model, args.rejected_model)]
    bank = _sealed(args.parent_bank_root / "validation_summary.json")
    review = _sealed(args.parent_bank_root / "independent_review.json")
    rows = newly_successful_rows(bank, review)
    if (
        parent["model_hash"] != plan["parent_model_hash"]
        or rejected["model_hash"] != plan["rejected_base_model_hash"]
        or len(rows) != plan["expected_courses"]
    ):
        raise ValueError("declared parent, rejected base and complete four-course set required")
    if (
        model["base_model"] != rejected
        or rejected["frozen_parent"] != parent
        or model["consolidation_manifest"]["bank_summary_hash"] != bank["report_hash"]
        or model["consolidation_manifest"]["bank_review_hash"] != review["report_hash"]
    ):
        raise ValueError("actual qualified NN and explicit rejected-child counterfactual required")
    old = _sealed(args.rejected_bank_root / "validation_summary.json")
    old_review = _sealed(args.rejected_bank_root / "independent_review.json")
    if (
        old_review["source_summary_hash"] != old["report_hash"]
        or old_review["physical_reports_reviewed"] != 156
    ):
        raise ValueError("complete independent rejected-child counterexample required")
    if (
        old["commitment"]["model_hash"] != rejected["model_hash"]
        or old["commitment"]["warm_model_hash"] != parent["model_hash"]
    ):
        raise ValueError("counterfactual policy bindings changed")
    maximum_actor = max(
        folder_bytes(args.rejected_bank_root / f"seed{r['seed']}-lane{r['lane']}-candidate-actor")
        for r in rows
    )
    # Candidate embeds additional memory; count it twice per report plus 15%
    # and 256 MiB overhead. Zero additional source models are allocated here.
    budget = (
        int(1.15 * len(rows) * (3 * maximum_actor + 2 * len(json.dumps(model["memory"]).encode())))
        + 256 * 1024**2
    )
    if shutil.disk_usage(args.output_root.parent).free < 100 * 1024**3 + budget:
        raise ValueError("bounded counterfactual budget plus 100 GiB reserve required")
    args.output_root.mkdir(exist_ok=False)
    (args.output_root / "logs").mkdir()
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    commitment = dict(
        schema="soccer.rsi.consolidation_counterfactual_commitment.v1",
        plan_hash=hash_json(plan),
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        qualified_model_hash=parent["model_hash"],
        rejected_model_hash=rejected["model_hash"],
        consolidated_model_hash=model["model_hash"],
        courses=[[r["seed"], r["lane"]] for r in rows],
        course_selection="ALL_NEWLY_SUCCESSFUL_QUALIFIED_PARENT_COURSES",
        partition="TRAIN_CONSUMED",
        parent_bank_hash=bank["report_hash"],
        parent_review_hash=review["report_hash"],
        rejected_bank_hash=old["report_hash"],
        rejected_review_hash=old_review["report_hash"],
        storage_budget_bytes=budget,
        fresh_holdout_open_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.output_root / "commitment.json", commitment)
    measured_rows = list(
        ordered_audits(
            execute_course,
            [
                dict(row=r, args=vars(args), runner=str(runner), gpu=i % 4)
                for i, r in enumerate(rows)
            ],
            4,
        )
    )
    if (
        _head(source) != commitment["source_commit"]
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
    ):
        raise ValueError("counterfactual source drift")
    summary = dict(
        schema="soccer.rsi.consolidation_counterfactual_physics.v1",
        commitment=commitment,
        courses=commitment["courses"],
        rows=measured_rows,
        physical_executions=4 * len(rows),
        **counts(measured_rows),
        qualification="CONSUMED_COUNTERFACTUAL_NOT_FULL_BANK_OR_FRESH",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "validation_summary.json", summary)
    print({k: v for k, v in summary.items() if k not in ("rows", "commitment")}, flush=True)


if __name__ == "__main__":
    main()
