"""Expand real on-policy exploration to every consumed warm-actor failure.

Failures are selected from a sealed complete comparison, not from the fresh
exam. Each finished course is journaled before the next starts. No learning or
promotion occurs in this collector; every sampled motor command is audited.
"""

import argparse
import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.step_motor_network import validate_model
from rosclaw_soccer.rsi.stochastic_step_execution import make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def sampling_seed(course: int, sample: int, *, generation: int = 0) -> int:
    if (
        type(course) is not int
        or type(sample) is not int
        or not 0 <= course < 52
        or not 0 <= sample < 16
        or type(generation) is not int
        or not 0 <= generation < 32
    ):
        raise ValueError("bounded declared curriculum/sample index required")
    # Keep the declared historical gen0 paired-noise ablation unchanged.
    # Later policy iterations must not replay the random stream used to fit
    # their parent and call that independent on-policy exploration.
    return 202610335 + generation * 100000 + course * 100 + sample


def failure_rows(
    summary: dict[str, Any], review: dict[str, Any], *, arm: str = "warm"
) -> list[dict[str, Any]]:
    """Complete paired consumed review is required before automatic selection."""
    rows = summary.get("rows", [])
    if (
        arm not in ("warm", "candidate")
        or summary.get("schema") != "soccer.rsi.protected_phase_bank_physics.v1"
        or summary.get("commitment", {}).get("partition") != "TRAIN_CONSUMED"
        or summary.get("physical_executions") != 156
        or summary.get("independent_contexts") != 52
        or len(rows) != 52
        or review.get("schema") != "soccer.rsi.protected_phase_bank_review.v1"
        or review.get("source_summary_hash") != summary.get("report_hash")
        or review.get("physical_reports_reviewed") != 156
        or review.get("motor_frames_reconstructed") != 31200
        or any(
            obj.get(k) is not False
            for obj in (summary, review)
            for k in ("promotion_authorized", "hardware_authorized")
        )
        or len({(r["seed"], r["lane"]) for r in rows}) != 52
        or [r["index"] for r in rows] != list(range(52))
        or any(type(r[arm].get("high_quality")) is not bool for r in rows)
        or sum(r[arm]["high_quality"] for r in rows) != summary[f"{arm}_high_quality"]
        or summary[f"{arm}_high_quality"] != review[f"{arm}_high_quality"]
    ):
        raise ValueError("complete independently reviewed consumed physics required")
    failures = [r for r in rows if not r[arm]["high_quality"]]
    if not failures:
        raise ValueError("no warm-actor failures to train")
    return failures


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "bank-physics-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "warm-model",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--samples-per-course", type=int, default=8)
    parser.add_argument(
        "--behavior-kind", choices=("warm", "output-memory", "smooth-memory"), default="warm"
    )
    parser.add_argument("--pilot-root", type=Path)
    parser.add_argument("--first-four-courses", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if not 4 <= args.samples_per_course <= 16:
        parser.error("four to sixteen preregistered samples per failure required")
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    warm = json.loads(args.warm_model.read_text())
    arm = "warm"
    make_view: Callable[..., dict[str, Any]] = make_sampling_view
    if args.first_four_courses and args.behavior_kind != "smooth-memory":
        parser.error("the fixed four-course ablation is only declared for smooth exploration")
    if args.behavior_kind in ("output-memory", "smooth-memory"):
        from rosclaw_soccer.rsi.output_memory_step_motor import (
            make_sampling_view as output_sampling,
        )
        from rosclaw_soccer.rsi.output_memory_step_motor import validate_model as output_validate

        if args.behavior_kind == "smooth-memory":
            from rosclaw_soccer.rsi.smooth_memory_motor import (
                make_sampling_view as smooth_sampling,
            )
            from rosclaw_soccer.rsi.smooth_memory_motor import validate_model as smooth_validate

            smooth_validate(warm)
            make_view = smooth_sampling
        else:
            output_validate(warm)
            make_view = output_sampling
        if warm["generation"] >= 32 or args.pilot_root is None:
            parser.error("memory exploration requires a qualified bounded current parent")
        pilot = _sealed(args.pilot_root / "validation_summary.json")
        pilot_review = _sealed(args.pilot_root / "independent_review.json")
        if (
            pilot["commitment"]["model_hash"] != warm["model_hash"]
            or pilot_review["source_summary_hash"] != pilot["report_hash"]
            or pilot_review["actual_reports_reviewed"] != 12
            or pilot_review["safe_pelvis"] is not True
            or any(
                pilot_review[k] != 0
                for k in ("old_high_quality_loss", "old_clean_foot_loss", "new_out_of_play")
            )
            or any(
                pilot_review.get(k) is not False
                for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError("independently qualified retained current parent required")
        base = warm
        arm = "candidate"
    elif warm.get("schema") != "soccer.rsi.compiled_step_motor_decoder.v1":
        parser.error("frozen compiled warm actor required")
    else:
        base = warm["base_model"]
        validate_model(base)
    bank = _sealed(args.bank_physics_root / "validation_summary.json")
    review = _sealed(args.bank_physics_root / "independent_review.json")
    courses = failure_rows(bank, review, arm=arm)
    total_failure_courses = len(courses)
    if args.first_four_courses:
        if len(courses) < 4:
            parser.error("four failure courses required before ablation allocation")
        courses = courses[:4]
    if arm == "candidate" and warm["generation"] > 0:
        expected_hash = warm["model_hash"]
    elif args.behavior_kind == "smooth-memory":
        parent = warm["frozen_parent"]
        expected_hash = (
            parent["model_hash"]
            if parent["generation"] > 0
            else parent["frozen_parent"]["model_hash"]
        )
    else:
        expected_hash = (
            warm["frozen_parent"]["model_hash"] if arm == "candidate" else warm["model_hash"]
        )
    if (
        bank["commitment"]["model_hash" if arm == "candidate" else "warm_model_hash"]
        != expected_hash
    ):
        parser.error("failure source used a different warm actor")
    views = [
        make_view(
            base,
            seed=sampling_seed(i, s, generation=warm["generation"] if arm == "candidate" else 0),
            std=0.1,
        )
        for i in range(len(courses))
        for s in range(args.samples_per_course)
    ]
    commitment = dict(
        schema="soccer.rsi.failure_step_exploration_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        helper_hash=hash_bytes(
            (source / "scripts/rsi_collect_approach_lateral_tracking_v286.py").read_bytes()
        ),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        base_model_hash=base["model_hash"],
        warm_model_hash=warm["model_hash"],
        bank_hash=bank["report_hash"],
        bank_review_hash=review["report_hash"],
        courses=[[r["seed"], r["lane"]] for r in courses],
        samples_per_course=args.samples_per_course,
        std_raw=0.1,
        sampling_view_hashes=[v["model_hash"] for v in views],
        execution_timeout_s=600,
        partition="TRAIN_CONSUMED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if arm == "candidate":
        commitment.update(
            schema="soccer.rsi.output_memory_exploration_commitment.v1",
            behavior_kind="OUTPUT_MEMORY_CURRENT_PARENT",
            failure_reference_arm=arm,
            pilot_hash=pilot["report_hash"],
            pilot_review_hash=pilot_review["report_hash"],
            frozen_parent_model_hash=warm["parent_model_hash"],
        )
        if warm["generation"] > 0:
            commitment["failure_reference_model_hash"] = expected_hash
            commitment["sampling_generation"] = warm["generation"]
            commitment["sampling_seed_namespace"] = "GENERATION_STRIDE_100000"
    if args.behavior_kind == "smooth-memory":
        commitment.update(
            schema="soccer.rsi.smooth_memory_exploration_commitment.v1",
            behavior_kind="OUTPUT_MEMORY_CURRENT_PARENT_AR1",
            frozen_parent_model_hash=warm["parent_model_hash"],
            failure_reference_model_hash=expected_hash,
            failure_reference_total_courses=total_failure_courses,
            course_selection="FIRST_FOUR_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER"
            if args.first_four_courses
            else "ALL_CURRENT_PARENT_FAILURES_IN_SEALED_ORDER",
            sampling_rho=0.9,
        )
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    for folder in ("logs", "models"):
        (args.output_root / folder).mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)
    for i, view in enumerate(views):
        write_once(args.output_root / "models" / f"sample-{i}.json", view)

    def worker(gpu: int) -> list[dict[str, Any]]:
        records = []
        for i in range(gpu, len(courses), 4):
            old = courses[i]
            seed, lane = old["seed"], old["lane"]
            common = dict(
                root=args.output_root,
                runner=runner,
                isaac_python=args.isaac_python,
                g1_usd=args.g1_usd,
                model_root=args.model_root,
                actor=args.late_swing_policy,
                seed=seed,
                lane=lane,
                gpu=gpu,
                gain=1.2,
                negative_only=True,
                core_root=args.core_root,
                resume=args.resume,
                execution_timeout_s=600,
            )
            for pending_arm, kind in [("reproduction", "parent"), ("greedy", "actor")] + [
                (f"sample-{s}", "actor") for s in range(args.samples_per_course)
            ]:
                stem = f"seed{seed}-lane{lane}-{pending_arm}-{kind}"
                if (args.output_root / "logs" / f"{stem}.log").exists() and not (
                    args.output_root / stem / "report.json"
                ).is_file():
                    raise ValueError(
                        "preserve and explicitly archive failed attempt before recovery"
                    )
            parent, _ = _run(**common, arm="reproduction", kind="parent")
            parent_path = (
                args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            )
            greedy, outcome = _run(
                **common,
                arm="greedy",
                kind="actor",
                motor_step=args.warm_model,
                parent_report_override=parent_path,
            )
            old_folder = args.bank_physics_root / f"seed{seed}-lane{lane}-{arm}-actor"
            old_report = _sealed(old_folder / "report.json")
            old_parent = _sealed(
                args.bank_physics_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            )
            if old_report["report_hash"] != old[arm]["report_hash"] or any(
                current[k] != historical[k]
                for current, historical in ((parent, old_parent), (greedy, old_report))
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError("expanded training changed frozen warm/parent physical traces")
            sampled = []
            for s in range(args.samples_per_course):
                j = i * args.samples_per_course + s
                raw, result = _run(
                    **common,
                    arm=f"sample-{s}",
                    kind="actor",
                    motor_step=args.output_root / "models" / f"sample-{j}.json",
                    parent_report_override=parent_path,
                )
                sampled.append(
                    dict(
                        sample=s,
                        view_hash=views[j]["model_hash"],
                        report_hash=raw["report_hash"],
                        high_quality=high_quality(result),
                        **result,
                    )
                )
                print(
                    f"FAILURE_EXPLORED course={i} sample={s} HQ={high_quality(result)}", flush=True
                )
            record = dict(
                index=i,
                seed=seed,
                lane=lane,
                parent_report_hash=parent["report_hash"],
                greedy=dict(
                    report_hash=greedy["report_hash"], high_quality=high_quality(outcome), **outcome
                ),
                samples=sampled,
            )
            write_once(args.output_root / f"row-{i}.json", record)
            records.append(record)
        return records

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(
            [r for group in pool.map(worker, range(4)) for r in group], key=lambda r: r["index"]
        )
    if _head(source) != commitment["source_commit"]:
        raise ValueError("source drift invalidates exploration")
    result = dict(
        schema="soccer.rsi.smooth_memory_failure_exploration.v1"
        if args.behavior_kind == "smooth-memory"
        else "soccer.rsi.output_memory_failure_exploration.v1"
        if arm == "candidate"
        else "soccer.rsi.failed_step_course_exploration.v1",
        commitment=commitment,
        rows=rows,
        independent_contexts=len(courses),
        exploration_executions=len(views),
        physical_executions=len(courses) * (args.samples_per_course + 2),
        high_quality_samples=sum(s["high_quality"] for r in rows for s in r["samples"]),
        qualification="TRAIN_CONSUMED_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "training_summary.json", result)
    print(
        json.dumps({k: v for k, v in result.items() if k not in ("rows", "commitment")}), flush=True
    )


if __name__ == "__main__":
    main()
