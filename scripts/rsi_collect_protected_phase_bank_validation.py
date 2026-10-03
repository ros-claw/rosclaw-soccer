"""Paired full consumed-bank physics for protected per-frame motor learning.

Every parent is reproduced on the pinned source; every warm/candidate action is
audited. All results, including losses, are retained. This is not a fresh exam.
"""

import argparse
import json
import shutil
import traceback
import uuid
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.compiled_step_inference import make_preview
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from rosclaw_soccer.rsi.protected_phase_step_network import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_sim_execution_pool import execute_shards
from scripts.rsi_train_protected_online_motor_v308 import _head


def review_course(job: dict[str, Any]) -> dict[str, Any]:
    """Independently reconstruct one declared row; no outcome-label shortcut."""
    root = Path(job["root"])
    i, course, recorded, commitment = (
        job["index"],
        job["course"],
        job["recorded"],
        job["commitment"],
    )
    seed, lane = course["seed"], course["lane"]
    if recorded["index"] != i or (recorded["seed"], recorded["lane"]) != (seed, lane):
        raise ValueError("bank row identity changed")
    parent = _sealed(root / f"seed{seed}-lane{lane}-reproduction-parent/report.json")
    old = _sealed(Path(course["parent_report"]))
    if (
        parent["report_hash"] != recorded["parent_report_hash"]
        or parent["source_hash"] != commitment["runner_hash"]
        or any(
            parent[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        )
    ):
        raise ValueError("bank parent reproduction changed")
    from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach

    audit_lateral_approach(root / f"seed{seed}-lane{lane}-reproduction-parent")
    measured = dict(index=i, seed=seed, lane=lane, parent_report_hash=parent["report_hash"])
    for arm, model_key in (("warm", "warm_model_hash"), ("candidate", "model_hash")):
        folder = root / f"seed{seed}-lane{lane}-{arm}-actor"
        raw = _sealed(folder / "report.json")
        outcome = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
        if (
            raw["parent_report_hash"] != parent["report_hash"]
            or raw["source_hash"] != commitment["runner_hash"]
            or raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
            != commitment[model_key]
            or raw["environments"][0]["course"] != parent["environments"][0]["course"]
        ):
            raise ValueError("consumed candidate lost policy, source or physical binding")
        measured[arm] = dict(report_hash=raw["report_hash"], **outcome)
    if measured != recorded:
        raise ValueError("bank comparison differs from independently audited physics")
    print(f"INDEPENDENT_BANK_ROW_RECONSTRUCTED index={i} seed={seed} lane={lane}", flush=True)
    return measured


def review(root: Path, bank_path: Path, *, workers: int = 1) -> dict[str, Any]:
    from scripts.rsi_audit_memory_learning_rollouts import ordered_audits

    summary = _sealed(root / "validation_summary.json")
    bank = _sealed(bank_path)
    commitment = summary["commitment"]
    if (
        summary["schema"] != "soccer.rsi.protected_phase_bank_physics.v1"
        or commitment != json.loads((root / "commitment.json").read_text())
        or commitment["learning_bank_hash"] != bank["report_hash"]
        or commitment["partition"] != "TRAIN_CONSUMED"
        or summary["physical_executions"] != 156
        or summary["independent_contexts"] != 52
        or summary["promotion_authorized"] is not False
        or summary["hardware_authorized"] is not False
        or len(summary["rows"]) != 52
        or len(bank["courses"]) != 52
        or len({(c["seed"], c["lane"]) for c in bank["courses"]}) != 52
    ):
        raise ValueError("complete consumed comparison required")
    reused = "reused_baseline_summary_hash" in commitment
    if summary.get("new_physical_executions", 156) != (52 if reused else 156) or summary.get(
        "reused_baseline_reports", 0
    ) != (104 if reused else 0):
        raise ValueError("reused controls cannot be counted as newly executed physics")
    if reused:
        old = _sealed(Path(commitment["baseline_reuse_root"]) / "validation_summary.json")
        verify_baseline_reuse(old, bank, commitment)
        if old["report_hash"] != commitment["reused_baseline_summary_hash"] or any(
            (before["parent_report_hash"], before["warm"]["report_hash"])
            != (after["parent_report_hash"], after["warm"]["report_hash"])
            for before, after in zip(old["rows"], summary["rows"], strict=True)
        ):
            raise ValueError("reused control lineage changed")
    jobs = [
        dict(root=str(root), index=i, course=course, recorded=recorded, commitment=commitment)
        for i, (course, recorded) in enumerate(zip(bank["courses"], summary["rows"], strict=True))
    ]
    rows = list(ordered_audits(review_course, jobs, workers))
    if (
        _sealed(root / "validation_summary.json")["report_hash"] != summary["report_hash"]
        or _sealed(bank_path)["report_hash"] != bank["report_hash"]
        or json.loads((root / "commitment.json").read_text()) != commitment
    ):
        raise ValueError("complete bank commitment changed during independent reconstruction")
    result = score(rows)
    if any(result[k] != summary[k] for k in result):
        raise ValueError("full bank scores do not reconstruct")
    review_report = dict(
        schema="soccer.rsi.protected_phase_bank_review.v1",
        source_summary_hash=summary["report_hash"],
        physical_reports_reviewed=156,
        new_physical_executions=summary.get("new_physical_executions", 156),
        reused_baseline_reports=summary.get("reused_baseline_reports", 0),
        motor_frames_reconstructed=31200,
        **result,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    review_report["report_hash"] = hash_json(review_report)
    return review_report


def score(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return dict(
        warm_high_quality=sum(r["warm"]["high_quality"] for r in rows),
        candidate_high_quality=sum(r["candidate"]["high_quality"] for r in rows),
        old_high_quality_loss=sum(
            r["warm"]["high_quality"] and not r["candidate"]["high_quality"] for r in rows
        ),
        old_clean_foot_loss=sum(
            r["warm"]["clean_foot_only"] and not r["candidate"]["clean_foot_only"] for r in rows
        ),
        safe_pelvis=all(r["candidate"]["minimum_pelvis_z_m"] >= 0.65 for r in rows),
        new_out_of_play=sum(
            r["warm"]["maximum_lateral_excursion_m"]
            <= 4
            < r["candidate"]["maximum_lateral_excursion_m"]
            for r in rows
        ),
    )


def verify_baseline_reuse(
    previous: dict[str, Any], bank: dict[str, Any], commitment: dict[str, Any]
) -> None:
    """Only complete controls under identical physical bindings are reusable."""
    if (
        previous.get("schema") != "soccer.rsi.protected_phase_bank_physics.v1"
        or previous.get("physical_executions") != 156
        or previous.get("independent_contexts") != 52
        or len(previous.get("rows", [])) != 52
        or any(
            previous.get(k) is not False for k in ("promotion_authorized", "hardware_authorized")
        )
        or any(
            previous["commitment"].get(k) != commitment[k]
            for k in (
                "runner_hash",
                "asset_hash",
                "core_commit",
                "learning_bank_hash",
                "warm_model_hash",
                "partition",
            )
        )
        or [(r["index"], r["seed"], r["lane"]) for r in previous["rows"]]
        != [(i, c["seed"], c["lane"]) for i, c in enumerate(bank["courses"])]
    ):
        raise ValueError(
            "baseline reuse requires complete identical physical bindings and course identities"
        )


def validate_bank_models(candidate: dict[str, Any], warm: dict[str, Any]) -> None:
    """Later learned actors must retain their current parent, not just first warm start."""
    if candidate.get("schema") == "soccer.rsi.advantage_memory_motor.v1":
        from rosclaw_soccer.rsi.advantage_memory_motor import validate_model as advantage_validate

        advantage_validate(candidate)
        if candidate["initial_actor"]["baseline"]["base_model"]["frozen_parent"] != warm:
            raise ValueError("advantage-memory proposal requires exact qualified NN parent")
        return
    if candidate.get("schema") == "soccer.rsi.current_memory_guarded_motor.v1":
        from rosclaw_soccer.rsi.current_memory_motor import validate_model as current_validate

        current_validate(candidate)
        if candidate["baseline"]["base_model"]["frozen_parent"] != warm:
            raise ValueError("current-memory proposal requires exact qualified NN parent")
        return
    if candidate.get("schema") == "soccer.rsi.consolidated_smooth_motor.v1":
        from rosclaw_soccer.rsi.consolidated_smooth_motor import (
            validate_model as consolidated_validate,
        )

        consolidated_validate(candidate)
        if candidate["base_model"]["frozen_parent"] != warm:
            raise ValueError(
                "consolidation requires exact qualified neural parent, not rejected AR"
            )
        return
    if candidate.get("schema") == "soccer.rsi.smooth_memory_motor.v1":
        from rosclaw_soccer.rsi.smooth_memory_motor import validate_model as smooth_validate

        smooth_validate(candidate)
        if warm.get("schema") == "soccer.rsi.output_memory_step_motor.v1":
            from rosclaw_soccer.rsi.output_memory_step_motor import (
                validate_model as output_validate,
            )

            output_validate(warm)
            aligned = candidate["frozen_parent"] == warm
        elif warm.get("schema") == "soccer.rsi.smooth_memory_motor.v1":
            smooth_validate(warm)
            aligned = (
                candidate["generation"] == warm["generation"] + 1
                and candidate["learning_receipt"]["learner_parent_hash"] == warm["model_hash"]
                and candidate["frozen_parent"] == warm["frozen_parent"]
            )
        else:
            aligned = False
        if not aligned:
            raise ValueError("smooth actor requires its exact current learned parent")
        return
    if candidate.get("schema") == "soccer.rsi.output_memory_step_motor.v1":
        from rosclaw_soccer.rsi.output_memory_step_motor import validate_model as output_validate

        output_validate(candidate)
        if warm.get("schema") == "soccer.rsi.kernel_guarded_step_actor_critic.v1":
            from rosclaw_soccer.rsi.kernel_guarded_step_network import (
                validate_model as kernel_check,
            )

            kernel_check(warm)
            aligned = candidate["frozen_parent"] == warm
        elif warm.get("schema") == "soccer.rsi.output_memory_step_motor.v1":
            output_validate(warm)
            aligned = (
                candidate["generation"] == warm["generation"] + 1
                and candidate["learning_receipt"]["learner_parent_hash"] == warm["model_hash"]
                and candidate["frozen_parent"] == warm["frozen_parent"]
                and candidate["output_memory"] == warm["output_memory"]
            )
        else:
            aligned = False
        if not aligned:
            raise ValueError("memory actor requires its exact current learned parent")
        return
    make_preview(warm)
    if candidate.get("schema") == "soccer.rsi.kernel_replay_motor.v1":
        from rosclaw_soccer.rsi.kernel_replay_motor import validate_model as replay_validate

        replay_validate(candidate)
        predecessor = candidate["frozen_parent"]["encoder"]["base_model"]
    elif candidate.get("schema") == "soccer.rsi.selective_phase_memory.v1":
        from rosclaw_soccer.rsi.selective_phase_memory import validate_model as selective_validate

        selective_validate(candidate)
        predecessor = candidate["transfer_model"]["phase_model"]["base_model"]
    elif candidate.get("schema") == "soccer.rsi.memory_guarded_phase_transfer.v1":
        from rosclaw_soccer.rsi.memory_guarded_phase_transfer import (
            validate_model as memory_validate,
        )

        memory_validate(candidate)
        predecessor = candidate["phase_model"]["base_model"]
    elif candidate.get("schema") == "soccer.rsi.kernel_guarded_step_actor_critic.v1":
        from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model as kernel_validate

        kernel_validate(candidate)
        predecessor = candidate["encoder"]["base_model"]
    else:
        validate_model(candidate)
        predecessor = candidate["base_model"]
    if predecessor != warm["base_model"]:
        raise ValueError("historical actor requires the same frozen compiled warm base")


def preserve_failed_attempt(root: Path, seed: int, lane: int, arm: str, kind: str) -> None:
    stem = f"seed{seed}-lane{lane}-{arm}-{kind}"
    if (root / "logs" / f"{stem}.log").exists():
        try:
            resolve_physical_report(root / stem / "report.json")
        except ValueError as exc:
            raise ValueError(
                "failed attempt log requires explicit archived recovery, not overwrite"
            ) from exc


def execute_bank_shard(job: dict[str, Any]) -> list[dict[str, Any]]:
    """Persist a shard exception before ordered pool collection can hide it.

    Other workers are not cancelled or restarted. A failure record grants no
    authority and does not convert partial artifacts into a passing bank.
    """
    try:
        return _execute_bank_shard(job)
    except Exception as exc:
        root, gpu = job["args"].output_root, job["gpu"]
        failure = dict(
            schema="soccer.rsi.bank_shard_failure.v1",
            gpu=gpu,
            exception_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
            exception_message=str(exc),
            traceback=traceback.format_exc(),
            completed_row_indices=[
                i for i in range(gpu, 52, 4) if (root / f"row-{i}.json").is_file()
            ],
            completed_paths_are_not_independent_verification=True,
            automatic_retry=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        failure["report_hash"] = hash_json(failure)
        path = root / f"shard-{gpu}-failure-{uuid.uuid4().hex}.json"
        try:
            write_once(path, failure)
            print(f"BANK_SHARD_FAILED gpu={gpu} evidence={path}", flush=True)
        except Exception as journal_error:
            exc.add_note(f"failure journal could not be written: {journal_error!r}")
            print(f"BANK_SHARD_FAILURE_JOURNAL_FAILED gpu={gpu}: {journal_error!r}", flush=True)
        raise


def _execute_bank_shard(job: dict[str, Any]) -> list[dict[str, Any]]:
    args, runner, gpu, courses = job["args"], job["runner"], job["gpu"], job["courses"]
    records = []
    for i in range(gpu, 52, 4):
        course = courses[i]
        seed, lane = course["seed"], course["lane"]
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
            resume=args.resume or job["reuse"],
            execution_timeout_s=600,
            compressed_report=args.compressed_reports,
            shared_model_report=args.shared_model_reports,
        )
        preserve_failed_attempt(args.output_root, seed, lane, "reproduction", "parent")
        parent, _ = _run(**common, arm="reproduction", kind="parent")
        old = _sealed(Path(course["parent_report"]))
        if any(
            parent[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("pinned source changed full-bank physical parent")
        parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        row = dict(index=i, seed=seed, lane=lane, parent_report_hash=parent["report_hash"])
        for arm, model_path in (("warm", args.warm_model), ("candidate", args.candidate_model)):
            preserve_failed_attempt(args.output_root, seed, lane, arm, "actor")
            raw, outcome = _run(
                **common,
                arm=arm,
                kind="actor",
                motor_step=model_path,
                parent_report_override=parent_path,
            )
            outcome["high_quality"] = high_quality(outcome)
            row[arm] = dict(report_hash=raw["report_hash"], **outcome)
        records.append(row)
        write_once(args.output_root / f"row-{i}.json", row)
        print(
            f"PROTECTED_BANK_EXECUTED i={i} warm={row['warm']['high_quality']} "
            f"candidate={row['candidate']['high_quality']}",
            flush=True,
        )
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "bank-path",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "warm-model",
        "candidate-model",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--review-only", action="store_true")
    parser.add_argument("--review-workers", type=int, choices=range(1, 5), default=1)
    parser.add_argument("--reuse-baseline-root", type=Path)
    parser.add_argument("--compressed-reports", action="store_true")
    parser.add_argument("--shared-model-reports", action="store_true")
    parser.add_argument("--spawn-execution-workers", action="store_true")
    args = parser.parse_args()
    if args.shared_model_reports and not args.compressed_reports:
        parser.error("shared proofs require explicit compressed reports")
    if args.review_only:
        result = review(args.output_root, args.bank_path, workers=args.review_workers)
        write_once(args.output_root / "independent_review.json", result)
        print(json.dumps(result), flush=True)
        return
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact

    warm = load_json_artifact(args.warm_model)
    candidate = load_json_artifact(args.candidate_model)
    validate_bank_models(candidate, warm)
    bank = _sealed(args.bank_path)
    if bank["partition"] != "TRAIN_CONSUMED" or len(bank["courses"]) != 52:
        parser.error("frozen aligned warm actor and complete consumed bank required")
    for course in bank["courses"]:
        _sealed(Path(course["parent_report"]))
    commitment = dict(
        schema="soccer.rsi.protected_phase_bank_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        learning_bank_hash=bank["report_hash"],
        model_hash=candidate["model_hash"],
        warm_model_hash=warm["model_hash"],
        partition="TRAIN_CONSUMED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    reuse = None
    if args.spawn_execution_workers:
        commitment["execution_worker_model"] = "FOUR_SPAWNED_GPU_SHARDS"
        commitment["execution_worker_source_hash"] = hash_bytes(Path(__file__).read_bytes())
        commitment["execution_pool_source_hash"] = hash_bytes(
            (Path(__file__).parent / "rsi_sim_execution_pool.py").read_bytes()
        )
    if args.compressed_reports:
        commitment["physical_report_representation"] = (
            "lossless_shared_model_gzip_json" if args.shared_model_reports else "lossless_gzip_json"
        )
    if args.reuse_baseline_root is not None:
        reuse = _sealed(args.reuse_baseline_root / "validation_summary.json")
        verify_baseline_reuse(reuse, bank, commitment)
        commitment["reused_baseline_summary_hash"] = reuse["report_hash"]
        commitment["baseline_reuse_root"] = str(args.reuse_baseline_root.resolve())
        commitment["baseline_reuse_is_new_physics"] = False
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)
    if reuse is not None:
        for row in reuse["rows"]:
            seed, lane = row["seed"], row["lane"]
            for arm, kind in (("reproduction", "parent"), ("warm", "actor")):
                stem = f"seed{seed}-lane{lane}-{arm}-{kind}"
                origin = args.reuse_baseline_root / stem
                if origin.resolve().parent != args.reuse_baseline_root.resolve() or any(
                    p.is_symlink() for p in origin.rglob("*")
                ):
                    raise ValueError(
                        "baseline evidence must be a local non-symlink physical directory"
                    )
                raw = _sealed(origin / "report.json")
                expected = (
                    row["parent_report_hash"]
                    if arm == "reproduction"
                    else row["warm"]["report_hash"]
                )
                if (
                    raw["report_hash"] != expected
                    or raw["source_hash"] != commitment["runner_hash"]
                    or raw["asset_hash"] != commitment["asset_hash"]
                    or raw["trace_hash"] != hash_bytes((origin / "trace.npz").read_bytes())
                    or raw["body_trace_hash"]
                    != hash_bytes((origin / "body_trace.npz").read_bytes())
                ):
                    raise ValueError("reused control evidence changed")
                destination = args.output_root / stem
                if not destination.exists():
                    shutil.copytree(origin, destination)
                    shutil.copy2(
                        args.reuse_baseline_root / "logs" / f"{stem}.log",
                        args.output_root / "logs" / f"{stem}.log",
                    )

    jobs = [
        dict(args=args, runner=runner, gpu=gpu, courses=bank["courses"], reuse=reuse is not None)
        for gpu in range(4)
    ]
    rows = sorted(
        [
            r
            for group in execute_shards(
                execute_bank_shard, jobs, spawn=args.spawn_execution_workers
            )
            for r in group
        ],
        key=lambda r: r["index"],
    )
    if args.spawn_execution_workers and (
        commitment["execution_worker_source_hash"] != hash_bytes(Path(__file__).read_bytes())
        or commitment["execution_pool_source_hash"]
        != hash_bytes((Path(__file__).parent / "rsi_sim_execution_pool.py").read_bytes())
    ):
        raise ValueError("spawned execution orchestration source drift")
    if (
        _head(source) != commitment["source_commit"]
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
    ):
        raise ValueError("full-bank source drift")
    summary = dict(
        schema="soccer.rsi.protected_phase_bank_physics.v1",
        commitment=commitment,
        rows=rows,
        physical_executions=156,
        new_physical_executions=52 if reuse is not None else 156,
        reused_baseline_reports=104 if reuse is not None else 0,
        independent_contexts=52,
        **score(rows),
        qualification="CONSUMED_ONLY_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "validation_summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("rows", "commitment")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
