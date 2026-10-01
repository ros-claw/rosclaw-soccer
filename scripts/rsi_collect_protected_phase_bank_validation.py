"""Paired full consumed-bank physics for protected per-frame motor learning.

Every parent is reproduced on the pinned source; every warm/candidate action is
audited. All results, including losses, are retained. This is not a fresh exam.
"""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.compiled_step_inference import make_preview
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.protected_phase_step_network import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def review(root: Path, bank_path: Path) -> dict[str, Any]:
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
    ):
        raise ValueError("complete consumed comparison required")
    rows = []
    for i, (course, recorded) in enumerate(zip(bank["courses"], summary["rows"], strict=True)):
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
        rows.append(measured)
    result = score(rows)
    if any(result[k] != summary[k] for k in result):
        raise ValueError("full bank scores do not reconstruct")
    review_report = dict(
        schema="soccer.rsi.protected_phase_bank_review.v1",
        source_summary_hash=summary["report_hash"],
        physical_reports_reviewed=156,
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
    args = parser.parse_args()
    if args.review_only:
        result = review(args.output_root, args.bank_path)
        write_once(args.output_root / "independent_review.json", result)
        print(json.dumps(result), flush=True)
        return
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    warm = json.loads(args.warm_model.read_text())
    candidate = json.loads(args.candidate_model.read_text())
    make_preview(warm)
    if candidate.get("schema") == "soccer.rsi.selective_phase_memory.v1":
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
    bank = _sealed(args.bank_path)
    if (
        bank["partition"] != "TRAIN_CONSUMED"
        or len(bank["courses"]) != 52
        or predecessor != warm["base_model"]
    ):
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
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)

    def preserve_failed_attempt(seed: int, lane: int, arm: str, kind: str) -> None:
        stem = f"seed{seed}-lane{lane}-{arm}-{kind}"
        if (args.output_root / "logs" / f"{stem}.log").exists() and not (
            args.output_root / stem / "report.json"
        ).is_file():
            raise ValueError(
                "failed attempt log requires explicit archived recovery, not overwrite"
            )

    def worker(gpu: int) -> list[dict[str, Any]]:
        records = []
        for i in range(gpu, 52, 4):
            course = bank["courses"][i]
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
                resume=args.resume,
                execution_timeout_s=600,
            )
            preserve_failed_attempt(seed, lane, "reproduction", "parent")
            parent, _ = _run(**common, arm="reproduction", kind="parent")
            old = _sealed(Path(course["parent_report"]))
            if any(
                parent[k] != old[k]
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError("pinned source changed full-bank physical parent")
            parent_path = (
                args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            )
            row = dict(index=i, seed=seed, lane=lane, parent_report_hash=parent["report_hash"])
            for arm, model_path in (("warm", args.warm_model), ("candidate", args.candidate_model)):
                preserve_failed_attempt(seed, lane, arm, "actor")
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

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(
            [r for group in pool.map(worker, range(4)) for r in group], key=lambda r: r["index"]
        )
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
