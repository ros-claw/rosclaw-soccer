"""One-time predeclared fresh physics exam, never a learning/tuning loop."""

from __future__ import annotations

import argparse
import json
import math
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.contact_motor_contract import load_policy
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.first_touch_course_catalog import sample_training_courses
from rosclaw_soccer.rsi.neural_fresh_statistics import paired_cluster_score
from rosclaw_soccer.rsi.online_motor_actor_critic import validate_model
from rosclaw_soccer.rsi.protected_online_evidence import review_online
from rosclaw_soccer.rsi.stable_motor_evidence import review_stable
from rosclaw_soccer.rsi.stable_motor_update import validate_stable_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "pool-ledger",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "learning-bank",
        "preview-root",
        "online-root",
        "stable-root",
        "validation-root",
        "training-root",
        "core-root",
        "parent-motor-policy",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--core-commit", required=True)
    parser.add_argument("--record-one-time-pool-use", action="store_true", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        _head(source) != args.source_commit
        or _head(args.core_root) != args.core_commit
        or (args.output_root.exists() and not args.resume)
    ):
        parser.error("frozen clean sources and fresh output or explicit audited resume required")
    quarantine = json.loads(
        (source / "docs/rsi/protocols/neural-motor-fresh-quarantine-v307.json").read_text()
    )
    protocol = json.loads(
        (source / "docs/rsi/protocols/neural-motor-fresh-execution-v310.json").read_text()
    )
    old = review_online(
        args.online_root, args.learning_bank, args.preview_root, args.validation_root
    )
    stable = review_stable(
        args.stable_root,
        args.online_root,
        args.learning_bank,
        args.preview_root,
        args.validation_root,
    )
    old_summary = _sealed(args.online_root / "training_summary.json")
    old_best = [
        g for g in old_summary["generations"] if g["model_hash"] == old_summary["best_model_hash"]
    ][-1]
    selected_review = (
        stable
        if stable["consumed_gate_passed"]
        and tuple(stable["generations"][0]["score"]["rank"]) > tuple(old_best["score"]["rank"])
        else old
    )
    candidate_path = (
        args.stable_root / "stable_model.json"
        if selected_review is stable
        else args.online_root / f"models/g{old_best['generation']}.json"
    )
    model = json.loads(candidate_path.read_text())
    validate_stable_model(model) if selected_review is stable else validate_model(model)
    parent, _ = load_policy(args.parent_motor_policy)
    training = _sealed(args.training_root / "training_summary.json")
    if (
        selected_review["consumed_gate_passed"] is not True
        or selected_review["best_model_hash"] != model["model_hash"]
        or model["fresh_quarantine_protocol_hash"] != hash_json(quarantine)
        or quarantine["partition"] != "FRESH_QUARANTINED_NOT_OPENED"
        or quarantine["data_generated"] is not False
        or quarantine["labels_observed"] is not False
        or parent["policy_hash"] != protocol["parent_policy_hash"]
        or training["report_hash"] != protocol["parent_training_report_hash"]
    ):
        raise ValueError("fresh exam entry, parent or pre-training quarantine commitment drift")
    courses = [
        (seed, lane) for seed in quarantine["seeds"] for lane in quarantine["lanes_per_seed"]
    ]
    # Preflight every declaration before creating a usage record or GPU worker.
    for seed, lane in courses:
        course = sample_training_courses(seed, 16)[lane]
        if len(course) != 3 or not all(math.isfinite(value) for value in course):
            raise ValueError("preregistered finite physical course required")
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    commitment = dict(
        source_commit=args.source_commit,
        core_commit=args.core_commit,
        runner_hash=hash_bytes(runner.read_bytes()),
        collector_hash=hash_bytes(Path(__file__).read_bytes()),
        protocol_hash=hash_json(protocol),
        quarantine_hash=hash_json(quarantine),
        selected_model_hash=model["model_hash"],
        selected_consumed_review_hash=selected_review["report_hash"],
        parent_policy_hash=parent["policy_hash"],
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        output_root=str(args.output_root.resolve()),
        activation_ceiling="SIM_ONLY",
        partition="FRESH_TEST_ONCE",
    )
    args.pool_ledger.mkdir(parents=True, exist_ok=True)
    ledger_path = args.pool_ledger / f"{hash_json(quarantine)[7:]}.json"
    if ledger_path.exists() and not args.resume:
        raise ValueError(
            "fresh pool already allocated; renamed outputs or a new candidate cannot reopen it"
        )
    write_once(
        ledger_path,
        dict(
            commitment=commitment,
            partition_after_allocation="CONSUMED_NO_REUSE_AS_FRESH",
            promotion_authorized=False,
        ),
    )
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)
    write_once(args.output_root / "selected_consumed_review.json", selected_review)

    def worker(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for i in range(gpu, len(courses), 4):
            seed, lane = courses[i]
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
            )
            _run(**common, arm="reproduction", kind="parent")
            parent_report = (
                args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            )
            parent_raw, parent_outcome = _run(
                **common,
                arm="champion",
                kind="actor",
                motor_policy=args.parent_motor_policy,
                parent_report_override=parent_report,
            )
            candidate_raw, candidate_outcome = _run(
                **common,
                arm="candidate",
                kind="actor",
                motor_online=candidate_path,
                parent_report_override=parent_report,
            )
            parent_outcome["high_quality"] = high_quality(parent_outcome)
            candidate_outcome["high_quality"] = high_quality(candidate_outcome)
            rows.append(
                dict(
                    seed=seed,
                    lane=lane,
                    parent={**parent_outcome, "report_hash": parent_raw["report_hash"]},
                    candidate={**candidate_outcome, "report_hash": candidate_raw["report_hash"]},
                )
            )
            print(
                f"FRESH_COURSE_AUDITED i={i} parent={parent_outcome['high_quality']} "
                f"candidate={candidate_outcome['high_quality']}",
                flush=True,
            )
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(
            [r for batch in pool.map(worker, range(4)) for r in batch],
            key=lambda r: (r["seed"], r["lane"]),
        )
    if (
        _head(source) != args.source_commit
        or _head(args.core_root) != args.core_commit
        or json.loads(candidate_path.read_text())["model_hash"] != model["model_hash"]
    ):
        raise ValueError("fresh exam source or frozen candidate changed")
    score = paired_cluster_score(rows, quarantine)
    result = dict(
        schema="soccer.rsi.neural_fresh_exam.v310",
        commitment=commitment,
        rows=rows,
        score=score,
        physical_episode_count=120,
        partition_after_observation="CONSUMED_NO_REUSE_AS_FRESH",
        qualification="FRESH_EFFECT_AND_SAFETY_ONLY"
        if score["fresh_effect_gate_passed"] and score["fresh_safety_gate_passed"]
        else "REJECTED",
        promotion_authorized=False,
        hardware_authorized=False,
        cpu_mujoco_status="NOT_RUN",
        continuous_chain_status="NOT_QUALIFIED",
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "exam_summary.json", result)
    print(f"FRESH_EXAM_COMPLETE score={score}", flush=True)


if __name__ == "__main__":
    main()
