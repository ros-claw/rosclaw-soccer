"""Physical counterfactual of ONE frozen neural proposal across 12 consumed courses.

Reproduce the final v306 champion first; model sees measured proprioception,
never course identity, donor IDs or future outcomes. No training in validation.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import (
    _sealed,
    retention_score,
    review_curriculum,
)
from rosclaw_soccer.rsi.motor_bootstrap_network import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "training-root",
        "validation-root",
        "neural-model",
        "learning-bank",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    runner = root / "scripts/rsi_isaac_vector_first_touch.py"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        head != args.source_commit
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        or (args.output_root.exists() and not args.resume)
    ):
        parser.error("frozen clean source and fresh output or explicit resume required")
    training = _sealed(args.training_root / "training_summary.json")
    bank = _sealed(args.learning_bank)
    prior = _sealed(args.validation_root / "validation_summary.json")
    model = json.loads(args.neural_model.read_text(encoding="utf-8"))
    protocol = json.loads(
        (root / "docs/rsi/protocols/neural-motor-preview-v307.json").read_text(encoding="utf-8")
    )
    quarantine = json.loads(
        (root / "docs/rsi/protocols/neural-motor-fresh-quarantine-v307.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        protocol["courses"] != [list(c) for c in COURSES]
        or protocol["physical_episodes"] != 36
        or model.get("epochs") != protocol["fit_epochs"]
        or model.get("fresh_quarantine_protocol_hash") != hash_json(quarantine)
    ):
        raise ValueError("preregistered neural preview protocol drift")
    validate_model(model)
    review = review_curriculum(args.training_root, args.validation_root)
    if (
        model["bank_hash"] != bank["report_hash"]
        or bank["sample_count"] != 384
        or bank["distinct_consumed_course_count"] != 12
        or bank["independent_review_hash"] != review["report_hash"]
        or not review["training_complete"]
    ):
        raise ValueError("neural model does not bind the completed physical learning bank")
    old_policy = Path(training["best"]["policy"])
    old_arm = f"g{training['best']['generation']}-c{training['best']['candidate']}"
    commitment = {
        "source_commit": head,
        "runner_hash": hash_bytes(runner.read_bytes()),
        "collector_hash": hash_bytes(Path(__file__).read_bytes()),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "training_report_hash": training["report_hash"],
        "bank_hash": bank["report_hash"],
        "neural_model_hash": model["model_hash"],
        "reference_validation_hash": prior["report_hash"],
        "protocol_hash": hash_json(protocol),
        "activation_ceiling": "SIM_ONLY",
        "partition": "TRAIN_CONSUMED",
    }
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)

    def shared_args(index: int, gpu: int) -> dict[str, Any]:
        seed, lane = COURSES[index]
        return dict(
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
            resume=args.resume,
        )

    def reproduce(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for index in range(gpu, 12, 4):
            seed, lane = COURSES[index]
            shared = shared_args(index, gpu)
            _run(**shared, arm="reproduction", kind="parent")
            parent = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            old, old_outcome = _run(
                **shared,
                arm="champion",
                kind="actor",
                motor_policy=old_policy,
                parent_report_override=parent,
            )
            historical = _sealed(
                args.training_root / f"seed{seed}-lane{lane}-{old_arm}-actor/report.json"
            )
            if any(
                old[k] != historical[k]
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError(
                    "source change failed champion physics reproduction; no neural evaluation"
                )
            rows.append({"course_index": index, "report": old, "outcome": old_outcome})
            print(f"NEURAL_CHAMPION_REPRODUCED course={index}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        reproduced = {
            row["course_index"]: row for batch in pool.map(reproduce, range(4)) for row in batch
        }
    if len(reproduced) != 12:
        raise ValueError("all 12 champion courses must reproduce BEFORE neural evaluation")
    write_once(
        args.output_root / "reproduction.json",
        {"matched_course_count": 12, "rows": [reproduced[index] for index in range(12)]},
    )

    def worker(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for index in range(gpu, 12, 4):
            seed, lane = COURSES[index]
            shared = shared_args(index, gpu)
            parent = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            old, old_outcome = reproduced[index]["report"], reproduced[index]["outcome"]
            neural, outcome = _run(
                **shared,
                arm="neural",
                kind="actor",
                motor_bootstrap=args.neural_model,
                parent_report_override=parent,
            )
            context = neural["contact_motor_policy"]["bootstrap_proof"]["context"]
            expected = [
                s["observation"]
                for s in bank["critic_samples"]
                if s["audit_metadata"]["course"] == [seed, lane]
            ]
            if len(expected) != 32 or any(features != context for features in expected):
                raise ValueError("neural observation prefix differs from physical training context")
            rows.append(
                {
                    "course_index": index,
                    "seed": seed,
                    "lane": lane,
                    "champion": {
                        **old_outcome,
                        "high_quality": high_quality(old_outcome),
                        "report_hash": old["report_hash"],
                    },
                    "neural": {
                        **outcome,
                        "high_quality": high_quality(outcome),
                        "report_hash": neural["report_hash"],
                    },
                }
            )
            print(f"NEURAL_PREVIEW_AUDITED course={index}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(
            (row for batch in pool.map(worker, range(4)) for row in batch),
            key=lambda r: r["course_index"],
        )
    if (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
    ):
        raise ValueError("neural experiment source drift")
    by_course = {(r["seed"], r["lane"]): r for r in prior["rows"]}
    result = retention_score([r["neural"] for r in rows], [by_course[c] for c in COURSES])
    champion_loss = sum(
        r["champion"]["high_quality"] and not r["neural"]["high_quality"] for r in rows
    )
    summary = {
        "schema": "soccer.rsi.neural_motor_consumed_preview.v307",
        "activation_ceiling": "SIM_ONLY",
        "partition": "TRAIN_CONSUMED",
        "commitment": commitment,
        "rows": rows,
        "score": result,
        "champion_high_quality_count": sum(r["champion"]["high_quality"] for r in rows),
        "champion_high_quality_loss": champion_loss,
        "independent_physical_episode_count": 36,
        "distinct_consumed_course_count": 12,
        "consumed_neural_gate_passed": result["consumed_training_gate_passed"]
        and champion_loss == 0,
        "promotion_authorized": False,
        "fresh_holdout_open_authorized": False,
        "not_claimed": [
            "online RL",
            "fresh generalization",
            "continuous team qualification",
            "hardware authority",
        ],
    }
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "preview_summary.json", summary)
    print(
        json.dumps(
            {"score": result, "champion_loss": champion_loss, "report_hash": summary["report_hash"]}
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
