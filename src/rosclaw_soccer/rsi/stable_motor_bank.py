"""Recalibration replay from independently reviewed online physical episodes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.protected_online_evidence import review_online
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_train_protected_online_motor_v308 import _normalize


def stable_replay(
    root: Path, bank_path: Path, preview_root: Path, validation_root: Path
) -> dict[str, Any]:
    review = review_online(root, bank_path, preview_root, validation_root)
    if not review["training_complete"]:
        raise ValueError("finished independently reconstructed online physics required")
    bank = _sealed(bank_path)
    samples = [
        dict(
            observation=s["observation"],
            normalized_action=_normalize(s["motor_parameters"]),
            outcome=s["learning_labels"],
            report_hash=s["audit_metadata"]["source_report_hash"],
        )
        for s in bank["critic_samples"]
    ]

    def episode(index: int, arm: str, outcome: dict[str, Any]) -> None:
        seed, lane = COURSES[index]
        report = _sealed(root / f"seed{seed}-lane{lane}-{arm}-actor/report.json")
        policy = report["contact_motor_policy"]
        proof = policy.get("online_motor_proof", policy.get("bootstrap_proof"))
        samples.append(
            dict(
                observation=proof["context"],
                normalized_action=_normalize(
                    [v for knot in policy["knots_rad"] for v in knot] + [policy["phase_gap_end_m"]]
                ),
                outcome=outcome,
                report_hash=report["report_hash"],
            )
        )

    reproduced = json.loads((root / "reproduction.json").read_text())
    for row in reproduced["rows"]:
        episode(row["course_index"], "neural-reproduction", row["outcome"])
    for generation in review["generations"]:
        g = generation["generation"]
        feedback = _sealed(root / f"feedback-g{g}.json")
        samples.extend(feedback["samples"])
        for row in generation["rows"]:
            episode(row["course_index"], f"g{g}-neural", row)
    result = dict(
        schema="soccer.rsi.stable_replay_bank.v309",
        samples=samples,
        record_count=len(samples),
        independent_situation_count=12,
        prior_review_hash=review["report_hash"],
        prior_best_model_hash=review["best_model_hash"],
        body_hash=review["body_hash"],
        partition="TRAIN_CONSUMED",
        runtime_selection_authorized=False,
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
