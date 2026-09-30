"""Broader consumed-state replay after a completed, permanently consumed exam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.bootstrap_motor_execution import context_at30
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.neural_fresh_evidence import review_fresh
from rosclaw_soccer.rsi.stable_motor_bank import stable_replay
from rosclaw_soccer.rsi.stable_motor_evidence import review_stable
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_train_protected_online_motor_v308 import _normalize


def build_bank(
    stable_root: Path,
    online_root: Path,
    bank_path: Path,
    preview_root: Path,
    validation_root: Path,
    fresh_root: Path,
    old_quarantine: Path,
    pool_ledger: Path,
) -> dict[str, Any]:
    stable_review = review_stable(
        stable_root, online_root, bank_path, preview_root, validation_root
    )
    fresh_review = review_fresh(fresh_root, old_quarantine, pool_ledger)
    if fresh_review["pool_state"] != "CONSUMED_NO_REUSE_AS_FRESH":
        raise ValueError("unopened fresh data cannot become training replay")
    replay = stable_replay(online_root, bank_path, preview_root, validation_root)
    samples = list(replay["samples"])
    stable_commitment = json.loads((stable_root / "commitment.json").read_text())
    fresh_commitment = json.loads((fresh_root / "commitment.json").read_text())
    records, protected = [], []
    fresh_summary = _sealed(fresh_root / "exam_summary.json")
    courses = [(stable_root, seed, lane, "stable", stable_commitment) for seed, lane in COURSES]
    courses += [
        (fresh_root, r["seed"], r["lane"], "candidate", fresh_commitment)
        for r in fresh_summary["rows"]
    ]
    for root, seed, lane, neural_arm, commitment in courses:
        neural_folder = root / f"seed{seed}-lane{lane}-{neural_arm}-actor"
        report = _sealed(neural_folder / "report.json")
        measured = _outcome(neural_folder, report["contact_motor_policy_hash"], commitment)
        proof = report["contact_motor_policy"]["online_motor_proof"]
        features = proof["context"]
        if proof["model"]["model_hash"] != stable_review["best_model_hash"]:
            raise ValueError("consumed neural predecessor differs between source pools")
        with np.load(neural_folder / "body_trace.npz", allow_pickle=False) as body:
            if list(context_at30(body)) != features:
                raise ValueError("replay context differs from actual precontact body")
        if measured["outcome"]["high_quality"]:
            protected.append(features)
        arms = [neural_arm] + (["champion"] if root == fresh_root else [])
        for arm in arms:
            folder = root / f"seed{seed}-lane{lane}-{arm}-actor"
            raw = _sealed(folder / "report.json")
            outcome = _outcome(folder, raw["contact_motor_policy_hash"], commitment)["outcome"]
            with np.load(folder / "body_trace.npz", allow_pickle=False) as body:
                if list(context_at30(body)) != features:
                    raise ValueError("consumed physical arms diverged before frame-30 decision")
            policy = raw["contact_motor_policy"]
            samples.append(
                dict(
                    observation=features,
                    normalized_action=_normalize(
                        [v for knot in policy["knots_rad"] for v in knot]
                        + [policy["phase_gap_end_m"]]
                    ),
                    outcome=outcome,
                    report_hash=raw["report_hash"],
                )
            )
        records.append(
            dict(
                seed=seed,
                lane=lane,
                observation=features,
                predecessor_outcome=measured["outcome"],
                report_hash=report["report_hash"],
                predecessor_folder=str(neural_folder),
                parent_report=str(root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"),
            )
        )
    if len(records) != 52 or len(protected) != 38 or len(samples) != 584:
        raise ValueError("complete 52 consumed contexts, 38 anchors and 584 records required")
    result = dict(
        schema="soccer.rsi.progressive_motor_learning_bank.v312",
        partition="TRAIN_CONSUMED",
        samples=samples,
        courses=records,
        protected_features=protected,
        distinct_physical_context_count=52,
        record_count=len(samples),
        protected_observation_count=len(protected),
        stable_review_hash=stable_review["report_hash"],
        fresh_review_hash=fresh_review["report_hash"],
        predecessor_model_hash=stable_review["best_model_hash"],
        runtime_course_identity_input_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
