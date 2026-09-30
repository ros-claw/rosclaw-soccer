"""Read-only reconstruction of the one-time neural fresh physics exam."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed
from rosclaw_soccer.rsi.neural_fresh_statistics import paired_cluster_score
from rosclaw_soccer.sim.contracts import hash_json


def review_fresh(root: Path, quarantine_path: Path, pool_ledger: Path) -> dict[str, Any]:
    commitment = json.loads((root / "commitment.json").read_text())
    quarantine = json.loads(quarantine_path.read_text())
    summary = _sealed(root / "exam_summary.json")
    ledger = json.loads((pool_ledger / f"{hash_json(quarantine)[7:]}.json").read_text())
    if (
        commitment["quarantine_hash"] != hash_json(quarantine)
        or summary["commitment"] != commitment
        or ledger.get("commitment") != commitment
        or ledger.get("partition_after_allocation") != "CONSUMED_NO_REUSE_AS_FRESH"
        or summary.get("partition_after_observation") != "CONSUMED_NO_REUSE_AS_FRESH"
        or summary.get("promotion_authorized") is not False
        or summary.get("hardware_authorized") is not False
        or summary.get("physical_episode_count") != 120
    ):
        raise ValueError("fresh exam usage, provenance or authority drift")
    entry = _sealed(root / "selected_consumed_review.json")
    if (
        entry["report_hash"] != commitment["selected_consumed_review_hash"]
        or entry.get("consumed_gate_passed") is not True
        or entry.get("best_model_hash") != commitment["selected_model_hash"]
    ):
        raise ValueError("fresh candidate did not pass its committed consumed entry review")
    rows = []
    for seed in quarantine["seeds"]:
        for lane in quarantine["lanes_per_seed"]:
            parent = _sealed(root / f"seed{seed}-lane{lane}-reproduction-parent/report.json")
            arms = {}
            for arm, label in (("champion", "parent"), ("candidate", "candidate")):
                folder = root / f"seed{seed}-lane{lane}-{arm}-actor"
                report = _sealed(folder / "report.json")
                if (
                    report["parent_report_hash"] != parent["report_hash"]
                    or parent["source_hash"] != commitment["runner_hash"]
                    or parent["asset_hash"] != commitment["asset_hash"]
                ):
                    raise ValueError("fresh action detached from frozen physical parent")
                if (
                    label == "parent"
                    and report["contact_motor_policy_hash"] != commitment["parent_policy_hash"]
                ):
                    raise ValueError("fresh parent motor policy changed")
                if (
                    label == "candidate"
                    and report["contact_motor_policy"]["online_motor_proof"]["model"]["model_hash"]
                    != commitment["selected_model_hash"]
                ):
                    raise ValueError("fresh neural candidate changed after allocation")
                result = _outcome(folder, report["contact_motor_policy_hash"], commitment)
                arms[label] = {**result["outcome"], "report_hash": report["report_hash"]}
            rows.append(dict(seed=seed, lane=lane, **arms))
    # Reject missing/duplicated summary rows instead of silently truncating zip.
    paired_cluster_score(summary["rows"], quarantine)
    for raw, cached in zip(rows, summary["rows"], strict=True):
        if (
            raw["seed"] != cached["seed"]
            or raw["lane"] != cached["lane"]
            or any(
                cached[arm][k] != raw[arm][k]
                for arm in ("parent", "candidate")
                for k in cached[arm]
            )
        ):
            raise ValueError("fresh outcome ledger differs from measured action and physics")
    score = paired_cluster_score(rows, quarantine)
    qualification = (
        "FRESH_EFFECT_AND_SAFETY_ONLY"
        if score["fresh_effect_gate_passed"] and score["fresh_safety_gate_passed"]
        else "REJECTED"
    )
    if summary["score"] != score or summary["qualification"] != qualification:
        raise ValueError("fresh statistical or physical gate does not reconstruct")
    result = dict(
        schema="soccer.rsi.independent_neural_fresh_review.v310",
        source_summary_hash=summary["report_hash"],
        score=score,
        qualification=qualification,
        audited_physical_episode_count=120,
        distinct_physical_course_count=40,
        independent_seed_cluster_count=10,
        pool_state="CONSUMED_NO_REUSE_AS_FRESH",
        candidate_model_hash=commitment["selected_model_hash"],
        promotion_authorized=False,
        hardware_authorized=False,
        missing_promotion_evidence=["cpu_mujoco", "continuous_chain"],
    )
    result["report_hash"] = hash_json(result)
    return result
