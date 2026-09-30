"""Independent replay, neural weight reconstruction and raw physics for v309."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed, retention_score
from rosclaw_soccer.rsi.stable_motor_bank import stable_replay
from rosclaw_soccer.rsi.stable_motor_update import stable_update, validate_stable_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES


def review_stable(
    root: Path, online_root: Path, bank_path: Path, preview_root: Path, validation_root: Path
) -> dict[str, Any]:
    commitment = json.loads((root / "commitment.json").read_text())
    summary = _sealed(root / "preview_summary.json")
    prior = _sealed(online_root / "training_summary.json")
    replay = stable_replay(online_root, bank_path, preview_root, validation_root)
    best = [g for g in prior["generations"] if g["model_hash"] == prior["best_model_hash"]]
    parent_path = (
        online_root / f"models/g{best[-1]['generation']}.json"
        if best
        else online_root / "models/initial.json"
    )
    parent = json.loads(parent_path.read_text())
    candidate = stable_update(parent, replay["samples"], replay["report_hash"])
    validate_stable_model(candidate)
    if (
        commitment["replay_hash"] != replay["report_hash"]
        or commitment["candidate_model_hash"] != candidate["model_hash"]
        or commitment["prior_best_model_hash"] != parent["model_hash"]
        or _sealed(root / "learning_replay.json") != replay
        or json.loads((root / "stable_model.json").read_text()) != candidate
        or summary["commitment"] != commitment
        or summary["physical_episode_count"] != 36
    ):
        raise ValueError("stable actor weights or physical protocol do not reconstruct")
    reproduced = json.loads((root / "reproduction.json").read_text())
    if reproduced.get("matched_course_count") != 12:
        raise ValueError("complete byte-identical parent reproduction required")
    rows = []
    for i, (seed, lane) in enumerate(COURSES):
        parent_report = _sealed(root / f"seed{seed}-lane{lane}-reproduction-parent/report.json")
        for arm, model in (("champion", parent), ("stable", candidate)):
            folder = root / f"seed{seed}-lane{lane}-{arm}-actor"
            report = _sealed(folder / "report.json")
            result = _outcome(folder, report["contact_motor_policy_hash"], commitment)
            if (
                report["parent_report_hash"] != parent_report["report_hash"]
                or report["contact_motor_policy"]["online_motor_proof"]["model"] != model
            ):
                raise ValueError("stable physical action not bound to requested model and parent")
            if arm == "champion":
                old_arm = f"g{best[-1]['generation']}-neural" if best else "neural-reproduction"
                historical = _sealed(
                    online_root / f"seed{seed}-lane{lane}-{old_arm}-actor/report.json"
                )
                if any(
                    report[k] != historical[k]
                    for k in (
                        "body_trace_hash",
                        "trace_hash",
                        "asset_hash",
                        "sonic_qualification_hash",
                    )
                ):
                    raise ValueError("stable source failed historical neural physics reproduction")
            else:
                outcome = result["outcome"]
                cached = summary["rows"][i]
                if (
                    cached["course_index"] != i
                    or cached["report_hash"] != report["report_hash"]
                    or any(
                        cached[k] != outcome[k]
                        for k in cached
                        if k not in ("course_index", "report_hash")
                    )
                ):
                    raise ValueError("stable physical score differs from raw action and physics")
                rows.append({"course_index": i, **outcome, "report_hash": report["report_hash"]})
    references = _sealed(validation_root / "validation_summary.json")
    by_course = {(r["seed"], r["lane"]): r for r in references["rows"]}
    score = retention_score(rows, [by_course[c] for c in COURSES])
    old_neural = _sealed(preview_root / "preview_summary.json")
    loss = sum(
        old_neural["rows"][i]["neural"]["high_quality"] and not rows[i]["high_quality"]
        for i in range(12)
    )
    if (
        score != summary["score"]
        or loss != summary["previous_nine_high_quality_loss"]
        or summary["consumed_gate_passed"] != (score["consumed_training_gate_passed"] and loss == 0)
    ):
        raise ValueError("stable consumed gate does not reproduce")
    ledger = dict(
        model_hash=candidate["model_hash"],
        score=score,
        previous_nine_high_quality_loss=loss,
        rows=rows,
        consumed_gate_passed=summary["consumed_gate_passed"],
    )
    result = dict(
        schema="soccer.rsi.independent_stable_replay_review.v309",
        training_complete=True,
        body_hash=commitment["asset_hash"],
        best_model_hash=candidate["model_hash"],
        consumed_gate_passed=summary["consumed_gate_passed"],
        audited_physical_episode_count=36,
        independently_reviewed_replay_hash=replay["report_hash"],
        generations=[ledger],
        learning_kind="offline_replay_recalibration",
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
