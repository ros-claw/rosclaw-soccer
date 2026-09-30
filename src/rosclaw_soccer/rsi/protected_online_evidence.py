"""Read-only replay of physical feedback, critic/actor updates and retention.

An incomplete experiment reports progress only. Cached scores and learner
claims never substitute for reconstructing actions and measuring raw physics.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_contract import load_policy
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome, _sealed, retention_score
from rosclaw_soccer.rsi.motor_learning_bank import causal_context
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    actor_parameters,
    make_model,
    update_from_physics,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_train_protected_online_motor_v308 import _normalize


def validate_feedback(feedback: dict[str, Any], parent_hash: str, generation: int) -> None:
    expected = {(i, c) for i in (0, 2, 5) for c in range(12)}
    samples = feedback.get("samples", [])
    actual = {(s["course_index"], s["candidate"]) for s in samples}
    if (
        feedback.get("parent_model_hash") != parent_hash
        or len(samples) != 36
        or actual != expected
        or feedback.get("runtime_selection_authorized") is not False
        or feedback.get("generation") != generation
    ):
        raise ValueError("physical feedback batch or online model lineage drift")


def review_online(
    root: Path, bank_path: Path, preview_root: Path, validation_root: Path
) -> dict[str, Any]:
    commitment = json.loads((root / "commitment.json").read_text())
    bank, preview, reference = (
        _sealed(bank_path),
        _sealed(preview_root / "preview_summary.json"),
        _sealed(validation_root / "validation_summary.json"),
    )
    if (
        commitment["bank_hash"] != bank["report_hash"]
        or commitment["predecessor_preview_hash"] != preview["report_hash"]
        or commitment["activation_ceiling"] != "SIM_ONLY"
        or commitment["partition"] != "TRAIN_CONSUMED"
    ):
        raise ValueError("online experiment predecessor commitment drift")
    contexts = {
        tuple(s["audit_metadata"]["course"]): s["observation"] for s in bank["critic_samples"]
    }
    protected_indices = [i for i, r in enumerate(preview["rows"]) if r["neural"]["high_quality"]]
    base = json.loads((root / "models/initial.json").read_text())["base_model"]
    best = make_model(
        base, [contexts[COURSES[i]] for i in protected_indices], preview["report_hash"]
    )
    if (
        json.loads((root / "models/initial.json").read_text()) != best
        or best["model_hash"] != commitment["initial_model_hash"]
    ):
        raise ValueError("initial protected model does not derive from actual predecessor")
    reference_map = {(r["seed"], r["lane"]): r for r in reference["rows"]}
    ordered_reference = [reference_map[c] for c in COURSES]
    initial_score = retention_score([r["neural"] for r in preview["rows"]], ordered_reference)
    best_rank = (1.0, *initial_score["rank"])
    replay = [
        dict(
            observation=s["observation"],
            normalized_action=_normalize(s["motor_parameters"]),
            outcome=s["learning_labels"],
            report_hash=s["audit_metadata"]["source_report_hash"],
        )
        for s in bank["critic_samples"]
    ]
    reproduced = json.loads((root / "reproduction.json").read_text())
    if reproduced.get("matched_course_count") != 12:
        raise ValueError("complete predecessor reproduction required for online review")

    def physical(index: int, arm: str) -> tuple[dict[str, Any], dict[str, Any], list[float]]:
        seed, lane = COURSES[index]
        folder = root / f"seed{seed}-lane{lane}-{arm}-actor"
        report = _sealed(folder / "report.json")
        actual = _outcome(folder, report["contact_motor_policy_hash"], commitment)
        parent = _sealed(root / f"seed{seed}-lane{lane}-reproduction-parent/report.json")
        if (
            report["parent_report_hash"] != parent["report_hash"]
            or parent["source_hash"] != commitment["runner_hash"]
        ):
            raise ValueError("online actor detached from frozen parent")
        for key in ("source_hash", "asset_hash", "sonic_qualification_hash"):
            if report[key] != parent[key]:
                raise ValueError("online body or controller provenance drift")
        with (
            np.load(folder / "body_trace.npz", allow_pickle=False) as body,
            np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
        ):
            context = list(causal_context(body, motor, actual["outcome"]["first_contact_frame"]))
        if context != contexts[COURSES[index]]:
            raise ValueError("online physical context is not the frozen causal context")
        return report, actual["outcome"], context

    for i in range(12):
        report, outcome, context = physical(i, "neural-reproduction")
        seed, lane = COURSES[i]
        old = _sealed(preview_root / f"seed{seed}-lane{lane}-neural-actor/report.json")
        if any(
            report[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("predecessor neural physics failed byte reproduction")
        ledger = reproduced["rows"][i]
        if ledger["report_hash"] != report["report_hash"] or any(
            ledger["outcome"][k] != outcome[k] for k in ledger["outcome"]
        ):
            raise ValueError("reproduction ledger differs from raw measured outcome")
        replay.append(
            dict(
                observation=context,
                normalized_action=_normalize(actor_parameters(best, context)),
                outcome=ledger["outcome"],
                report_hash=report["report_hash"],
            )
        )
    generations, audited_episodes = [], 24
    for generation in range(3):
        if not (root / f"generation-{generation}.json").is_file():
            break
        feedback = _sealed(root / f"feedback-g{generation}.json")
        validate_feedback(feedback, best["model_hash"], generation)
        for sample in feedback["samples"]:
            i, c = sample["course_index"], sample["candidate"]
            report, outcome, context = physical(i, f"g{generation}-c{c}-explore")
            policy, knots = load_policy(root / f"policies/g{generation}-i{i}-c{c}.json")
            action = _normalize([*knots.ravel().tolist(), policy["phase_gap_end_m"]])
            if (
                report["contact_motor_policy"] != policy
                or sample["normalized_action"] != action
                or sample["observation"] != context
                or sample["report_hash"] != report["report_hash"]
                or any(sample["outcome"][k] != outcome[k] for k in sample["outcome"])
            ):
                raise ValueError("critic replay not bound to actual physical action and outcome")
            audited_episodes += 1
        replay.extend(feedback["samples"])
        model = update_from_physics(best, replay, feedback["report_hash"])
        if model != json.loads((root / f"models/g{generation}.json").read_text()):
            raise ValueError("actor or critic weights do not reproduce from physical replay")
        ledger = _sealed(root / f"generation-{generation}.json")
        rows = []
        for i in range(12):
            report, outcome, _ = physical(i, f"g{generation}-neural")
            if report["contact_motor_policy"]["online_motor_proof"]["model"] != model:
                raise ValueError("neural evaluation does not execute the replay-derived model")
            cached = ledger["rows"][i]
            if (
                cached["course_index"] != i
                or cached["report_hash"] != report["report_hash"]
                or any(cached[k] != outcome[k] for k in outcome if k in cached)
            ):
                raise ValueError("online neural score differs from raw physics")
            rows.append(outcome)
            audited_episodes += 1
        score = retention_score(rows, ordered_reference)
        loss = sum(
            preview["rows"][i]["neural"]["high_quality"] and not rows[i]["high_quality"]
            for i in range(12)
        )
        if (
            ledger["score"] != score
            or ledger["previous_nine_high_quality_loss"] != loss
            or ledger["consumed_gate_passed"]
            != (score["consumed_training_gate_passed"] and loss == 0)
        ):
            raise ValueError("online gate or retention claim disagrees with reconstructed physics")
        current_rank = (float(loss == 0), *score["rank"])
        if current_rank > best_rank:
            best, best_rank = model, current_rank
        generations.append(ledger)
        if ledger["consumed_gate_passed"]:
            break
    complete = (root / "training_summary.json").is_file()
    if complete:
        summary = _sealed(root / "training_summary.json")
        if (
            summary["generations"] != generations
            or summary["best_model_hash"] != best["model_hash"]
            or summary["total_replay_records"] != len(replay)
            or (len(generations) != 3 and not generations[-1]["consumed_gate_passed"])
        ):
            raise ValueError("completed online training summary does not reproduce")
    result = dict(
        schema="soccer.rsi.independent_protected_online_review.v308",
        training_complete=complete,
        audited_physical_episode_count=audited_episodes,
        complete_generation_count=len(generations),
        generations=generations,
        best_model_hash=best["model_hash"],
        consumed_gate_passed=complete and any(g["consumed_gate_passed"] for g in generations),
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
        runtime_selection_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    return result
