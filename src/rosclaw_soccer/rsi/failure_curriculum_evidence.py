"""Read-only independent review of the consumed motor curriculum.

An incomplete run is progress, never a passed experiment. Each completed
candidate is reconstructed from its physical/action traces before ranking.
This does not authorize promotion, fresh evaluation or hardware execution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.approach_lateral_tracking_evidence import audit_lateral_approach
from rosclaw_soccer.rsi.contact_motor_contract import load_policy
from rosclaw_soccer.rsi.contextual_first_touch_option import first_touch_reward
from rosclaw_soccer.rsi.independent_first_touch_bank import post_contact_displacement
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.physical_report_io import load_physical_report
from rosclaw_soccer.rsi.taskspace_swing_evidence import audit_taskspace_swing_trace
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality


def retention_score(rows: list[dict[str, Any]], reference: list[dict[str, Any]]) -> dict[str, Any]:
    """Independently name every lexicographic guardrail, not just a reward sum."""
    if len(rows) != 12 or len(reference) != 12:
        raise ValueError("complete 12-course comparison required")
    counters = {
        "predecessor_high_quality_loss": 0,
        "raw_gain_high_quality_loss": 0,
        "old_clean_foot_loss": 0,
        "new_out_of_play": 0,
        "old_high_quality_loss": 0,
    }
    failures: dict[str, list[list[int]]] = {k: [] for k in counters}
    for course, row, anchor in zip(COURSES, rows, reference, strict=True):
        for arm in ("gain_08", "gain_12", "learned"):
            for key in ("high_quality", "clean_foot_only"):
                if type(anchor["arms"][arm][key]) is not bool:
                    raise ValueError("boolean historical outcome required")
        if any(type(row[key]) is not bool for key in ("high_quality", "clean_foot_only")):
            raise ValueError("boolean measured outcome required")
        if not all(
            type(row[k]) in (int, float) and np.isfinite(row[k])
            for k in ("minimum_pelvis_z_m", "maximum_lateral_excursion_m", "reward")
        ) or not np.isfinite(anchor["arms"]["gain_08"]["maximum_lateral_excursion_m"]):
            raise ValueError("finite physical measurements required")
        lost = {
            "predecessor_high_quality_loss": anchor["arms"]["learned"]["high_quality"]
            and not row["high_quality"],
            "raw_gain_high_quality_loss": anchor["arms"]["gain_12"]["high_quality"]
            and not row["high_quality"],
            "old_clean_foot_loss": anchor["arms"]["gain_08"]["clean_foot_only"]
            and not row["clean_foot_only"],
            "new_out_of_play": anchor["arms"]["gain_08"]["maximum_lateral_excursion_m"]
            <= 4
            < row["maximum_lateral_excursion_m"],
            "old_high_quality_loss": anchor["arms"]["gain_08"]["high_quality"]
            and not row["high_quality"],
        }
        for key, value in lost.items():
            if value:
                counters[key] += 1
                failures[key].append(list(course))
    safe = all(row["minimum_pelvis_z_m"] >= 0.65 for row in rows)
    successes = sum(row["high_quality"] for row in rows)
    rank = [float(safe), *[-float(v) for v in counters.values()], float(successes)]
    rank.append(float(sum(row["reward"] for row in rows)))
    return {
        "safe_pelvis_guardrail": safe,
        **counters,
        "high_quality_count": successes,
        "rank": rank,
        "lost_courses": failures,
        "consumed_training_gate_passed": safe and not any(counters.values()) and successes >= 10,
    }


def _sealed(path: Path) -> dict[str, Any]:
    report = (
        load_physical_report(path)
        if path.name in ("report.json", "report.json.gz")
        else load_json_artifact(path)
    )
    if report.get("report_hash") != hash_json(
        {k: v for k, v in report.items() if k != "report_hash"}
    ):
        raise ValueError(f"unsealed evidence: {path}")
    return report


def learning_frontier(candidates: list[dict[str, Any]]) -> dict[str, Any]:
    """Future-labelled offline teacher coverage, explicitly NOT a runtime selector."""
    covered, uncovered = [], []
    for index, (seed, lane) in enumerate(COURSES):
        eligible = [
            candidate
            for candidate in candidates
            if candidate["rows"][index]["high_quality"]
            and candidate["rows"][index]["minimum_pelvis_z_m"] >= 0.65
        ]
        if not eligible:
            uncovered.append([seed, lane])
            continue
        # A rejected global model may preserve a useful local demonstration.
        # Reward labels are permitted ONLY for constructing future training data.
        teacher = max(
            eligible,
            key=lambda r: (r["rows"][index]["reward"], -r["generation"], -r["candidate"]),
        )
        covered.append(
            {
                "course": [seed, lane],
                "donor_policy_hash": teacher["policy_hash"],
                "donor_generation": teacher["generation"],
                "donor_candidate": teacher["candidate"],
                "raw_report_hash": teacher["raw_report_hashes"][index],
            }
        )
    return {
        "schema": "soccer.rsi.offline_teacher_frontier.v1",
        "partition": "TRAIN_CONSUMED",
        "selection_uses_future_outcome_labels": True,
        "runtime_selection_authorized": False,
        "promotion_authorized": False,
        "covered_course_count": len(covered),
        "uncovered_courses": uncovered,
        "demonstration_donors": covered,
        "interpretation": "Offline teacher union, not success of any single deployed policy",
    }


def _outcome(
    folder: Path,
    policy_hash: str,
    commitment: dict[str, Any],
    *,
    decoder_sink: list[Any] | None = None,
) -> dict[str, Any]:
    report = _sealed(folder / "report.json")
    if (
        report.get("contact_motor_policy_hash") != policy_hash
        or report.get("source_hash") != commitment["runner_hash"]
        or report.get("asset_hash") != commitment["asset_hash"]
        or report.get("navigation_lateral_ball_gain") != 1.2
        or report.get("navigation_lateral_negative_only") is not True
    ):
        raise ValueError("raw candidate provenance mismatch")
    audit = (
        audit_lateral_approach(folder)
        if decoder_sink is None
        else audit_lateral_approach(folder, decoder_sink=decoder_sink)
    )
    with np.load(folder / "late_swing_action_trace.npz", allow_pickle=False) as trace:
        action = audit_taskspace_swing_trace(trace, report, frames=300, count=1)
    observed = report["environments"][0]
    with np.load(folder / "trace.npz", allow_pickle=False) as trace:
        displacement = post_contact_displacement(
            trace["ball_position_m"][:, 0], observed["first_contact_frame"]
        )
    row = {
        "contact_body_indices": observed["contact_body_indices"],
        "first_contact_frame": observed["first_contact_frame"],
        "minimum_pelvis_z_m": observed["minimum_pelvis_z_m"],
        "maximum_lateral_excursion_m": report["single_instance_max_lateral_excursion_m"],
        "clean_foot_only": bool(
            observed["contact_body_indices"] and set(observed["contact_body_indices"]) <= {0, 1}
        ),
        "action_audit": action,
        "command_audit_hash": audit["report_hash"],
        "command_active_frames": audit["active_frames"],
        **displacement,
    }
    row["reward"] = first_touch_reward(
        {**row, "max_lateral_excursion_m": row["maximum_lateral_excursion_m"]}
    )
    row["high_quality"] = high_quality(row)
    return {"report": report, "outcome": row}


def review_curriculum(root: Path, validation_root: Path) -> dict[str, Any]:
    """Audit complete candidates in an ongoing or completed immutable run."""
    commitment = json.loads((root / "commitment.json").read_text(encoding="utf-8"))
    prior = _sealed(validation_root / "validation_summary.json")
    if (
        commitment["prior_validation_hash"] != prior["report_hash"]
        or prior.get("promotion_authorized") is not False
        or prior.get("fresh_holdout_open_authorized") is not False
        or len(prior["rows"]) != len(COURSES)
        or {(r["seed"], r["lane"]) for r in prior["rows"]} != set(COURSES)
    ):
        raise ValueError("curriculum reference mismatch")
    references = {(r["seed"], r["lane"]): r for r in prior["rows"]}
    ordered_reference = [references[course] for course in COURSES]
    candidates = []
    audited_episodes = 0
    for generation in range(4):
        for candidate in range(8):
            arm = f"g{generation}-c{candidate}"
            folders = [root / f"seed{seed}-lane{lane}-{arm}-actor" for seed, lane in COURSES]
            if not all((folder / "report.json").is_file() for folder in folders):
                continue
            policy_path = root / f"policies/{arm}.json"
            policy, _ = load_policy(policy_path)
            if policy["training_commitment"] != hash_json(commitment):
                raise ValueError("candidate not bound to frozen training commitment")
            rows, reports = [], []
            for (seed, lane), folder in zip(COURSES, folders, strict=True):
                measured = _outcome(folder, policy["policy_hash"], commitment)
                raw = measured["report"]
                parent = _sealed(root / f"seed{seed}-lane{lane}-reproduction-parent/report.json")
                if (
                    raw["training_course_seed"] != seed
                    or raw["single_course_lane"] != lane
                    or raw["parent_report_hash"] != parent["report_hash"]
                    or raw["sonic_qualification_hash"] != parent["sonic_qualification_hash"]
                    or raw["environments"][0]["course"] != parent["environments"][0]["course"]
                ):
                    raise ValueError("candidate course or foundation anchor mismatch")
                if generation == 0 and candidate == 0:
                    clone = _sealed(root / f"seed{seed}-lane{lane}-reproduction-actor/report.json")
                    historical = _sealed(
                        validation_root / f"seed{seed}-lane{lane}-learned-actor/report.json"
                    )
                    if any(
                        raw[k] != clone[k] or clone[k] != historical[k]
                        for k in ("body_trace_hash", "trace_hash")
                    ):
                        raise ValueError("old model/initial clone physics reproduction mismatch")
                rows.append(measured["outcome"])
                reports.append(raw["report_hash"])
                audited_episodes += 1
            score = retention_score(rows, ordered_reference)
            entry = {
                "generation": generation,
                "candidate": candidate,
                "policy": str(policy_path),
                "policy_hash": policy["policy_hash"],
                "phase_gap_end_m": policy["phase_gap_end_m"],
                "score": score,
                "rows": rows,
                "raw_report_hashes": reports,
            }
            ledger_path = root / f"{arm}-result.json"
            if ledger_path.is_file():
                ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
                if (
                    ledger["rows"] != rows
                    or ledger["reports"] != reports
                    or ledger["score"] != score["rank"]
                    or ledger["generation"] != generation
                    or ledger["candidate"] != candidate
                    or Path(ledger["policy"]).resolve() != policy_path.resolve()
                ):
                    raise ValueError("trainer result disagrees with independently audited physics")
            candidates.append(entry)
    if not candidates:
        raise ValueError("no complete 12-course candidate available yet")
    # Explicit tie rule: earliest generation, then lowest candidate index.
    best = max(candidates, key=lambda r: (r["score"]["rank"], -r["generation"], -r["candidate"]))
    final_path = root / "training_summary.json"
    complete = final_path.is_file()
    if complete:
        summary = _sealed(final_path)
        if (
            len(candidates) != 32
            or summary["commitment"] != commitment
            or summary["best"]["generation"] != best["generation"]
            or summary["best"]["candidate"] != best["candidate"]
            or summary["best"]["score"] != best["score"]["rank"]
            or summary["consumed_training_gate_passed"]
            is not best["score"]["consumed_training_gate_passed"]
            or summary["independent_physical_episode_count"] != 408
            or summary["distinct_consumed_course_count"] != 12
            or summary["promotion_authorized"] is not False
            or summary["fresh_holdout_open_authorized"] is not False
        ):
            raise ValueError("final training summary disagrees with independent review")
    result = {
        "schema": "soccer.rsi.failure_curriculum_independent_review.v1",
        "activation_ceiling": "SIM_ONLY",
        "partition": "TRAIN_CONSUMED",
        "training_complete": complete,
        "audited_candidate_physical_episode_count": audited_episodes,
        "complete_candidate_count": len(candidates),
        "distinct_consumed_course_count": 12,
        "candidates": candidates,
        "best": best,
        "offline_learning_frontier": learning_frontier(candidates),
        "consumed_training_gate_passed": complete
        and best["score"]["consumed_training_gate_passed"],
        "promotion_authorized": False,
        "fresh_holdout_open_authorized": False,
        "missing_qualifications": ["fresh_holdout", "cpu_mujoco", "continuous_team"],
        "commitment_hash": hash_json(commitment),
        "reference_report_hash": prior["report_hash"],
    }
    result["report_hash"] = hash_json(result)
    return result
