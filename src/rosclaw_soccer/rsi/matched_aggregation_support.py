"""Authenticate matched NAV evidence before considering another numeric fit.

No physics, optimizer, data admission or deployment. A prior student failure
alone is insufficient: the same-approach expert must offer train-source value.
"""

import json
import subprocess
from collections.abc import Callable
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.matched_reference_comparison import compare_matched_references
from scripts.rsi_compare_matched_references import digest, load_cohort, require_terminal

PROFILE = "matched-zero-supported-one-round-reference-aggregation-fit-v1"
LEGACY_PROFILE = "conditional-one-round-reference-aggregation-fit-v1"


def require_matched_support(
    spec: dict[str, Any],
    *,
    current_contact_configuration: str,
    current_complete_hash: str,
    sealed: Callable[..., dict[str, Any]],
    pins: dict[str, str],
) -> dict[str, Any]:
    if set(spec) != {"configuration", "configuration_file_hash", "unit"}:
        raise ValueError(
            "explicit whole matched comparison configuration and terminal unit required"
        )
    require_terminal(spec["unit"])
    cp = Path(spec["configuration"])
    if digest(cp) != spec["configuration_file_hash"]:
        raise ValueError("original matched comparison configuration required")
    pins[str(cp)] = digest(cp)
    cfg = json.loads(cp.read_text())
    source = Path(cfg["source_root"])
    if (
        cfg.get("profile") != "matched-positive-zero-teacher-student-comparison-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or set(cfg["cohorts"]) != {"zero", "student"}
        or Path(cfg["cohorts"]["student"]["contact_configuration"]).resolve()
        != Path(current_contact_configuration).resolve()
        or cfg["cohorts"]["student"]["contact_configuration_file_hash"]
        != digest(current_contact_configuration)
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "neural_training_authorized",
                "dataset_admission_authorized",
                "private_fresh_accessed",
            )
        )
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "same current student, explicit matched zero and original frozen comparison required"
        )
    report = sealed(Path(cfg["output_root"]) / "complete.json")
    cohorts = {role: load_cohort(c, sealed=sealed, pins=pins) for role, c in cfg["cohorts"].items()}
    comparison = compare_matched_references(cohorts["zero"], cohorts["student"])
    if (
        report.get("schema") != "soccer.rsi.matched_reference_comparison_complete.v1"
        or report["configuration"] != cfg
        or report["cohort_complete_hashes"]
        != {role: c["complete"]["report_hash"] for role, c in cohorts.items()}
        or report["cohort_audit_hashes"]
        != {role: c["audit"]["report_hash"] for role, c in cohorts.items()}
        or report["cohort_complete_hashes"]["student"] != current_complete_hash
        or report["comparison"] != comparison
        or report.get("original_task_labels_unchanged") is not True
        or report.get("scientific_recommendation_not_automatic_successor") is not True
        or any(
            type(report.get(k)) is not int or report[k] != 0
            for k in ("new_optimizer_updates", "new_physical_steps", "new_foundation_inferences")
        )
        or any(report.get(k) is not False for k in ("hardware_authorized", "promotion_authorized"))
    ):
        raise ValueError(
            "whole original three-way report must rebuild, not a caller-supplied approval"
        )
    for name, h in report["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("original comparison inputs and physical witness bytes changed")
        pins[name] = h
    supported = (
        comparison["next_research"]
        == "BOUNDED_SAME_STATE_EXPERT_SUPERVISION_RESEARCH_HAS_TRAIN_SOURCE_SUPPORT"
    )
    return dict(
        matched_comparison_hash=report["report_hash"],
        matched_comparison_configuration_file_hash=spec["configuration_file_hash"],
        current_student_complete_hash=current_complete_hash,
        expert_advantage_training_sources=comparison["expert_advantage_training_sources"],
        teacher_minus_zero_train_net=comparison["teacher_minus_zero_train_net"],
        considers_one_numeric_research_fit=supported,
        validation_used_to_select_fit=False,
        counterfactual_expert_success_verified=False,
        teacher_qualified=False,
        dataset_admission_authorized=False,
        neural_training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def constrain_recommendation(decision: dict[str, Any], support: dict[str, Any]) -> dict[str, Any]:
    """Add a required causal prerequisite; never expand an existing fit budget."""
    result = dict(
        decision,
        matched_support=support,
        original_action=decision["action"],
        query_body_distribution="current student's recorded states, bound to prepared generation",
    )
    if type(support.get("considers_one_numeric_research_fit")) is not bool:
        raise ValueError("authenticated explicit matched research consideration required")
    if (
        not support["considers_one_numeric_research_fit"]
        and decision["finite_numeric_fit_recommended"]
    ):
        result.update(
            action="IMPROVE_EXPERT_OR_APPROACH_NO_FIT_WITHOUT_MATCHED_TRAIN_SOURCE_SUPPORT",
            finite_numeric_fit_recommended=False,
        )
    return result
