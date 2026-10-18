"""Conditional once-only 128-step numeric aggregation after the WHOLE current exam.

No new physics, implicit retry, validation selection, data admission or activation.
Recorded own-body expert suggestions are unqualified counterfactual labels.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.matched_aggregation_support import (
    LEGACY_PROFILE,
    PROFILE,
    constrain_recommendation,
    require_matched_support,
)
from rosclaw_soccer.rsi.reference_aggregation_decision import recommend_aggregation
from rosclaw_soccer.rsi.reference_state_aggregation import LABEL_CEILING
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout, require_cohort
from rosclaw_soccer.rsi.reference_student_imitation import (
    fit_numeric_reference_student,
    prepare_numeric_batch,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile") not in (LEGACY_PROFILE, PROFILE)
        or (cfg.get("profile") == PROFILE and type(cfg.get("matched_comparison")) is not dict)
        or (cfg.get("profile") == LEGACY_PROFILE and "matched_comparison" in cfg)
        or cfg.get("mode") != "SIM_ONLY"
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or cfg.get("acknowledged_label_ceiling") != LABEL_CEILING
        or cfg.get("experimental_numeric_fit_authorized") is not True
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "teacher_auto_admission",
                "private_fresh_accessed",
                "online_rl_claimed",
                "validation_checkpoint_selection",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("fit_steps", 128),
                ("fit_seed", 2026101902),
                ("maximum_numeric_candidates", 1),
                ("new_physical_steps", 0),
                ("new_foundation_inferences", 0),
            )
        )
        or Path(__file__).resolve() != source / "scripts/rsi_fit_reference_state_aggregation.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "one matched seed/128-step research fit, explicit ceiling and frozen source required"
        )
    for unit in cfg["upstream_units"]:
        require_terminal(unit)
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get("report_hash") != hash_json({k: x for k, x in v.items() if k != "report_hash"}):
            raise ValueError("original sealed complete evidence required")
        pins[str(path)] = digest(path)
        return v

    prepared = sealed(cfg["prepared_aggregation"])
    proposed = prepared["proposed_batch"]
    if (
        prepared["report_hash"] != cfg["prepared_aggregation_hash"]
        or prepared["schema"] != "soccer.rsi.proposed_reference_aggregation_prepared.v1"
        or proposed["report_hash"]
        != hash_json({k: v for k, v in proposed.items() if k != "report_hash"})
        or proposed["acknowledged_label_ceiling"] != LABEL_CEILING
        or prepared["dataset_admission_authorized"] is not False
        or prepared["original_query_eligibility_overwritten"] is not False
    ):
        raise ValueError("original proposed-only, not admitted, 29-body aggregation required")
    cp = Path(cfg["current_contact_configuration"])
    if digest(cp) != cfg["current_contact_configuration_file_hash"]:
        raise ValueError("fixed current exam contact configuration required")
    pins[str(cp)] = digest(cp)
    cc = json.loads(cp.read_text())
    pp = Path(cc["producer_config"])
    if digest(pp) != cc["producer_config_file_hash"]:
        raise ValueError("original current own-body producer configuration required")
    pins[str(pp)] = digest(pp)
    pc = json.loads(pp.read_text())
    proot = Path(pc["output_root"])
    declaration, complete = (sealed(proot / n) for n in ("declaration.json", "complete.json"))
    audit, dense = sealed(cc["readback"]), sealed(cc["complete_path"])
    pairs = require_cohort(pc, declaration, complete, audit)
    replay = sealed(Path(cc["replay_output_root"]) / "complete.json")
    metrics = sealed(cc["metrics_output"])
    measured = measured_readout(declaration, complete, replay, metrics)
    if (
        audit["reader_source_hash"] != digest(cc["audit_source"])
        or dense["upstream_readback_hash"] != audit["report_hash"]
        or dense["replay_complete_hash"] != replay["report_hash"]
        or dense["metrics_hash"] != metrics["report_hash"]
        or dense["paired_results"] != pairs
        or dense["measured_contact_readout"] != measured
    ):
        raise ValueError(
            "whole current physical exam, independent audit and dense diagnostics required"
        )
    for owner in (prepared, declaration, audit):
        for name, h in {**owner["input_file_hashes"], **owner.get("prepared_pins", {})}.items():
            if digest(name) != h:
                raise ValueError("original prepared data or current physical sources changed")
            pins[name] = h
    w = prepared["numeric_arrays"]
    if digest(w["path"]) != w["sha256"]:
        raise ValueError("original whole proposed numeric arrays required")
    pins[w["path"]] = w["sha256"]
    with np.load(w["path"], allow_pickle=False) as a:
        if set(a.files) != {
            "actual_observations",
            "reference_proposals",
            "partitions",
            "source_seeds",
            "experimental_supervision",
        }:
            raise ValueError("complete original actual-input aggregation arrays required")
        batch = prepare_numeric_batch(
            a["actual_observations"],
            a["reference_proposals"],
            partitions=a["partitions"].tolist(),
            source_seeds=a["source_seeds"].tolist(),
            eligible=a["experimental_supervision"].tolist(),
        )
    if (
        batch["original_observations"].shape != (29, 300, 135)
        or int(batch["train"].sum()) != 6
        or int(batch["validation"].sum()) != 2
        or hash_bytes(batch["original_observations"].tobytes())
        != proposed["actual_measured_inputs_hash"]
        or hash_bytes(batch["targets"].tobytes()) != proposed["same_body_label_proposals_hash"]
        or batch["eligible"] != [r["experimental_numeric_supervision"] for r in proposed["rows"]]
    ):
        raise ValueError("exact proposed six-train/two-validation whole batch required")
    positive_nonfoot = sum(
        r["job"]["plan"] == "positive19"
        and r["job"]["role"] == "student"
        and r["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        for r in measured["rows"]
    )
    decision = recommend_aggregation(pairs, positive_dense_nonfoot=positive_nonfoot)
    if cfg["profile"] == PROFILE:
        if (
            prepared.get("actual_behavior_model_hash")
            != declaration["models"]["student"]["model_hash"]
            or prepared.get("actual_behavior_fit_report_hash") != declaration["fit_report_hash"]
            or prepared.get("bounded_current_own_state_window_not_cumulative_dagger") is not True
        ):
            raise ValueError(
                "proposed input window must belong to the current actual student generation"
            )
        support = require_matched_support(
            cfg["matched_comparison"],
            current_contact_configuration=cfg["current_contact_configuration"],
            current_complete_hash=complete["report_hash"],
            sealed=sealed,
            pins=pins,
        )
        decision = constrain_recommendation(decision, support)
    if args.preflight:
        print(
            "AGGREGATION_FIT_SCOPE_READY_ZERO_OPTIMIZER_ALLOCATIONS", decision["action"], flush=True
        )
        return
    root.mkdir()
    write_once(
        root / "started.json",
        dict(configuration=cfg, decision=decision, automatic_retry_authorized=False),
    )
    try:
        report = dict(
            schema="soccer.rsi.reference_state_aggregation_fit_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            prepared_aggregation_hash=prepared["report_hash"],
            current_exam_complete_hash=complete["report_hash"],
            current_audit_hash=audit["report_hash"],
            current_dense_hash=dense["report_hash"],
            paired_results=pairs,
            decision=decision,
            counterfactual_labels_are_not_verified_success=True,
            own_body_student_evaluated=False,
            dataset_admission_authorized=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            online_rl_claimed=False,
            new_physical_steps=0,
            new_foundation_inferences=0,
        )
        if decision["finite_numeric_fit_recommended"]:
            fit = fit_numeric_reference_student(
                batch,
                world_hash=pc["mesh_task_contact_world_hash"],
                parent_model_hash=pc["behavior_model_hash"],
                seed=cfg["fit_seed"],
                steps=cfg["fit_steps"],
                ignore_past_reference=True,
            )
            write_once(root / "numeric-fit.json", fit)
            report.update(
                status="NUMERIC_AGGREGATION_FIT_REQUIRES_NEW_OWN_BODY_EXAM",
                numeric_fit_hash=fit["report_hash"],
                numeric_model_hash=fit["numeric_model"]["model_hash"],
                optimizer_updates=fit["learning_receipt"]["completed_optimizer_steps"],
            )
        else:
            report.update(
                status="NO_FIT_CURRENT_EVIDENCE_REQUIRES_DIFFERENT_RESEARCH_BRANCH",
                optimizer_updates=0,
            )
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source or evidence changed during conditional aggregation fit")
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(report["status"], report["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
