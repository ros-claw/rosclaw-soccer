"""Authenticate the whole closed C2 study before a C3 heading experiment."""

import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.original_ik_planner_diagnostics import ik_contact_readout
from scripts.rsi_load_original_ik_cohort import load_closed_original_ik
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def load_heading_predecessor(spec, *, sealed, pins):
    require_terminal(spec["stage_unit"])
    path = Path(spec["stage_readback"])
    if path.is_symlink() or digest(path) != spec["stage_file_hash"]:
        raise ValueError("original unchanged C2 final-stage file required")
    stage = sealed(path)
    cfg = stage["configuration"]
    source = Path(cfg["source_root"])
    if (
        stage["report_hash"] != spec["stage_readback_hash"]
        or stage.get("schema") != "soccer.rsi.original_ik_first_touch_stage_pairs_readback.v1"
        or cfg.get("stage_profile") != "original-ik-pulse"
        or cfg.get("mode") != "SIM_ONLY"
        or Path(cfg["output"]).resolve() != path.resolve()
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or stage.get("all_six_receipts_and_failures_retained") is not True
        or any(
            type(stage.get(k)) is not int or stage[k] != n
            for k, n in (
                ("source_native_bodies", 6),
                ("derived_courses", 3),
                ("source_scene_clusters", 3),
                ("new_physical_steps", 0),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            stage.get(k) is not False
            for k in (
                "teacher_qualified",
                "dataset_admission_authorized",
                "receiving_skill_qualified",
                "successor_readiness_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("whole normal-closed, source-pinned C2 final-stage report required")
    # Final stage authenticated the contact producer, actual dense replay,
    # metrics, and its own source. Recheck every original file, not just its seal.
    for name, h in stage["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("original whole C2 final-stage evidence or source changed")
        pins[name] = h
    cohort = load_closed_original_ik(stage["effective_cohort_spec"], sealed=sealed, pins=pins)
    contact_spec = stage["effective_contact_spec"]
    require_terminal(contact_spec["unit"])
    contact, replay, metrics = (
        sealed(contact_spec[k]) for k in ("complete", "replay_complete", "metrics")
    )
    measured = ik_contact_readout(cohort["declaration"], cohort["complete"], replay, metrics)
    if (
        stage["native_complete_hash"] != cohort["complete"]["report_hash"]
        or stage["native_audit_hash"] != cohort["audit"]["report_hash"]
        or stage["contact_complete_hash"] != contact["report_hash"]
        or stage["stage_diagnostics"] != measured["assessment"]["stages"]
        or stage["pulse_dense_nonfoot_endpoints"] != measured["pulse_endpoints_with_nonfoot_over1n"]
        or contact.get("schema") != "soccer.rsi.original_ik_planner_contact_complete.v1"
        or contact["report_hash"] != contact_spec["complete_hash"]
        or contact["upstream_readback_hash"] != cohort["audit"]["report_hash"]
        or contact["replay_complete_hash"] != replay["report_hash"]
        or contact["metrics_hash"] != metrics["report_hash"]
        or contact["measured_contact_readout"] != measured
        or any(
            type(contact.get(k)) is not int or contact[k] != n
            for k, n in (
                ("new_fixed_control_replay_substeps", 18000),
                ("new_closed_loop_policy_trials", 0),
                ("new_neural_inferences", 0),
                ("new_optimizer_updates", 0),
                ("total_metrics_endpoints", 6),
            )
        )
        or any(
            contact.get(k) is not False
            for k in (
                "teacher_qualified",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("all C2 body, CPU, dense and stage outcomes must reconstruct")
    for name, h in contact["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("original complete dense witness bytes changed")
        pins[name] = h
    return dict(cohort=cohort, stage=stage, contact=contact, replay=replay, metrics=metrics)
