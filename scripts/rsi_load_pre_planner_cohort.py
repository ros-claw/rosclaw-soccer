"""Authenticate an original six-body pulse cohort, never execute it again."""

import json
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import (
    pulse_contact_readout,
    require_pulse_cohort,
)
from scripts.rsi_read_pre_planner_navigation import read_journal
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def load_closed_pulse(cfg, *, sealed, pins):
    for k in ("producer_unit", "audit_unit"):
        require_terminal(cfg[k])
    pp = Path(cfg["producer_config"])
    if digest(pp) != cfg["producer_config_file_hash"]:
        raise ValueError("original pulse producer configuration required")
    pins[str(pp)] = digest(pp)
    producer = json.loads(pp.read_text())
    source, root = Path(producer["source_root"]), Path(producer["output_root"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != producer["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or Path(cfg["producer_declaration"]).resolve() != (root / "declaration.json").resolve()
        or Path(cfg["producer_complete"]).resolve() != (root / "complete.json").resolve()
        or Path(cfg["audit_source"]).resolve()
        != (source / "scripts/rsi_read_pre_planner_navigation.py").resolve()
    ):
        raise ValueError("original frozen pulse source and owned outputs required")
    declaration, complete, audit = (
        sealed(cfg[k])
        for k in ("producer_declaration", "producer_complete", "source_terminal_readback")
    )
    assessment = require_pulse_cohort(producer, declaration, complete, audit)
    tip, state = read_journal(root, declaration)
    if (
        audit["report_hash"] != cfg["source_terminal_readback_hash"]
        or audit["reader_source_hash"] != digest(cfg["audit_source"])
        or audit["ledger_tip"] != tip
        or any(
            state["completed"][j["job"]] != r
            for j, r in zip(declaration["scope"]["jobs"], complete["results"], strict=True)
        )
    ):
        raise ValueError("whole original terminal audit and journal membership required")
    pins[cfg["audit_source"]] = digest(cfg["audit_source"])
    for owner in (declaration, audit):
        for name, h in {**owner["input_file_hashes"], **owner.get("prepared_pins", {})}.items():
            if digest(name) != h or (name in pins and pins[name] != h):
                raise ValueError("original pulse source/input/witness bytes changed")
            pins[name] = h
    for i, r in enumerate(complete["results"]):
        if sealed(root / f"receipt-{i:04d}.json", "receipt_hash") != r:
            raise ValueError("all original pulse receipt files required")
    return dict(
        producer=producer,
        declaration=declaration,
        complete=complete,
        audit=audit,
        assessment=assessment,
    )


def require_negative_predecessor(spec, *, sealed, pins):
    """Positive research follows whole negative evidence, never a selected prefix."""
    from rosclaw_soccer.rsi.planner_navigation_pulse_scope import PROFILE

    cohort = load_closed_pulse(spec, sealed=sealed, pins=pins)
    if cohort["producer"]["exam_profile"] != PROFILE:
        raise ValueError("original negative six-body predecessor required")
    require_terminal(spec["contact_unit"])
    contact = sealed(spec["contact_complete"])
    replay = sealed(spec["contact_replay_complete"])
    metrics = sealed(spec["contact_metrics"])
    measured = pulse_contact_readout(cohort["declaration"], cohort["complete"], replay, metrics)
    if (
        contact.get("schema") != "soccer.rsi.pre_planner_pulse_contact_complete.v1"
        or contact["report_hash"] != spec["contact_complete_hash"]
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
        or [r["group"] for r in contact.get("paired_motion_diagnostics", [])] != [28, 76, 12]
    ):
        raise ValueError("whole negative dense/motion diagnostics and unchanged labels required")
    for path, h in contact["input_file_hashes"].items():
        if digest(path) != h or (path in pins and pins[path] != h):
            raise ValueError("original negative diagnostics source or evidence changed")
        pins[path] = h
    return dict(
        negative_native_hash=cohort["complete"]["report_hash"],
        negative_readback_hash=cohort["audit"]["report_hash"],
        negative_contact_hash=contact["report_hash"],
        assessment=cohort["assessment"],
        changed_polarity_is_not_learning=True,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )
