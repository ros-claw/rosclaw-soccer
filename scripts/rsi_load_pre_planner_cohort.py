"""Authenticate an original six-body pulse cohort, never execute it again."""

import json
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import require_pulse_cohort
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
