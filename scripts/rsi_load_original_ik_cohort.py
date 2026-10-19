"""Read all six source-owned original-IK bodies, never run or admit a policy."""

import json
import subprocess
from pathlib import Path

from rosclaw_soccer.rsi.original_ik_planner_scope import assess_ik_scope, ik_scope, require_ik_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_read_pre_planner_navigation import read_journal
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def load_closed_original_ik(cfg, *, sealed, pins):
    for key in ("producer_unit", "audit_unit"):
        require_terminal(cfg[key])
    path = Path(cfg["producer_config"])
    if digest(path) != cfg["producer_config_file_hash"]:
        raise ValueError("original frozen IK producer configuration required")
    pins[str(path)] = digest(path)
    producer = json.loads(path.read_text())
    require_ik_scope(producer)
    source, root = Path(producer["source_root"]), Path(producer["output_root"])
    reader = source / "scripts/rsi_read_original_ik_planner.py"
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != producer["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or Path(cfg["producer_declaration"]).resolve() != (root / "declaration.json").resolve()
        or Path(cfg["producer_complete"]).resolve() != (root / "complete.json").resolve()
        or Path(cfg["audit_source"]).resolve() != reader.resolve()
    ):
        raise ValueError("original frozen IK source and owned output paths required")
    declaration, complete, audit = (
        sealed(cfg[k])
        for k in ("producer_declaration", "producer_complete", "source_terminal_readback")
    )
    scope = ik_scope(producer["behavior_model_hash"])
    if (
        declaration.get("schema") != "soccer.rsi.original_ik_planner_six_declaration.v1"
        or complete.get("schema") != "soccer.rsi.original_ik_planner_six_complete.v1"
        or audit.get("schema") != "soccer.rsi.original_ik_planner_six_terminal_readback.v1"
        or any(declaration.get(k) != v for k, v in producer.items())
        or hash_json(declaration["scope"]) != hash_json(scope)
        or complete["declaration_hash"] != declaration["report_hash"]
        or audit["complete_hash"] != complete["report_hash"]
        or audit["report_hash"] != cfg["source_terminal_readback_hash"]
        or audit["reader_source_hash"] != digest(reader)
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 6),
                ("new_independent_cpu_bodies", 6),
                ("native_substeps", 18000),
                ("independent_cpu_substeps", 18000),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            type(audit.get(k)) is not int or audit[k] != n
            for k, n in (
                ("source_native_bodies", 6),
                ("source_independent_cpu_bodies", 6),
                ("prior_actual_cpu_ik_reconstructions_authenticated", 6),
                ("new_physical_steps", 0),
                ("new_foundation_inferences", 0),
                ("optimizer_updates", 0),
            )
        )
        or any(
            d.get(k) is not False
            for d in (complete, audit)
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("whole six original-IK bodies and terminal reconstruction audit required")
    assessment = assess_ik_scope(scope, complete["results"])
    tip, state = read_journal(root, declaration)
    if (
        assessment != complete["assessment"]
        or assessment != audit["assessment"]
        or audit["ledger_tip"] != tip
        or len(audit["rows"]) != 6
        or any(
            state["completed"][j["job"]] != r
            or row["receipt_hash"] != r["receipt_hash"]
            or row["job"] != j
            or row["outcome"] != r["outcome"]
            or row["alignment"] != r["qualification_alignment"]
            or sealed(root / f"receipt-{i:04d}.json", "receipt_hash") != r
            for i, (j, r, row) in enumerate(
                zip(scope["jobs"], complete["results"], audit["rows"], strict=True)
            )
        )
    ):
        raise ValueError("all ordered IK receipts, outcomes and journal membership required")
    for d in (declaration, audit):
        for name, h in {**d["input_file_hashes"], **d.get("prepared_pins", {})}.items():
            if digest(name) != h or (name in pins and pins[name] != h):
                raise ValueError("original IK source, input or witness bytes changed")
            pins[name] = h
    pins[str(reader)] = digest(reader)
    return dict(producer=producer, declaration=declaration, complete=complete, audit=audit)
