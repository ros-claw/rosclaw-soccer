"""Independent integrity validator for team pass-handshake evidence."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.team_pass_handshake_discovery import (
    _implementation_hash,
    _select,
)


def validate_team_pass_handshake_discovery(path: Path) -> dict[str, Any]:
    """Recompute gates for either an honest PASS or REJECTED report."""

    source = path.expanduser().resolve()
    report = _bound_report(source)
    request_path = source.parent / "request.json"
    request = _read_object(request_path)
    rows = report.get("rows")
    selected = report.get("selected")
    context_hashes = request.get("context_hashes")
    if (
        not isinstance(rows, list)
        or not rows
        or not isinstance(selected, list)
        or not isinstance(context_hashes, list)
        or not all(isinstance(value, str) for value in context_hashes)
    ):
        raise ValueError("team handshake evidence rows are invalid")
    derived_selected = [_select(rows, context_hash) for context_hash in context_hashes]
    recovered = {row["context_hash"] for row in rows if row["strict_team_chain"]}
    gates = {
        "all_contexts_have_strict_handshake": recovered == set(context_hashes),
        "all_selected_safe": all(row["quality"]["safe"] for row in derived_selected),
        "all_selected_strict_team_chain": all(row["strict_team_chain"] for row in derived_selected),
        "all_selected_clear_outcome": all(
            row["quality"]["clear_outcome"] for row in derived_selected
        ),
        "finisher_actor_executed_all": all(row["finisher_actor_active"] for row in rows),
        "no_teacher_or_scripted_contact": all(
            not row["teacher_active"] and not row["scripted_contact_active"] for row in rows
        ),
        "all_trajectories_bound": all(
            hash_bytes((source.parent / row["trajectory"]["file"]).read_bytes())
            == row["trajectory"]["file_hash"]
            for row in rows
        ),
    }
    expected_status = (
        "PASS_TEAM_PASS_HANDSHAKE_DISCOVERY"
        if all(gates.values())
        else "REJECTED_TEAM_PASS_HANDSHAKE_DISCOVERY"
    )
    if (
        hash_bytes(request_path.read_bytes()) != report.get("request_hash")
        or request.get("implementation_hash") != _implementation_hash()
        or report.get("implementation_hash") != _implementation_hash()
        or report.get("gates") != gates
        or selected != derived_selected
        or report.get("status") != expected_status
        or report.get("promotion_eligible") is not False
        or report.get("physics_authority") != "CPU_MUJOCO"
        or report.get("activation_ceiling") != "SIM_ONLY"
        or report.get("hardware_command_sent") is not False
        or report.get("pixels_used_for_scoring") is not False
    ):
        raise ValueError("team handshake evidence authority is invalid")
    return report


def _bound_report(path: Path) -> dict[str, Any]:
    value = _read_object(path)
    claimed = value.pop("report_hash", None)
    if claimed != hash_json(value):
        raise ValueError("team handshake report integrity changed")
    value["report_hash"] = claimed
    return value


def _read_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object: {path}")
    return cast(dict[str, Any], value)


__all__ = ["validate_team_pass_handshake_discovery"]
