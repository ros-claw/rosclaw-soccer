"""Authenticate reused receding evidence; never elevate a single success to a teacher."""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.first_touch_qualification import require_first_touch_qualification
from rosclaw_soccer.sim.contracts import hash_json


def authenticate_receipt(
    receipt: dict[str, Any],
    *,
    world_hash: str,
    sealed: Callable[[Path], dict[str, Any]],
    digest: Callable[[Path], str],
    pins: dict[str, str],
    zero_cerebellar: bool = False,
) -> None:
    if (
        receipt["receipt_hash"]
        != hash_json({k: v for k, v in receipt.items() if k != "receipt_hash"})
        or receipt["witnesses"]["world"]["sha256"] != world_hash
    ):
        raise ValueError("whole unchanged original receipt and world required")
    for witness in receipt["witnesses"].values():
        path = Path(witness["path"])
        if digest(path) != witness["sha256"]:
            raise ValueError("original witness bytes changed")
        pins[str(path)] = witness["sha256"]
    with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as a:
        trace = dict(a)
    outcome = label_first_touch(trace)
    alignment = require_first_touch_qualification(
        sealed(Path(receipt["witnesses"]["independent_review"]["path"])),
        sealed(Path(receipt["witnesses"]["foundation_review"]["path"])),
        outcome,
        **(
            dict(
                cerebellar_model_hash=receipt["cerebellar_model_hash"], cerebellar_sampling_draws=0
            )
            if zero_cerebellar
            else {}
        ),
    )
    if outcome.to_dict() != receipt["outcome"] or alignment != receipt["qualification_alignment"]:
        raise ValueError("original task and independent CPU qualification differ")


def load_scope_evidence(
    cfg: dict[str, Any],
    sealed: Callable[[Path], dict[str, Any]],
    digest: Callable[[Path], str],
    pins: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    proof_path, contact_path = (
        Path(cfg[k]) for k in ("archived_expert_readback", "archived_expert_contact_metrics")
    )
    proof, contact = sealed(proof_path), sealed(contact_path)
    if (
        proof["report_hash"] != cfg["archived_expert_readback_hash"]
        or proof.get("schema") != "soccer.rsi.archived_receding_expert_readback.v1"
        or proof.get("actual_all3000_cerebellar_outputs_exactly_zero") is not True
        or proof.get("teacher_qualified") is not False
        or contact["report_hash"] != cfg["archived_expert_contact_metrics_hash"]
        or contact.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
    ):
        raise ValueError("exact archived zero expert and whole-body dense readback required")
    for doc in (proof, contact):
        for name, h in doc["input_file_hashes"].items():
            path = Path(name)
            if digest(path) != h:
                raise ValueError("archived evidence input bytes changed")
            pins[name] = h
    rows = [
        r for r in contact["rows"] if r["source_receipt_hash"] == proof["recovered_receipt_hash"]
    ]
    if (
        len(rows) != 1
        or rows[0]["original_outcome"] != proof["recovered_outcome"]
        or rows[0]["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        or not np.isfinite(rows[0]["metrics"]["maximum_linear_newton_residual_n"])
        or rows[0]["metrics"]["maximum_linear_newton_residual_n"] > 1e-7
    ):
        raise ValueError("archived expert needs its actual clean whole-body ball-contact review")
    old_complete = sealed(Path(cfg["archived_expert_complete"]))
    archived = old_complete["results"][7]
    if archived["receipt_hash"] != proof["recovered_receipt_hash"]:
        raise ValueError("exact original zero-control C7, not another selected success, required")
    screen, audit = (
        sealed(Path(cfg[k])) for k in ("baseline_complete", "baseline_terminal_readback")
    )
    if (
        audit["report_hash"] != cfg["baseline_terminal_readback_hash"]
        or audit.get("complete_hash") != screen["report_hash"]
        or audit.get("source_native_bodies") != 18
        or len(screen["results"]) != 18
        or audit.get("schema") != "soccer.rsi.mesh_option4_teacher_screen_terminal_readback.v1"
        or audit.get("reader_source_hash") != digest(Path(cfg["baseline_reader_source"]))
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
            )
        )
    ):
        raise ValueError("whole original baseline terminal audit required")
    controls = []
    for index, group in ((12, 76), (14, 77)):
        receipt = screen["results"][index]
        if (
            receipt["group"] != group
            or receipt["candidate"] is not None
            or not any(
                r["receipt_hash"] == receipt["receipt_hash"]
                and r["outcome"] == receipt["outcome"]
                and r["job"] == receipt["job"]
                for r in audit["rows"]
            )
        ):
            raise ValueError("both exact original None receding controls required")
        controls.append(dict(role="control", pair_id=f"receding-g{group}", receipt=receipt))
    for receipt, zero in [(i["receipt"], False) for i in controls] + [(archived, True)]:
        authenticate_receipt(
            receipt,
            world_hash=cfg["mesh_task_contact_world_hash"],
            sealed=sealed,
            digest=digest,
            pins=pins,
            zero_cerebellar=zero,
        )
    for key in (
        "archived_expert_readback",
        "archived_expert_contact_metrics",
        "archived_expert_complete",
        "baseline_complete",
        "baseline_terminal_readback",
        "baseline_reader_source",
    ):
        pins[cfg[key]] = digest(Path(cfg[key]))
    return controls, archived
