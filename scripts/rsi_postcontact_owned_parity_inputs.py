"""Post-touch selected-window implementation parity on six original contexts.

All18 source outcomes stay authenticated; six retention windows are fixed
comparison sites, not new held-out courses. Selected offsets/gold enter only
the parity experiment and post-forecast comparison, never a deployed actor.
"""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_owned_selected_parity_inputs import bind_case
from scripts.rsi_owned_selected_parity_inputs import require_scope as old_scope
from scripts.rsi_read_postcontact_transfer import check_complete
from scripts.rsi_replay_postcontact_full_bank_physics import require_full_physical_review
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("profile") != "six-postcontact-owned-selected-parity-v1"
        or cfg.get("frame_selection") != "CLOSED_POSTCONTACT_SELECTED_ACTION_COMPARISON_ONLY"
        or cfg.get("selected_source_indices") != list(range(12, 18))
        or cfg.get("new_task_gain_claimed") is not False
    ):
        raise ValueError(
            "fixed postcontact six selected-window implementation-only parity required"
        )
    normalized = dict(
        cfg,
        profile="six-selected-owned-continuation-parity-v1",
        frame_selection="CLOSED_H300_SELECTED_ACTION_COMPARISON_ONLY",
    )
    normalized.pop("selected_source_indices")
    normalized.pop("new_task_gain_claimed")
    old_scope(normalized)


def bind_postcontact_case(case, row, checked, decision, gold, actual, *, source_index):
    names = {
        k
        for k in actual
        if k.startswith(
            (
                "postcontact_exit_",
                "postcontact_release_",
                "early_impact_feedback_",
                "impact_feedback_",
            )
        )
    }
    if (
        names
        != {
            "postcontact_exit_requested_offset_nm",
            "postcontact_exit_applied_offset_nm",
            "postcontact_exit_forecast_steps",
        }
        or type(source_index) is not int
        or not 0 <= source_index < 6
        or type(row["index"]) is not int
        or row["index"] != source_index + 12
        or row["source_index"] != source_index
        or row["partition"] != "CONSUMED_ORIGINAL_RETENTION"
        or checked["index"] != row["index"]
        or checked["source_case_hash"] != row["report_hash"]
        or decision["schema"] != "soccer.rsi.preaction_postcontact80ms_exit_decision.v1"
        or type(case["frame"]) is not int
        or type(decision["frame"]) is not int
        or decision["frame"] != case["frame"] + 1
        or not np.any(actual["force_n"][decision["frame"] - 1, 0, :2] > 1)
    ):
        raise ValueError(
            "same fixed original retention and actual completed post-touch window required"
        )
    # Adapt only metadata names. The actual exit fields are never relabeled
    # into the earlier-impact namespace; shared numeric arrays remain intact.
    bind_case(
        case,
        dict(job=case["job"], source_receipt_hash=row["source_receipt_hash"]),
        dict(job=case["job"], selected_window_alignment=checked["selected_execution_alignment"]),
        decision,
        gold,
        actual,
    )


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    for key in ("producer_unit", "readback_unit", "physical_unit"):
        require_terminal(p.cfg[key])

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in p.pins and p.pins[str(file)] != h)
        ):
            raise ValueError(
                "whole original postcontact physical/NN comparison input bytes required"
            )
        p.pins[str(file)] = h

    def sealed(name, expected=None):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole postcontact parity input seal required")
        return d

    root, physical = Path(p.cfg["producer_root"]), Path(p.cfg["physical_root"])
    c = sealed(root / "complete.json", p.cfg["producer_complete_file_hash"])
    d = sealed(root / "declaration.json")
    adaptive, _ = check_complete(c, d, root)
    rb = sealed(p.cfg["readback_path"], p.cfg["readback_file_hash"])
    pc = sealed(physical / "complete.json", p.cfg["physical_complete_file_hash"])
    pd = sealed(physical / "declaration.json")
    if (
        not adaptive
        or rb["schema"] != "soccer.rsi.postcontact_transfer_readback.v1"
        or rb["complete_hash"] != c["report_hash"]
        or len(rb["rows"]) != 18
        or pc["schema"] != "soccer.rsi.recorded_postcontact18_complete.v1"
        or pc["declaration_hash"] != pd["report_hash"]
        or len(pc["rows"]) != 18
        or pc["actual_physical_steps"] != 54000
        or pc["new_optimizer_updates"] != 0
        or pc["new_neural_inferences"] != 0
        or any(
            pc.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or pd["readback_hash"] != rb["report_hash"]
        or pd["producer_complete_hash"] != c["report_hash"]
        or d["config"]["mean_model_hash"] != p.cfg["mean_model_hash"]
        or d["config"]["stage_readback"] != p.cfg["stage_readback"]
    ):
        raise ValueError(
            "same whole18 adaptive raw/physical chain and unchanged source mean required"
        )
    for parent in (rb, pd):
        for name, h in parent["input_file_hashes"].items():
            pin(name, h)
    for i, (row, checked, pr) in enumerate(zip(c["rows"], rb["rows"], pc["rows"], strict=True)):
        if (
            row != sealed(root / f"case-{i}-complete.json")
            or pr != sealed(physical / f"case-{i}-complete.json")
            or checked["index"] != i
            or checked["source_case_hash"] != row["report_hash"]
            or pr["index"] != i
            or pr["source_case_hash"] != row["report_hash"]
        ):
            raise ValueError("all18 unchanged original source cases and recorded bodies required")
        require_full_physical_review(pr["physical_review"])
    for i, case in enumerate(p.cases):
        ci = 12 + i
        row, checked = c["rows"][ci], rb["rows"][ci]
        selections = checked["selected_execution_alignment"]
        if len(selections) != 1:
            raise ValueError(
                "exactly one complete observed postcontact window per fixed case required"
            )
        frame = selections[0]["forecast_start_frame"]
        decision = sealed(root / f"case-{ci}-preaction-evidence/frame-{frame:03d}.json")
        selected = decision["selected_index"]
        if type(selected) is not int or not 0 <= selected < len(decision["raw_witnesses"]):
            raise ValueError("actual bounded selected action index required")
        witness = decision["raw_witnesses"][selected]
        pin(witness["path"], witness["sha256"])
        pin(row["arrays"]["path"], row["arrays"]["sha256"])
        with (
            np.load(witness["path"], allow_pickle=False) as a,
            np.load(row["arrays"]["path"], allow_pickle=False) as b,
        ):
            bind_postcontact_case(case, row, checked, decision, dict(a), dict(b), source_index=i)
        case["selected_gold_witness"] = dict(witness)
    p.selected_h300_hash = c["report_hash"]
    return p
