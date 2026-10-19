"""Admission for80ms local authority screen, blocked until zero readback exists."""

import json
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_early_feedback_calibration import prepare as prepare_early
from scripts.rsi_early_feedback_calibration import require_scope as zero_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    fixed = dict(
        profile="six-early-feedback-impact-screen-v1",
        maximum_candidate_actions_per_case=13,
        maximum_nonzero_candidate_actions=72,
        repeat_zero_after_candidates=True,
        feedback_forecast_substep_budget=3360,
        combined_substep_budget=21360,
        maximum_branch_encoder_decoder_pairs=252,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("exact bounded six-case21360-step early-feedback screen required")
    normalized = dict(cfg)
    normalized.update(
        profile="six-zero-early-feedback80ms-v1",
        feedback_forecast_substep_budget=480,
        combined_substep_budget=18480,
        branch_encoder_decoder_pairs=36,
        candidate_actions_optimized=0,
    )
    zero_scope(normalized)


def validate_readback(d, cfg, cases):
    if (
        d.get("schema") != "soccer.rsi.zero_early_feedback_calibration_readback.v1"
        or d.get("report_hash") != cfg["early_zero_readback_hash"]
        or d.get("report_hash") != hash_json({k: v for k, v in d.items() if k != "report_hash"})
        or d.get("all_six_no_intervention_feedback_branches_qualified") is not True
        or type(d.get("rows")) is not list
        or len(d["rows"]) != 6
        or len(cases) != 6
        or any(
            type(d.get(k)) is not int or d[k] != 0
            for k in ("new_physical_steps", "new_optimizer_updates")
        )
        or any(
            d.get(k) is not False
            for k in (
                "nonzero_interventions_qualified",
                "full_h300_teacher_qualified",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("whole independent80ms ZERO readback required before any screen")
    for i, (row, case) in enumerate(zip(d["rows"], cases, strict=True)):
        if (
            type(row.get("index")) is not int
            or row["index"] != i
            or row.get("job") != case["job"]
            or row.get("source_receipt_hash") != case["receipt"]["receipt_hash"]
            or row.get("original_outcome") != case["receipt"]["outcome"]
            or any(
                row.get(k) is not True
                for k in (
                    "all40_substeps_match_original_dense",
                    "all_three_neural_and_target_updates_verified",
                    "all_neural_next_capture_bits_match",
                    "next_target_decomposition_matches",
                    "repeated_whole_branch_bytes_equal",
                )
            )
        ):
            raise ValueError("all six unchanged original80ms contexts including failures required")


def prepare(path):
    p = prepare_early(path, scope_validator=require_scope)
    require_terminal(p.cfg["early_zero_unit"])
    require_terminal(p.cfg["early_zero_readback_unit"])
    rp = Path(p.cfg["early_zero_readback"])
    if rp.is_symlink() or digest(rp) != p.cfg["early_zero_readback_file_hash"]:
        raise ValueError("exact independent80ms zero readback bytes required")
    d = json.loads(rp.read_text())
    validate_readback(d, p.cfg, p.cases)
    if type(d.get("input_file_hashes")) is not dict or not d["input_file_hashes"]:
        raise ValueError("complete zero readback authentication chain required")
    for name, h in d["input_file_hashes"].items():
        if (name in p.pins and p.pins[name] != h) or digest(name) != h:
            raise ValueError("original80ms zero proof/model/source changed")
        p.pins[name] = h
    if str(rp) in p.pins and p.pins[str(rp)] != digest(rp):
        raise ValueError("zero readback identity conflicts with prior source pins")
    p.pins[str(rp)] = digest(rp)
    p.early_zero_readback = d
    return p
