"""Whole-six H300 development diagnostic admission, not teacher acceptance."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_calibrate_causal_impact_replay import compare_frame
from scripts.rsi_early_feedback_screen_inputs import validate_readback as validate_zero
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "six-prospective-early-impact-h300-diagnostic-v1"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("new_causal_rollouts", 6),
                ("reused_original_controls", 6),
                ("native_step_budget", 18000),
                ("forecast_step_budget", 10800),
                ("combined_step_budget", 28800),
                ("official_horizon_frames", 300),
                ("post_contact_frames", 60),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "teacher_qualified",
                "training_authorized",
                "private_fresh_accessed",
                "actor_future_event_input_allowed",
                "independent_qualification_claimed",
                "new_neural_policy_trained",
            )
        )
    ):
        raise ValueError("exact whole6 prospective H300/28800-step development diagnostic required")


def validate_screen(d, cases):
    if (
        d.get("schema") != "soccer.rsi.early_feedback_impact_screen_readback.v1"
        or d.get("all_original_six_including_failures_retained") is not True
        or d.get("candidate_jacobian_bank_independently_regenerated") is not True
        or d.get("score_formula_shared_with_producer") is not True
        or d.get("candidate_geometry_and_bank_formula_shared_with_producer") is not True
        or type(d.get("rows")) is not list
        or len(d["rows"]) != 6
        or len(cases) != 6
        or type(d.get("raw_branch_count")) is not int
        or not 12 <= d["raw_branch_count"] <= 84
        or type(d.get("nonzero_candidates")) is not int
        or not 0 <= d["nonzero_candidates"] <= 72
        or d["raw_branch_count"] != d["nonzero_candidates"] + 12
        or type(d.get("local_proxy_improved_source_cases")) is not int
        or any(
            type(d.get(k)) is not int or d[k] != 0
            for k in ("new_physical_steps", "new_optimizer_updates")
        )
        or any(
            d.get(k) is not False
            for k in (
                "nonzero_branch_independent_physics_replayed",
                "prospective_causal_trigger_qualified",
                "full_m0_gain_proven",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("complete unqualified whole6 early80ms raw screen readback required")
    promising_failure = False
    for i, (row, case) in enumerate(zip(d["rows"], cases, strict=True)):
        if (
            type(row.get("index")) is not int
            or row["index"] != i
            or row.get("job") != case["job"]
            or row.get("source_receipt_hash") != case["receipt"]["receipt_hash"]
            or row.get("original_outcome") != case["receipt"]["outcome"]
            or row.get("all_raw_branch_scores_offsets_and_feedback_targets_checked") is not True
            or row.get("zero_branches_match_original_dense_and_neural_bits") is not True
            or type(row.get("local_proxy_improved")) is not bool
        ):
            raise ValueError("all unchanged ordered screen contexts including failures required")
        promising_failure |= bool(
            row["local_proxy_improved"] and not case["receipt"]["outcome"]["task_pass"]
        )
    if not promising_failure:
        raise ValueError("no original failure has local authority improvement; do not blind-pilot")
    if d["local_proxy_improved_source_cases"] != sum(r["local_proxy_improved"] for r in d["rows"]):
        raise ValueError("complete screen improvement counts differ from all six rows")


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)

    def sealed(name, expected=None):
        file = Path(name)
        h = digest(file)
        if file.is_symlink() or (expected is not None and h != expected):
            raise ValueError("same ordinary predecessor evidence bytes required")
        if str(file) in p.pins and p.pins[str(file)] != h:
            raise ValueError("predecessor identity conflicts with original proof chain")
        d = json.loads(file.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("complete predecessor logical seal required")
        p.pins[str(file)] = h
        return d

    def absorb(d):
        for name, h in d["input_file_hashes"].items():
            if (name in p.pins and p.pins[name] != h) or digest(name) != h:
                raise ValueError("whole predecessor source/model/data authentication changed")
            p.pins[name] = h

    for key in (
        "early_zero_unit",
        "early_zero_readback_unit",
        "screen_unit",
        "screen_readback_unit",
    ):
        require_terminal(p.cfg[key])
    zero = sealed(p.cfg["early_zero_readback"], p.cfg["early_zero_readback_file_hash"])
    validate_zero(zero, p.cfg, p.cases)
    absorb(zero)
    screen = sealed(p.cfg["screen_readback"], p.cfg["screen_readback_file_hash"])
    if screen["report_hash"] != p.cfg["screen_readback_hash"]:
        raise ValueError("same completed early80ms screen readback required")
    validate_screen(screen, p.cases)
    absorb(screen)
    require_terminal(p.cfg["causal_calibration_unit"])
    root = Path(p.cfg["causal_calibration_root"])
    complete = sealed(root / "complete.json", p.cfg["causal_calibration_complete_file_hash"])
    declaration = sealed(root / "declaration.json")
    if (
        complete["report_hash"] != p.cfg["causal_calibration_complete_hash"]
        or complete["schema"] != "soccer.rsi.full_causal_replay_calibration_complete.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("actual_native_steps", 18000),
                ("forecast_steps", 0),
                ("new_policy_candidates", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or complete["impact_enabled_false_only"] is not True
        or complete["all_six_source_outcomes_and_neural_bits_matched"] is not True
        or len(complete["rows"]) != 6
        or any(
            complete[k] is not False
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("whole original actual G1 impact-OFF calibration required")
    absorb(declaration)
    for i, (row, case) in enumerate(zip(complete["rows"], p.cases, strict=True)):
        if (
            sealed(root / f"case-{i}-complete.json") != row
            or type(row["index"]) is not int
            or row["index"] != i
            or row["source_receipt_hash"] != case["receipt"]["receipt_hash"]
            or row["job"] != case["job"]
            or row["original_outcome"] != case["receipt"]["outcome"]
            or row["causal_replay_outcome"] != case["receipt"]["outcome"]
            or row["all300_frames_and_all_current_neural_bits_matched"] is not True
            or row["causal_review"]["current_body_policy_recomputed"] is not True
            or row["causal_review"]["source_future_trace_or_targets_consumed"] is not False
            or type(row["causal_review"]["native_integrated_steps"]) is not int
            or row["causal_review"]["native_integrated_steps"] != 3000
        ):
            raise ValueError("all original calibration cases including failures required")
        witness = row["arrays"]
        if digest(witness["path"]) != witness["sha256"]:
            raise ValueError("whole original causal calibration arrays required")
        p.pins[witness["path"]] = witness["sha256"]
        with np.load(witness["path"], allow_pickle=False) as a:
            actual = dict(a)
        if any(len(v) != 300 for v in actual.values()):
            raise ValueError("all300 original causal calibration frames required")
        for frame in range(300):
            compare_frame({k: v[frame] for k, v in actual.items()}, case["trace"], frame)
    p.early_zero = zero
    p.early_screen = screen
    p.causal_calibration = complete
    return p
