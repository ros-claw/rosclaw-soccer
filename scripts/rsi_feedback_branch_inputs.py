"""Source/whole-six admission for zero-intervention full-feedback calibration."""

import json
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_impact_screen_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "six-zero-feedback40ms-v1"
        or cfg.get("frame_selection") != "RETROSPECTIVE_SOURCE_CONTACT_FRAME_DIAGNOSTIC_ONLY"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("zero_branches_per_case", 2),
                ("horizon_substeps", 20),
                ("recorded_replay_substep_budget", 18000),
                ("feedback_forecast_substep_budget", 240),
                ("combined_substep_budget", 18240),
                ("maximum_planner_run_calls", 60),
                ("current_encoder_decoder_pairs", 6),
                ("branch_encoder_decoder_pairs", 12),
                ("original_residual_decisions", 1800),
                ("original_recurrent_updates", 1620),
                ("new_closed_loop_policy_trials", 0),
                ("new_optimizer_updates", 0),
                ("candidate_actions_optimized", 0),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "training_authorized",
                "teacher_qualified",
                "actor_future_event_input_allowed",
            )
        )
    ):
        raise ValueError("exact six-case18240-step zero-action feedback calibration required")


def prepare(path, *, scope_validator=require_scope):
    p = original_prepare(path, scope_validator=scope_validator)
    require_terminal(p.cfg["restore_unit"])
    folder = Path(p.cfg["restore_root"])
    complete_path = folder / "complete.json"
    if digest(complete_path) != p.cfg["restore_complete_file_hash"]:
        raise ValueError("original real-model restoration result bytes required")

    def sealed(path):
        if path.is_symlink():
            raise ValueError("ordinary predecessor evidence required")
        d = json.loads(path.read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole predecessor logical seal required")
        h = digest(path)
        if str(path) in p.pins and p.pins[str(path)] != h:
            raise ValueError("source authentication chain conflicts")
        p.pins[str(path)] = h
        return d

    complete, declaration = sealed(complete_path), sealed(folder / "declaration.json")
    if (
        complete["report_hash"] != p.cfg["restore_complete_hash"]
        or complete["schema"] != "soccer.rsi.causal_tracker_restore_complete.v1"
        or complete["declaration_hash"] != declaration["report_hash"]
        or complete["all_six_original_cases_retained"] is not True
        or len(complete["rows"]) != 6
        or any(
            type(complete[k]) is not int or complete[k] != n
            for k, n in (
                ("planner_run_calls", 60),
                ("current_encoder_decoder_pairs", 6),
                ("next_offline_golden_encoder_decoder_pairs", 6),
                ("new_physical_steps", 0),
                ("new_closed_loop_policy_trials", 0),
                ("new_optimizer_updates", 0),
            )
        )
        or any(
            complete[k] is not False
            for k in (
                "complete_feedback_controller_qualified",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError("complete source-model restore predecessor required")
    for name, h in declaration["input_file_hashes"].items():
        if (name in p.pins and p.pins[name] != h) or digest(name) != h:
            raise ValueError("whole original restore input/model/source chain changed")
        p.pins[name] = h
    for i, (row, case) in enumerate(zip(complete["rows"], p.cases, strict=True)):
        if (
            sealed(folder / f"case-{i}-complete.json") != row
            or row["job"] != case["job"]
            or row["source_receipt_hash"] != case["receipt"]["receipt_hash"]
            or any(
                row[k] is not True
                for k in (
                    "current_model_call_and_history_bit_exact",
                    "next_offline_golden_model_call_bit_exact",
                    "restored_prefix_ends_at_current_frame",
                )
            )
        ):
            raise ValueError("all original ordered restoration cases required")
        if case["frame"] is None or not 30 <= case["frame"] < 298 or (case["frame"] + 1) % 20 == 0:
            raise ValueError("complete no-replan branch window required for every original case")
    mean = Path(p.cfg["mean_model_path"])
    if (
        str(mean) not in p.pins
        or digest(mean) != p.pins[str(mean)]
        or p.cfg["mean_model_hash"] != p.declaration["behavior_model_hash"]
    ):
        raise ValueError("same fully authenticated original fixed mean required")
    p.model_root = Path(declaration["config"]["model_root"])
    p.restore = complete
    return p
