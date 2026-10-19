"""Admission and action bank for a fixed whole-six retrospective torque screen."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.physics.g1_joint_task_kinematics import G1JointTaskKinematicSampler
from rosclaw_soccer.rsi.impact_torque_screen import impact_torque_candidates
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_feedback_branch_inputs import prepare as prepare_original
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "six-feedback-impact-screen-v1"
        or cfg.get("frame_selection") != "RETROSPECTIVE_SOURCE_CONTACT_FRAME_DIAGNOSTIC_ONLY"
        or cfg.get("repeat_zero_after_candidates") is not True
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("maximum_candidate_actions_per_case", 13),
                ("horizon_substeps", 20),
                ("recorded_replay_substep_budget", 18000),
                ("feedback_forecast_substep_budget", 1680),
                ("combined_substep_budget", 19680),
                ("maximum_planner_run_calls", 60),
                ("current_encoder_decoder_pairs", 6),
                ("maximum_branch_encoder_decoder_pairs", 84),
                ("original_residual_decisions", 1800),
                ("original_recurrent_updates", 1620),
                ("new_closed_loop_policy_trials", 0),
                ("new_optimizer_updates", 0),
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
        raise ValueError("exact bounded19680-step whole6 local feedback screen required")


def prepare(path):
    p = prepare_original(path, scope_validator=require_scope)
    require_terminal(p.cfg["zero_calibration_unit"])
    readback_path = Path(p.cfg["zero_calibration_readback"])
    if (
        readback_path.is_symlink()
        or digest(readback_path) != p.cfg["zero_calibration_readback_file_hash"]
    ):
        raise ValueError("original complete independent zero-feedback bytes required")
    d = json.loads(readback_path.read_text())
    if (
        d["report_hash"] != p.cfg["zero_calibration_readback_hash"]
        or d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"})
        or d["schema"] != "soccer.rsi.zero_feedback_calibration_readback.v1"
        or d["all_six_no_intervention_feedback_branches_qualified"] is not True
        or len(d["rows"]) != 6
        or any(
            type(d.get(k)) is not int or d[k] != 0
            for k in ("new_physical_steps", "new_optimizer_updates")
        )
        or any(
            d[k] is not False
            for k in (
                "nonzero_interventions_qualified",
                "full_h300_teacher_qualified",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or [r["source_receipt_hash"] for r in d["rows"]]
        != [c["receipt"]["receipt_hash"] for c in p.cases]
        or [r["job"] for r in d["rows"]] != [c["job"] for c in p.cases]
    ):
        raise ValueError("same complete six original zero-feedback contexts required")
    for name, h in d["input_file_hashes"].items():
        if (name in p.pins and p.pins[name] != h) or digest(name) != h:
            raise ValueError("whole source/model/zero feedback proof changed")
        p.pins[name] = h
    p.pins[str(readback_path)] = digest(readback_path)
    p.zero_readback = d
    return p


def branch_bank(model, data, ik, world_hash):
    """Only current actual causal IK swing side; no future contact-foot input."""
    eligible = type(ik.side) is int and ik.side in (0, 1) and ik.contact is None
    if eligible:
        sample = G1JointTaskKinematicSampler(model, expected_world_hash=world_hash).sample(data)
        bank = impact_torque_candidates(sample, ik.side)
    else:
        bank = [("baseline", np.zeros(29))]
    return [*bank, ("repeat_zero", np.zeros(29))], eligible


def best_proxy_index(predictions):
    feasible = [
        (i, p)
        for i, p in enumerate(predictions)
        if p["name"] != "repeat_zero" and p["score"]["feasible_predicted_impact"]
    ]
    return None if not feasible else min(feasible, key=lambda row: row[1]["score"]["proxy_cost"])[0]
