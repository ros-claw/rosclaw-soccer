"""Whole original binding plus one CLOSED consumed failure, not teacher admission."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_local_impact_robustness import require_scope as local_scope
from scripts.rsi_read_local_impact_robustness import verify_original_control_torques
from scripts.rsi_stage_release_contact_replay import digest


def require_scope(cfg):
    fixed = dict(
        mode="SIM_ONLY",
        profile="postcontact-release-native14-diagnostic-v1",
        source_cases=6,
        local_context_index=1,
        original_binding_index=0,
        diagnostic_physical_contexts=1,
        new_rollouts=14,
        native_step_budget=42000,
        forecast_step_budget=0,
        executed_control_foundation_pairs=4200,
        official_horizon_frames=300,
        post_contact_frames=60,
        maximum_release_duration_frames=3,
        source_campaign_completion_not_required=True,
        new_optimizer_updates=0,
        private_fresh_accessed=False,
        training_authorized=False,
        teacher_qualified=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()) or cfg.get(
        "action_names"
    ) != [*ACTION_NAMES, "baseline"]:
        raise ValueError("fixed single-context full14 postcontact authority diagnostic required")


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and expected != h)
            or (str(file) in p.pins and p.pins[str(file)] != h)
        ):
            raise ValueError("same closed local source bytes required")
        p.pins[str(file)] = h
        return file

    def sealed(name, expected=None):
        d = json.loads(pin(name, expected).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole closed local source seal required")
        return d

    root = Path(p.cfg["local_root"])
    declaration = sealed(root / "declaration.json", p.cfg["local_declaration_file_hash"])
    local_scope(declaration["config"])
    if declaration["config"]["output_root"] != str(root):
        raise ValueError("same local campaign source required")
    for name, h in declaration["input_file_hashes"].items():
        pin(name, h)
    courses = []
    for case in p.cases:
        w = case["receipt"]["witnesses"]["commitment"]
        courses.append(tuple(json.loads(pin(w["path"], w["sha256"]).read_text())["course"]))
    if declare_local_impact_perturbations(tuple(courses)) != declaration["plan"]:
        raise ValueError("whole original outcome-independent local plan required")
    index = p.cfg["local_context_index"]
    pair = sealed(root / f"pair-{index}-complete.json", p.cfg["local_pair_file_hash"])
    job = declaration["plan"]["jobs"][index]
    if (
        pair["index"] != index
        or pair["job"] != job
        or job["source_index"] != p.cfg["original_binding_index"]
        or len(pair["arms"]) != 2
    ):
        raise ValueError("one complete declared consumed diagnostic pair required")
    case = p.cases[job["source_index"]]
    original = pair["arms"][0]
    if (
        original != sealed(root / f"pair-{index}-original_control-complete.json")
        or original["arm"] != "original_control"
        or original["job"] != job
        or original["source_receipt_hash"] != case["receipt"]["receipt_hash"]
        or original["owned_continuation_enabled"] is not False
    ):
        raise ValueError("same closed original-control body required")
    file = pin(original["arrays"]["path"], original["arrays"]["sha256"])
    with np.load(file, allow_pickle=False) as a:
        trace = dict(a)
    verify_original_control_torques(trace)
    outcome = label_first_touch(trace).to_dict()
    if (
        outcome != original["outcome"]
        or outcome["task_pass"] is not False
        or outcome["safe"] is not True
        or outcome["quality_observed"] is not True
        or outcome["failure_reason"] != "BALL_OUTCOME_FAILED"
    ):
        raise ValueError(
            "closed observed safe ball-outcome failure required, not unsafe/censored teacher"
        )
    w = case["receipt"]["witnesses"]["commitment"]
    commitment = json.loads(Path(w["path"]).read_text())
    if (
        original["original_policy_hash"] != commitment["motor_policy_hash"]
        or original["review"]["native_integrated_steps"] != 3000
        or original["review"]["forecast_integrated_steps"] != 0
    ):
        raise ValueError("same original complete motor identity and physical execution required")
    p.case, p.course, p.original_trace, p.original_outcome = (
        case,
        tuple(job["course"]),
        trace,
        outcome,
    )
    p.local_pair_hash = pair["report_hash"]
    return p
