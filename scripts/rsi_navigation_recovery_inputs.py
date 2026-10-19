"""Admit nonzero navigation parity only after closed whole18 zero readback."""

import json
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_full_task_branch_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal

CONTEXTS = [0, 8, 10, 12]
ACTIONS = ["baseline", "nav_brake_0_7", "nav_brake_0"]


def require_scope(cfg):
    fixed = dict(
        profile="full-task-navigation-recovery4-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        diagnostic_context_indices=CONTEXTS,
        action_names=ACTIONS,
        main_reconstruction_runs=12,
        main_native_step_budget=36000,
        prediction_runs=16,
        prediction_native_step_budget=14320,
        combined_native_step_budget=50320,
        official_horizon_frames=300,
        post_contact_frames=60,
        constant_forward_approach=True,
        new_optimizer_updates=0,
        new_task_gain_claimed=False,
        teacher_qualified=False,
        training_authorized=False,
        private_fresh_accessed=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )
    if (
        any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items())
        or any(type(i) is not int for i in cfg["diagnostic_context_indices"])
        or any(type(a) is not str for a in cfg["action_names"])
    ):
        raise ValueError("fixed four consumed contexts, three actions,50320 native steps required")


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    require_terminal(p.cfg["zero_parity_unit"])
    require_terminal(p.cfg["zero_readback_unit"])

    def sealed(name, expected):
        file = Path(name)
        h = digest(file)
        if file.is_symlink() or h != expected or (str(file) in p.pins and p.pins[str(file)] != h):
            raise ValueError("whole closed zero prerequisite bytes required")
        p.pins[str(file)] = h
        value = json.loads(file.read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("whole closed zero prerequisite seal required")
        return value

    complete = sealed(
        Path(p.cfg["zero_parity_root"]) / "complete.json", p.cfg["zero_complete_file_hash"]
    )
    readback = sealed(p.cfg["zero_readback"], p.cfg["zero_readback_file_hash"])
    closure = sealed(p.cfg["zero_reader_closure"], p.cfg["zero_reader_closure_file_hash"])
    if (
        complete["schema"] != "soccer.rsi.full_task_zero18_complete.v1"
        or complete["combined_new_native_steps"] != 94840
        or complete["nonzero_action_prediction_qualified"] is not False
        or complete["teacher_qualified"] is not False
        or len(complete["rows"]) != 18
        or readback["schema"] != "soccer.rsi.full_task_zero18_raw_readback.v1"
        or readback["producer_complete_hash"] != complete["report_hash"]
        or len(readback["rows"]) != 18
        or closure["readback_hash"] != p.cfg["zero_readback_file_hash"]
        or closure["source_pinned_before_project_import"] is not True
    ):
        raise ValueError("actual whole18 zero plus independent raw/source closure required")
    for group in (readback["input_file_hashes"], closure["input_source_pins"]):
        for name, h in group.items():
            file = Path(name)
            if (
                file.is_symlink()
                or digest(file) != h
                or (str(file) in p.pins and p.pins[str(file)] != h)
            ):
                raise ValueError("zero proof source/input pins changed")
            p.pins[str(file)] = h
    p.jobs = [p.jobs[i] for i in CONTEXTS]
    if sum((300 - j["comparison_start_after_measured_foot_bin"]) * 40 for j in p.jobs) != 14320:
        raise ValueError("exact declared four-context prediction budget required")
    return p
