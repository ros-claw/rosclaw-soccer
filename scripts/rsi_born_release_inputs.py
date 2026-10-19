"""One consumed earlier-touch failed context, not a generalization teacher."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.born_postcontact_release import BORN_RELEASE_ACTIONS
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_full_task_branch_inputs import prepare as original_prepare
from scripts.rsi_navigation_recovery_inputs import require_closed_zero18
from scripts.rsi_stage_release_contact_replay import digest, require_terminal

ACTIONS = ["baseline", *BORN_RELEASE_ACTIONS]


def require_scope(cfg):
    fixed = dict(
        profile="full-task-born-release14-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        diagnostic_context_index=8,
        action_names=ACTIONS,
        main_reconstruction_runs=14,
        main_native_step_budget=42000,
        prediction_runs=14,
        prediction_native_step_budget=8960,
        combined_native_step_budget=50960,
        official_horizon_frames=300,
        post_contact_frames=60,
        constant_forward_approach=True,
        born_release_capture=True,
        new_optimizer_updates=0,
        new_task_gain_claimed=False,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()) or any(
        type(a) is not str for a in cfg["action_names"]
    ):
        raise ValueError(
            "whole14 declared born leases on consumedcontext8 within50960 steps required"
        )


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    require_closed_zero18(p)
    require_terminal(p.cfg["navigation_probe_unit"])
    require_terminal(p.cfg["navigation_probe_physics_unit"])

    def sealed(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in p.pins and p.pins[str(file)] != h)
        ):
            raise ValueError("closed failed navigation probe bytes required")
        p.pins[str(file)] = h
        value = json.loads(file.read_text())
        if value["report_hash"] != hash_json(
            {k: v for k, v in value.items() if k != "report_hash"}
        ):
            raise ValueError("closed failed navigation probe seal required")
        return value

    root = Path(p.cfg["navigation_probe_root"])
    physical = Path(p.cfg["navigation_probe_physics_root"])
    c = sealed(root / "complete.json", p.cfg["navigation_probe_complete_file_hash"])
    d = sealed(root / "declaration.json")
    pc = sealed(physical / "complete.json", p.cfg["navigation_probe_physics_complete_file_hash"])
    pd = sealed(physical / "declaration.json")
    if (
        c["schema"] != "soccer.rsi.navigation_recovery4_complete.v1"
        or c["declaration_hash"] != d["report_hash"]
        or len(c["rows"]) != 4
        or pc["schema"] != "soccer.rsi.navigation_recovery4_verification_complete.v1"
        or pc["declaration_hash"] != pd["report_hash"]
        or len(pc["rows"]) != 12
        or pd["producer_complete_hash"] != c["report_hash"]
        or pc["actual_physical_steps"] != 36000
        or c["teacher_qualified"] is not False
        or pc["teacher_qualified"] is not False
    ):
        raise ValueError(
            "whole failed4-context navigation probe and independent36000 steps required"
        )
    for doc in (d, pd):
        for name, h in doc["input_file_hashes"].items():
            if (
                Path(name).is_symlink()
                or digest(name) != h
                or (name in p.pins and p.pins[name] != h)
            ):
                raise ValueError("failed navigation probe input chain changed")
            p.pins[name] = h
    for gi, group in enumerate(c["rows"]):
        if len(group["actuals"]) != 3:
            raise ValueError("all navigation outcomes required")
        for ai, row in enumerate(group["actuals"]):
            proof = pc["rows"][gi * 3 + ai]
            if (
                row != sealed(root / f"case-{group['index']}-arm-{ai}-complete.json")
                or proof != sealed(physical / f"case-{gi * 3 + ai}-complete.json")
                or proof["source_case_hash"] != row["report_hash"]
                or proof["physical_review"]["actual_physical_steps"] != 3000
                or any(
                    not np.isfinite(v) or v > 1e-8
                    for v in proof["physical_review"]["maximum_absolute_errors"].values()
                )
            ):
                raise ValueError("all matched navigation-body physical proofs required")
    if [sum(g["actuals"][ai]["outcome"]["task_pass"] for g in c["rows"]) for ai in range(3)] != [
        1,
        1,
        1,
    ]:
        raise ValueError("this pivot requires its declared no-net-gain navigation result")
    p.jobs = [p.jobs[8]]
    if (
        p.jobs[0]["index"] != 8
        or (300 - p.jobs[0]["comparison_start_after_measured_foot_bin"]) * 140 != 8960
    ):
        raise ValueError("fixed earlier-touch context and fourteen future budgets required")
    return p
