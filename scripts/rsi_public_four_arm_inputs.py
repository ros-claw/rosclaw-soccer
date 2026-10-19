"""Admit the predeclared forty public courses, never sealed Fresh."""

import json
from pathlib import Path

from rosclaw_soccer.rsi.postcontact_robustness_plan import declare_robustness40
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    fixed = dict(
        profile="public-postcontact-four-arm40-v1",
        mode="SIM_ONLY",
        source_cases=6,
        new_rollouts=160,
        native_step_budget=480000,
        forecast_step_budget=44800,
        official_horizon_frames=300,
        post_contact_frames=60,
        new_optimizer_updates=0,
        baseline_rollouts_reused=0,
        reuse_owned_postcontact_updates=True,
        constant_forward_approach=False,
        collection_workers=1,
        teacher_qualified=False,
        training_authorized=False,
        private_fresh_accessed=False,
        promotion_authorized=False,
        hardware_authorized=False,
        automatic_retry=False,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("whole public40 four-arm serial owned-body diagnostic required")


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    # Every new course preflight completes before any body/factory allocation.
    p.plan = declare_robustness40(mean_model_hash=p.cfg["mean_model_hash"])
    require_terminal(p.cfg["owned_parity_unit"])

    def pin(name, expected=None):
        f = Path(name)
        h = digest(f)
        if (
            f.is_symlink()
            or (expected is not None and h != expected)
            or (str(f) in p.pins and p.pins[str(f)] != h)
        ):
            raise ValueError("same whole selected-owned comparison proof required")
        p.pins[str(f)] = h

    def sealed(name, expected=None):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole selected-owned proof seal required")
        return d

    root = Path(p.cfg["owned_parity_root"])
    c = sealed(root / "complete.json", p.cfg["owned_parity_complete_file_hash"])
    d = sealed(root / "declaration.json")
    rb = sealed(p.cfg["owned_parity_readback"], p.cfg["owned_parity_readback_file_hash"])
    if (
        c["declaration_hash"] != d["report_hash"]
        or len(c["rows"]) != 6
        or c["all_six_selected_action_forecasts_match_closed_h300_gold"] is not True
        or c["forecast_integrated_steps"] != 480
        or c["new_optimizer_updates"] != 0
        or rb["schema"] != "soccer.rsi.postcontact_owned_selected_full_readback.v1"
        or rb["source_complete_hash"] != c["report_hash"]
        or rb["six_cases_all_raw_fields_match_gold"] is not True
        or rb["full14_bank_or_realtime_controller_speedup_proven"] is not False
        or rb["teacher_qualified"] is not False
    ):
        raise ValueError("closed selected-window parity, not full teacher admission, required")
    for proof in (d, rb):
        for n, h in proof["input_file_hashes"].items():
            pin(n, h)
    for i, row in enumerate(c["rows"]):
        if row != sealed(root / f"case-{i}-complete.json"):
            raise ValueError("all six original ordered selected-window proofs required")
    p.owned_parity_hash = c["report_hash"]
    return p
