"""Whole-H300 action labels on all18 consumed contexts, not a runtime oracle."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.postcontact_release_control import ACTION_NAMES
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_postcontact_release_transfer_inputs import prepare as original_prepare
from scripts.rsi_replay_postcontact_authority import check_complete, check_row
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    fixed = dict(
        profile="postcontact-full-h300-bank252-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        comparison_contexts=18,
        comparison_rollouts=252,
        reused_authority_context_index=1,
        reused_authority_rollouts=14,
        new_rollouts=238,
        native_step_budget=714000,
        forecast_step_budget=0,
        official_horizon_frames=300,
        post_contact_frames=60,
        maximum_release_duration_frames=3,
        new_optimizer_updates=0,
        baseline_rollouts_reused=18,
        private_fresh_accessed=False,
        teacher_qualified=False,
        training_authorized=False,
        hardware_authorized=False,
        promotion_authorized=False,
        automatic_retry=False,
    )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()) or cfg.get(
        "action_names"
    ) != [*ACTION_NAMES, "baseline"]:
        raise ValueError(
            "exact all18 full-H300 action-bank252 scope with14 reused/238new bodies required"
        )


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)
    require_terminal(p.cfg["authority_unit"])
    root = Path(p.cfg["authority_root"])

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and expected != h)
            or (str(file) in p.pins and p.pins[str(file)] != h)
        ):
            raise ValueError("same whole authority input/source bytes required")
        p.pins[str(file)] = h

    def sealed(name, expected=None):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole reused authority seal required")
        return d

    c = sealed(root / "complete.json", p.cfg["authority_complete_file_hash"])
    d = sealed(root / "declaration.json")
    check_complete(c, d, root)
    for name, h in d["input_file_hashes"].items():
        pin(name, h)
    pd = sealed(Path(p.cfg["release_physics_root"]) / "declaration.json")
    index = p.cfg["reused_authority_context_index"]
    job = p.jobs[index]
    case = p.cases[job["source_index"]]
    cw = case["receipt"]["witnesses"]["commitment"]
    cm = json.loads(Path(cw["path"]).read_text())
    if (
        pd["producer_complete_hash"] != c["report_hash"]
        or d["course"] != job["course"]
        or d["original_source_receipt_hash"] != case["receipt"]["receipt_hash"]
        or job["source_index"] != 0
        or job["index"] != 1
    ):
        raise ValueError("same independently replayed source authority context required")
    for i, row in enumerate(c["rows"]):
        if row != sealed(root / f"trial-{i}-complete.json"):
            raise ValueError("all14 reused authority trials including failures required")
        w = row["arrays"]
        pin(w["path"], w["sha256"])
        with np.load(w["path"], allow_pickle=False) as a:
            trace = dict(a)
        check_row(
            row,
            trace,
            job["baseline_trace"],
            policy_hash=cm["motor_policy_hash"],
            receipt_hash=case["receipt"]["receipt_hash"],
        )
    p.reused_authority, p.reused_authority_hash = c["rows"], c["report_hash"]
    return p
