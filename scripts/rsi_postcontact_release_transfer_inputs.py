"""Admit all consumed public local contexts plus the six original retentions."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.rsi.local_impact_perturbation_plan import declare_local_impact_perturbations
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_feedback_branch_inputs import prepare as original_prepare
from scripts.rsi_read_local_impact_robustness import verify_original_control_torques
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    fixed = dict(
        profile="postcontact-fixed-release-transfer18-v1",
        mode="SIM_ONLY",
        action_name="forward_minus",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        new_rollouts=18,
        native_step_budget=54000,
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
    if cfg.get("profile") == "postcontact-exit-proxy-transfer18-v1":
        fixed.update(
            profile="postcontact-exit-proxy-transfer18-v1",
            action_name="CURRENT_BODY_EXIT_PROXY",
            forecast_step_budget=10080,
            postcontact_exit_enabled=True,
            maximum_forecasts_per_episode=14,
        )
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("fixed whole18 declared local transfer and retention profile required")


def prepare(path):
    p = original_prepare(path, scope_validator=require_scope)

    def pin(name, expected=None):
        file = Path(name)
        h = digest(file)
        if (
            file.is_symlink()
            or (expected is not None and h != expected)
            or (str(file) in p.pins and p.pins[str(file)] != h)
        ):
            raise ValueError("whole unchanged ordinary transfer predecessor required")
        p.pins[str(file)] = h

    def sealed(name, expected=None):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("whole transfer input seal required")
        return d

    require_terminal(p.cfg["local_unit"])
    require_terminal(p.cfg["release_physics_unit"])
    local = Path(p.cfg["local_root"])
    c = sealed(local / "complete.json", p.cfg["local_complete_file_hash"])
    d = sealed(local / "declaration.json")
    readback = sealed(p.cfg["local_readback"], p.cfg["local_readback_file_hash"])
    for name, h in readback["input_file_hashes"].items():
        pin(name, h)
    if (
        c["schema"] != "soccer.rsi.local_impact_robustness_complete.v1"
        or c["declaration_hash"] != d["report_hash"]
        or len(c["rows"]) != 12
        or readback["complete_hash"] != c["report_hash"]
    ):
        raise ValueError("closed full local24 raw readback required, not a partial gain")
    courses = []
    for case in p.cases:
        w = case["receipt"]["witnesses"]["commitment"]
        courses.append(tuple(json.loads(Path(w["path"]).read_text())["course"]))
    plan = declare_local_impact_perturbations(tuple(courses))
    if plan != d["plan"]:
        raise ValueError("all outcome-independent original local perturbations required")
    jobs = []
    for i, (pair, job) in enumerate(zip(c["rows"], plan["jobs"], strict=True)):
        if pair != sealed(local / f"pair-{i}-complete.json") or pair["job"] != job:
            raise ValueError("all original declared local contexts required")
        row = pair["arms"][0]
        if (
            row != sealed(local / f"pair-{i}-original_control-complete.json")
            or row["arm"] != "original_control"
        ):
            raise ValueError("original-control complete baseline required")
        pin(row["arrays"]["path"], row["arrays"]["sha256"])
        with np.load(row["arrays"]["path"], allow_pickle=False) as a:
            trace = dict(a)
        verify_original_control_torques(trace)
        case = p.cases[job["source_index"]]
        commitment = json.loads(
            Path(case["receipt"]["witnesses"]["commitment"]["path"]).read_text()
        )
        if (
            row["outcome"] != label_first_touch(trace).to_dict()
            or row["source_receipt_hash"] != case["receipt"]["receipt_hash"]
            or row["original_policy_hash"] != commitment["motor_policy_hash"]
        ):
            raise ValueError("same full baseline identity and primitive outcome required")
        jobs.append(
            dict(
                index=i,
                partition="CONSUMED_LOCAL_DEVELOPMENT",
                source_index=job["source_index"],
                course=job["course"],
                baseline_hash=row["report_hash"],
                baseline_outcome=row["outcome"],
                baseline_trace=trace,
            )
        )
    for i, (case, course) in enumerate(zip(p.cases, courses, strict=True)):
        trace = case["trace"]
        verify_original_control_torques(trace)
        outcome = label_first_touch(trace).to_dict()
        if outcome != case["receipt"]["outcome"]:
            raise ValueError("original retention raw outcome required")
        jobs.append(
            dict(
                index=12 + i,
                partition="CONSUMED_ORIGINAL_RETENTION",
                source_index=i,
                course=list(course),
                baseline_hash=case["receipt"]["receipt_hash"],
                baseline_outcome=outcome,
                baseline_trace=trace,
            )
        )
    physical = Path(p.cfg["release_physics_root"])
    pc = sealed(physical / "complete.json", p.cfg["release_physics_complete_file_hash"])
    pd = sealed(physical / "declaration.json")
    if (
        pc["schema"] != "soccer.rsi.recorded_postcontact14_complete.v1"
        or pc["declaration_hash"] != pd["report_hash"]
        or len(pc["rows"]) != 14
        or pc["actual_physical_steps"] != 42000
        or pc["physical_contexts"] != 1
        or pc["teacher_qualified"] is not False
    ):
        raise ValueError("full physical action-authority diagnostic required, not teacher")
    for name, h in pd["input_file_hashes"].items():
        pin(name, h)
    for i, row in enumerate(pc["rows"]):
        if (
            row != sealed(physical / f"trial-{i}-complete.json")
            or row["physical_review"]["actual_physical_steps"] != 3000
        ):
            raise ValueError("all fourteen independent recorded bodies required")
    p.jobs, p.local_complete_hash, p.release_physics_hash = (
        jobs,
        c["report_hash"],
        pc["report_hash"],
    )
    return p
