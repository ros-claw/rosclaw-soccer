"""Admit the closed whole18 baseline for prospective full-remainder parity."""

import json
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.first_touch_outcome import label_first_touch
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_postcontact_release_transfer_inputs import prepare as original_prepare
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    fixed = dict(
        profile="full-task-zero-branch18-v1",
        mode="SIM_ONLY",
        source_cases=6,
        local_contexts=12,
        retention_contexts=6,
        main_reconstruction_runs=18,
        main_native_step_budget=54000,
        prediction_runs=36,
        prediction_native_step_budget=40840,
        combined_native_step_budget=94840,
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
    if any(type(cfg.get(k)) is not type(v) or cfg[k] != v for k, v in fixed.items()):
        raise ValueError("whole18 main and36 zero predictions within94840 native steps required")


def prepare(path, *, scope_validator=require_scope):
    p = original_prepare(path, scope_validator=scope_validator)
    require_terminal(p.cfg["gold_unit"])
    require_terminal(p.cfg["gold_physical_unit"])

    def pin(name, expected=None):
        f = Path(name)
        h = digest(f)
        if (
            f.is_symlink()
            or (expected is not None and h != expected)
            or (str(f) in p.pins and p.pins[str(f)] != h)
        ):
            raise ValueError("whole immutable original full-task gold required")
        p.pins[str(f)] = h

    def sealed(name, expected=None):
        pin(name, expected)
        d = json.loads(Path(name).read_text())
        if d["report_hash"] != hash_json({k: v for k, v in d.items() if k != "report_hash"}):
            raise ValueError("complete full-task gold seal required")
        return d

    root, physical = Path(p.cfg["gold_root"]), Path(p.cfg["gold_physical_root"])
    c = sealed(root / "complete.json", p.cfg["gold_complete_file_hash"])
    d = sealed(root / "declaration.json")
    pc = sealed(physical / "complete.json", p.cfg["gold_physical_complete_file_hash"])
    pd = sealed(physical / "declaration.json")
    if (
        c["schema"] != "soccer.rsi.constant_forward18_complete.v1"
        or c["declaration_hash"] != d["report_hash"]
        or len(c["rows"]) != 18
        or d["config"]["mean_model_hash"] != p.cfg["mean_model_hash"]
        or pc["schema"] != "soccer.rsi.constant_forward18_verification_complete.v1"
        or pc["declaration_hash"] != pd["report_hash"]
        or len(pc["rows"]) != 18
        or pd["producer_complete_hash"] != c["report_hash"]
        or pc["actual_physical_steps"] != 54000
        or any(
            x.get(k) is not False
            for x in (c, pc)
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
    ):
        raise ValueError(
            "closed whole18 constant probe and exact independent physical proof required"
        )
    for data in (d, pd):
        for n, h in data["input_file_hashes"].items():
            pin(n, h)
    jobs = []
    for i, (row, job, proof) in enumerate(zip(c["rows"], p.jobs, pc["rows"], strict=True)):
        if (
            row != sealed(root / f"case-{i}-complete.json")
            or row["index"] != i
            or row["course"] != job["course"]
            or row["source_index"] != job["source_index"]
            or proof != sealed(physical / f"case-{i}-complete.json")
            or proof["source_case_hash"] != row["report_hash"]
            or proof["physical_review"]["actual_physical_steps"] != 3000
            or any(
                not np.isfinite(e) or e > 1e-8
                for e in proof["physical_review"]["maximum_absolute_errors"].values()
            )
        ):
            raise ValueError("all ordered same-course/policy gold and physical rows required")
        pin(row["arrays"]["path"], row["arrays"]["sha256"])
        with np.load(row["arrays"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        if label_first_touch(trace).to_dict() != row["outcome"]:
            raise ValueError("unchanged raw gold M0 label required")
        first = row["outcome"]["first_foot_contact_frame"]
        if type(first) is not int or not 30 <= first < 299:
            raise ValueError(
                "all fixed gold sources must have a genuine completed comparison foot bin"
            )
        jobs.append(
            dict(
                index=i,
                source_index=row["source_index"],
                course=row["course"],
                partition=row["partition"],
                source_case_hash=row["report_hash"],
                gold_arrays=row["arrays"],
                gold_outcome=row["outcome"],
                original_policy_hash=row["original_policy_hash"],
                comparison_start_after_measured_foot_bin=first + 1,
                gold_trace=trace,
            )
        )
    if sum((300 - j["comparison_start_after_measured_foot_bin"]) * 20 for j in jobs) != 40840:
        raise ValueError("exact94840-step retrospective parity budget required")
    navigation = Path(p.cfg["navigation_parity_root"])
    nc = sealed(navigation / "complete.json", p.cfg["navigation_parity_complete_file_hash"])
    nd = sealed(navigation / "declaration.json")
    if (
        nc["schema"] != "soccer.rsi.real_navigation_branch18_complete.v1"
        or nc["declaration_hash"] != nd["report_hash"]
        or len(nc["rows"]) != 18
        or nc["actual_planner_onnx_inferences"] != 322
        or nc["new_physical_steps"] != 0
        or nc["long_horizon_counterfactual_physics_qualified"] is not False
    ):
        raise ValueError(
            "whole actual navigation component parity, not body qualification, required"
        )
    for n, h in nd["input_file_hashes"].items():
        pin(n, h)
    for i, row in enumerate(nc["rows"]):
        if (
            row != sealed(navigation / f"case-{i}-complete.json")
            or row["source_case_hash"] != jobs[i]["source_case_hash"]
            or row["zero_and_repeat_references_events_targets_bit_exact"] is not True
        ):
            raise ValueError("all18 actual resource parity source bindings required")
    p.jobs = jobs
    p.gold_complete_hash = c["report_hash"]
    return p
