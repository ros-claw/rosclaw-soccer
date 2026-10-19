"""Source/whole-cohort admission for one bounded offline torque authority screen."""

import json
import subprocess
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from rosclaw_soccer.rsi.foot_heading_parity import original_arrays
from rosclaw_soccer.rsi.foot_heading_scope import assess_heading_scope
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_diagnose_six_contact_heading import stage_kind
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_scope(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "retrospective-impact-torque-six-v1"
        or cfg.get("frame_selection") != "RETROSPECTIVE_FIRST_FOOT_CONTACT_CONTROL_FRAME"
        or cfg.get("leg_eligibility") != "ORIGINAL_CAUSAL_SWING_EQUALS_FIRST_CONTACT_FOOT"
        or cfg.get("repeat_zero_after_candidates") is not True
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("source_cases", 6),
                ("maximum_candidate_actions_per_case", 13),
                ("substeps_per_forecast", 10),
                ("recorded_replay_substep_budget", 18000),
                ("counterfactual_substep_budget", 840),
                ("combined_substep_budget", 18840),
                ("new_closed_loop_policy_trials", 0),
                ("new_neural_inferences", 0),
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
        raise ValueError("fixed six retrospective diagnostic and18840-step maximum required")


def prepare(path):
    cfg = json.loads(path.read_text())
    require_scope(cfg)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        root.exists()
        or Path(__file__).resolve() != source / "scripts/rsi_impact_screen_inputs.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or not isinstance(cfg.get("deadline_unix"), (int, float))
        or not np.isfinite(cfg["deadline_unix"])
        or time.time() >= cfg["deadline_unix"]
    ):
        raise ValueError("unused exact frozen bounded impact diagnostic required")
    require_terminal(cfg["stage_unit"])
    pins = {str(path): digest(path)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(name, key="report_hash", expected_file_hash=None):
        p = Path(name)
        h = digest(p)
        if (
            p.is_symlink()
            or (expected_file_hash is not None and h != expected_file_hash)
            or (str(p) in pins and pins[str(p)] != h)
        ):
            raise ValueError("original ordinary impact evidence bytes required")
        d = json.loads(p.read_text())
        if d[key] != hash_json({k: v for k, v in d.items() if k != key}):
            raise ValueError("whole original impact evidence seal required")
        pins[str(p)] = h
        return d

    def linked(name, key="report_hash"):
        if str(Path(name)) not in pins:
            raise ValueError("linked evidence must already belong to whole geometry proof")
        return sealed(name, key)

    geometry = sealed(cfg["geometry_readback"], expected_file_hash=cfg["geometry_file_hash"])
    if (
        geometry["report_hash"] != cfg["geometry_report_hash"]
        or geometry.get("schema") != "soccer.rsi.whole_six_collision_heading_readback.v1"
        or geometry.get("stage_kind") != "foot-heading"
        or geometry.get("all6_including_nofoot_censored_and_failed_cases_retained") is not True
        or geometry.get("physical_integration_performed") is not False
        or len(geometry["rows"]) != 6
        or any(
            geometry.get(k) is not False
            for k in (
                "teacher_qualified",
                "training_authorized",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            type(geometry.get(k)) is not int or geometry[k] != 0
            for k in ("new_physical_steps", "new_neural_inferences", "new_optimizer_updates")
        )
    ):
        raise ValueError("whole previously source-authenticated C3 collision geometry required")
    # Reuse the complete immutable prior proof, not its outcome table alone.
    for name, h in geometry["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("whole C3 source/parent/body/dense evidence changed")
        pins[name] = h
    if str(Path(cfg["stage_readback"])) not in pins:
        raise ValueError("stage must be in the complete geometry authentication chain")
    stage = sealed(cfg["stage_readback"])
    if (
        stage_kind(stage) != "foot-heading"
        or stage["report_hash"] != geometry["stage_readback_hash"]
    ):
        raise ValueError("same completed C3 stage required")
    spec = stage["effective_cohort_spec"]
    for unit in (
        spec["producer_unit"],
        spec["audit_unit"],
        stage["effective_contact_spec"]["unit"],
    ):
        require_terminal(unit)
    declaration, complete = (linked(spec[k]) for k in ("producer_declaration", "producer_complete"))
    assessment = assess_heading_scope(declaration["scope"], complete["results"])
    if (
        complete["report_hash"] != geometry["native_complete_hash"]
        or complete["report_hash"] != stage["native_complete_hash"]
        or complete["assessment"] != assessment
        or stage["stage_diagnostics"] != assessment["stages"]
    ):
        raise ValueError("unchanged whole original six-body outcomes required")
    cases = []
    world_hash = declaration["mesh_task_contact_world_hash"]
    for i, (job, r, g) in enumerate(
        zip(declaration["scope"]["jobs"], complete["results"], geometry["rows"], strict=True)
    ):
        if (
            linked(Path(spec["producer_complete"]).parent / f"receipt-{i:04d}.json", "receipt_hash")
            != r
            or g["job"] != job
            or g["source_receipt_hash"] != r["receipt_hash"]
            or g["original_outcome"] != r["outcome"]
        ):
            raise ValueError(
                "every original ordered receipt and geometry including failures required"
            )
        for w in r["witnesses"].values():
            if pins.get(w["path"]) != w["sha256"] or digest(w["path"]) != w["sha256"]:
                raise ValueError("whole previously qualified ordinary native witnesses required")
        with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as archive:
            trace = dict(archive)
        original_arrays(trace)
        commitment = json.loads(Path(r["witnesses"]["commitment"]["path"]).read_text())
        physics = commitment["physics"]
        for name in ("canonical_pd_kp", "canonical_pd_kd"):
            gains = np.asarray(physics[name], dtype=np.float64)
            if (
                gains.shape != (29,)
                or not np.isfinite(gains).all()
                or np.any(gains < 0)
                or np.max(gains) > 2000
                or (name == "canonical_pd_kp" and np.any(gains <= 0))
            ):
                raise ValueError("original finite canonical PD gains required before allocation")
        if (
            r["witnesses"]["world"]["sha256"] != world_hash
            or commitment["compiled_model_hash"] != world_hash
            or physics["version"] != "3.12.0"
            or physics["dt_s"] != 0.002
            or physics["control_dt_s"] != 0.02
        ):
            raise ValueError("original unchanged C3 world and PD clock required")
        contact = g["contact"]
        frame, side = None, None
        reason = "NO_ORIGINAL_FOOT_CONTACT"
        if contact is not None:
            if (
                type(contact["first_foot_substep"]) is not int
                or not 0 <= contact["first_foot_substep"] < 3000
            ):
                raise ValueError("original dense contact clock required")
            frame = contact["first_foot_substep"] // 10
            if frame != r["outcome"]["first_foot_contact_frame"]:
                raise ValueError("dense contact and control-frame labels differ")
            observed_side = ("left_ankle_roll_link", "right_ankle_roll_link").index(contact["body"])
            causal_side = float(trace["foot_heading_state"][frame, 0, 1])
            if causal_side == observed_side:
                side, reason = observed_side, None
            else:
                reason = "FIRST_CONTACT_NOT_ORIGINAL_CAUSAL_SWING_FOOT"
        cases.append(
            dict(
                job=job,
                receipt=r,
                geometry=g,
                frame=frame,
                side=side,
                ineligible_reason=reason,
                trace=trace,
                physics=physics,
            )
        )
    return SimpleNamespace(
        cfg=cfg,
        source=source,
        root=root,
        pins=pins,
        geometry=geometry,
        stage=stage,
        declaration=declaration,
        complete=complete,
        cases=cases,
        world_hash=world_hash,
    )
