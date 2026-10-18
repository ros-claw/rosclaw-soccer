"""Once-only matched offline input ablation after measured feedback diagnosis.

Same sixteen source bodies, labels, eligibility, source split, normalization,
seed, optimizer and 128 steps. No past-reference input weights, no new decoder,
physics, Foundation inference, automatic activation or checkpoint selection.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.pre_parent_reference_features import trace_observations
from rosclaw_soccer.rsi.reference_feedback_diagnostics import diagnose_feedback
from rosclaw_soccer.rsi.reference_student_imitation import (
    fit_numeric_reference_student,
    prepare_numeric_batch,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_mechanism(report):
    """Predeclared numerical criterion, not physical qualification."""
    rows = report.get("rows", [])
    eligible = [r for r in rows if r.get("eligible_for_original_fit") is True]
    if (
        len(rows) != 16
        or len(eligible) != 4
        or sum(r["partition"] == "train" for r in eligible) != 3
        or sum(r["partition"] == "validation" for r in eligible) != 1
        or any(not r.get("original_fit_raw_mse_reproduced") for r in rows)
        or any(
            not r.get("measured_body_navigation_and_force_inputs_bit_exact_between_paths")
            for r in rows
        )
    ):
        raise ValueError("whole diagnosed original eligible train and validation sources required")
    for r in eligible:
        e = r["errors"]
        a = e["teacher_forced_projected"]["mean_squared_reference_error_rad2"]
        b = e["self_fed_projected"]["mean_squared_reference_error_rad2"]
        raw = e["teacher_forced_raw"]["mean_squared_reference_error_rad2"]
        if not (
            np.isfinite([a, b, raw]).all() and a > 0 and b > 1.5 * a and abs(a - raw) < 0.05 * a
        ):
            raise ValueError(
                "measured feedback loss must dominate projection on all eligible sources"
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile") != "matched-past-reference-input-ablation-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or cfg.get("offline_numeric_fit_authorized") is not True
        or cfg.get("ignore_past_reference") is not True
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "validation_checkpoint_selection",
                "online_rl_claimed",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("fit_steps", 128),
                ("fit_seed", 2026101902),
                ("new_physical_steps", 0),
                ("new_foundation_inferences", 0),
            )
        )
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or Path(__file__).resolve() != source / "scripts/rsi_fit_reference_feedback_ablation.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "matched finite offline fit, fixed frozen source and unused output required"
        )
    require_terminal(cfg["diagnostic_unit"])
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get("report_hash") != hash_json({k: x for k, x in v.items() if k != "report_hash"}):
            raise ValueError("original sealed complete evidence required")
        pins[str(path)] = digest(path)
        return v

    diagnostic = sealed(cfg["diagnostic_complete"])
    if diagnostic["report_hash"] != cfg["diagnostic_complete_hash"]:
        raise ValueError("fixed measured prior diagnostic required")
    require_mechanism(diagnostic)
    for name, h in diagnostic["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("original diagnosed evidence changed")
        pins[name] = h
    oldroot = Path(diagnostic["configuration"]["fit_complete"]).parent
    stage, previous = sealed(oldroot / "complete.json"), sealed(oldroot / "numeric-fit.json")
    if (
        stage["report_hash"] != diagnostic["fit_complete_hash"]
        or previous["report_hash"] != diagnostic["numeric_fit_hash"]
    ):
        raise ValueError("unchanged original fit and full source rows required")
    pc = json.loads(Path(stage["configuration"]["positive_configuration"]).read_text())
    positive = sealed(Path(pc["output_root"]) / "complete.json")
    if (
        positive["report_hash"] != stage["positive_complete_hash"]
        or [r["receipt_hash"] for r in positive["results"]] != stage["source_receipt_hashes"]
    ):
        raise ValueError("all original ordered physical endpoints required")
    traces = []
    for receipt in positive["results"]:
        w = receipt["witnesses"]["trace"]
        if diagnostic["input_file_hashes"].get(w["path"]) != w["sha256"]:
            raise ValueError("original diagnosed trace required")
        with np.load(w["path"], allow_pickle=False) as a:
            traces.append(dict(a))
    numeric = [trace_observations(t) for t in traces]
    batch = prepare_numeric_batch(
        np.stack([r["observation"] for r in numeric]),
        np.stack([r["same_state_reference_target_rad"] for r in numeric]),
        partitions=[r["partition"] for r in previous["final_same_state_diagnostics"]],
        source_seeds=[r["source_seed"] for r in previous["final_same_state_diagnostics"]],
        eligible=stage["eligibility"],
    )
    if (
        batch["mean"].tolist() != previous["numeric_model"]["mean"]
        or batch["scale"].tolist() != previous["numeric_model"]["scale"]
        or hash_bytes(batch["original_observations"].tobytes())
        != previous["whole_observations_hash"]
        or hash_bytes(batch["targets"].tobytes()) != previous["whole_reference_labels_hash"]
    ):
        raise ValueError("matched whole observations, labels and train-only normalization required")
    if args.preflight:
        print("MATCHED_PAST_REFERENCE_ABLATION_READY_ZERO_OPTIMIZER_ALLOCATIONS", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        fitted = fit_numeric_reference_student(
            batch,
            world_hash=pc["mesh_task_contact_world_hash"],
            parent_model_hash=pc["behavior_model_hash"],
            seed=cfg["fit_seed"],
            steps=cfg["fit_steps"],
            ignore_past_reference=True,
        )
        write_once(root / "numeric-fit.json", fitted)
        mjmodel = mujoco.MjModel.from_binary_path(
            positive["results"][0]["witnesses"]["world"]["path"]
        )
        limits = np.array([mjmodel.jnt_range[mjmodel.joint(n).id] for n in G1_DDS_JOINT_NAMES])
        feedback = []
        for trace in traces:
            row = diagnose_feedback(trace, fitted["numeric_model"], joint_limits=limits)
            for kind in ("raw", "projected"):
                if row["errors"]["teacher_forced_" + kind] != row["errors"]["self_fed_" + kind]:
                    raise ValueError(
                        "actual exported ablation must ignore all past-reference feedback"
                    )
            feedback.append(row)
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source or physical evidence changed during ablation")
        report = dict(
            schema="soccer.rsi.reference_feedback_ablation_fit_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            previous_fit_complete_hash=stage["report_hash"],
            prior_diagnostic_hash=diagnostic["report_hash"],
            numeric_fit_hash=fitted["report_hash"],
            numeric_model_hash=fitted["numeric_model"]["model_hash"],
            source_receipt_hashes=stage["source_receipt_hashes"],
            eligibility=stage["eligibility"],
            optimizer_updates=fitted["learning_receipt"]["completed_optimizer_steps"],
            same_recorded_body_feedback_diagnostics=feedback,
            all_original_failures_retained=True,
            own_body_student_evaluated=False,
            hardware_authorized=False,
            promotion_authorized=False,
            dataset_admission_authorized=False,
            teacher_qualified=False,
            new_physical_steps=0,
            new_foundation_inferences=0,
            online_rl_claimed=False,
            status="MATCHED_ABLATION_FIT_COMPLETE_REQUIRES_OWN_BODY_EXAM",
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(report["status"], report["numeric_model_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
