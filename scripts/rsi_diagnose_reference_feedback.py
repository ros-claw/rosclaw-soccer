"""Once-only feedback diagnosis from all sixteen authenticated closed fit inputs.

Loads original MJB only for ordered joint limits; creates no MjData and does no
physics, fitting or Foundation inference. Never modifies a current exam.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
from rosclaw_soccer.rsi.reference_feedback_diagnostics import diagnose_feedback
from rosclaw_soccer.rsi.reference_student_execution import own_model
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile") != "closed-sixteen-reference-feedback-diagnostic-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            cfg.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "private_fresh_accessed")
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("input_episodes", 16),
                ("new_physical_steps", 0),
                ("optimizer_updates", 0),
            )
        )
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or Path(__file__).resolve() != source / "scripts/rsi_diagnose_reference_feedback.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError(
            "exact frozen source, unused output and finite read-only SIM scope required"
        )
    require_terminal(cfg["fit_unit"])
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value.get(key) != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original sealed whole evidence required")
        pins[str(path)] = digest(path)
        return value

    stage_path = Path(cfg["fit_complete"])
    stage, fit = sealed(stage_path), sealed(stage_path.parent / "numeric-fit.json")
    if (
        stage["report_hash"] != cfg["fit_complete_hash"]
        or stage["numeric_fit_hash"] != fit["report_hash"]
        or stage["optimizer_updates"] != 128
        or stage["status"] != "NUMERIC_FIT_COMPLETE_REQUIRES_OWN_BODY_EXAM"
    ):
        raise ValueError("original completed fitted student required")
    # These pins include the independent audit, dense contacts, all source
    # receipts and traces, original model/world and training implementation.
    for name, h in stage["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("original closed fit source or physical input changed")
        pins[name] = h
    model = own_model(fit["numeric_model"])
    if (
        model["model_hash"] != cfg["numeric_model_hash"]
        or fit["learning_receipt"]["parameters"] != model["parameters"]
    ):
        raise ValueError("exact fitted numeric model required")
    pcpath = Path(stage["configuration"]["positive_configuration"])
    pc = json.loads(pcpath.read_text())
    proot = Path(pc["output_root"])
    declaration, complete = (sealed(proot / n) for n in ("declaration.json", "complete.json"))
    if (
        complete["report_hash"] != stage["positive_complete_hash"]
        or complete["declaration_hash"] != declaration["report_hash"]
        or len(complete["results"]) != 16
        or [r["receipt_hash"] for r in complete["results"]] != stage["source_receipt_hashes"]
        or len(fit["final_same_state_diagnostics"]) != 16
    ):
        raise ValueError("all sixteen original ordered failures and successes required")
    traces, worlds = [], []
    for i, expected in enumerate(complete["results"]):
        receipt = sealed(proot / f"receipt-{i:04d}.json", "receipt_hash")
        if receipt != expected:
            raise ValueError("unchanged original ordered receipt required")
        for name in ("trace", "world"):
            witness = receipt["witnesses"][name]
            path = Path(witness["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(proot.resolve())
                or digest(path) != witness["sha256"]
                or stage["input_file_hashes"].get(str(path)) != witness["sha256"]
            ):
                raise ValueError("original fit-authenticated owned trace and MJB required")
            pins[str(path)] = witness["sha256"]
        world = Path(receipt["witnesses"]["world"]["path"])
        if digest(world) != model["world_hash"]:
            raise ValueError("same original compiled world required")
        worlds.append(world)
        with np.load(receipt["witnesses"]["trace"]["path"], allow_pickle=False) as a:
            traces.append(dict(a))
    if args.preflight:
        print("SIXTEEN_CLOSED_FEEDBACK_DIAGNOSTICS_READY_ZERO_PHYSICS_OR_OPTIMIZER", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        # One MjModel, only joint ranges. No simulator state allocation/step.
        mjmodel = mujoco.MjModel.from_binary_path(str(worlds[0]))
        limits = np.array([mjmodel.jnt_range[mjmodel.joint(n).id] for n in G1_DDS_JOINT_NAMES])
        rows = []
        for i, trace in enumerate(traces):
            row = diagnose_feedback(trace, model, joint_limits=limits)
            reproduced = row["errors"]["teacher_forced_raw"]["mean_squared_reference_error_rad2"]
            original = fit["final_same_state_diagnostics"][i]["mean_squared_reference_error_rad2"]
            if not np.isclose(reproduced, original, rtol=1e-12, atol=1e-15):
                raise ValueError("original fitted same-state raw loss failed to reconstruct")
            receipt = complete["results"][i]
            row.update(
                receipt_hash=receipt["receipt_hash"],
                role=receipt["role"],
                group=receipt["group"],
                eligible_for_original_fit=stage["eligibility"][i],
                partition=fit["final_same_state_diagnostics"][i]["partition"],
                source_seed=fit["final_same_state_diagnostics"][i]["source_seed"],
                original_fit_raw_mse_reproduced=True,
            )
            rows.append(row)
            print(
                "FEEDBACK_DIAGNOSTIC",
                i,
                row["role"],
                row["group"],
                {k: v["mean_squared_reference_error_rad2"] for k, v in row["errors"].items()},
                flush=True,
            )
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("pinned source or closed evidence changed during diagnosis")
        report = dict(
            schema="soccer.rsi.reference_feedback_diagnostic_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            rows=rows,
            fit_complete_hash=stage["report_hash"],
            numeric_fit_hash=fit["report_hash"],
            all_original_failures_retained=True,
            native_physical_steps=0,
            new_foundation_inferences=0,
            optimizer_updates=0,
            independent_teacher_body_trajectories=16,
            compared_paths_per_body=2,
            student_body_rollout=False,
            causal_mechanism_proven=False,
            hardware_authorized=False,
            promotion_authorized=False,
            interpretation="Self-fed past actions on FIXED recorded bodies; not own-body success. "
            "Projection and feedback errors separated; "
            "physical distribution shift remains untested here.",
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print("REFERENCE_FEEDBACK_DIAGNOSIS_COMPLETE", report["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
