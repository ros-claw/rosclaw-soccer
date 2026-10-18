"""One predeclared offline zero-anchor candidate; no physics or activation."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.actual_winner_batch import prepare_actual_winner_batch
from rosclaw_soccer.rsi.actual_winner_supervision import select_actual_winner_paths
from rosclaw_soccer.rsi.reference_student_imitation import fit_numeric_reference_student
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compare_matched_references import digest, require_terminal


def load_batch(cfg, *, sealed, pins):
    """Rebuild the complete numeric selection; no caller-provided approval flag."""
    require_terminal(cfg["prepared_unit"])
    prepared = sealed(cfg["prepared_complete"])
    pc = prepared["configuration"]
    if (
        prepared["report_hash"] != cfg["prepared_complete_hash"]
        or prepared["schema"] != "soccer.rsi.actual_winner_batch_prepared.v1"
        or pc["profile"] != "proposed-actual-winner-zero-anchor-batch-v1"
        or pc["mode"] != "SIM_ONLY"
        or prepared["source_authentication_performed"] is not True
        or prepared["original_task_labels_unchanged"] is not True
        or any(
            prepared.get(k) is not False
            for k in (
                "dataset_admission_authorized",
                "neural_training_authorized",
                "teacher_qualified",
                "hardware_authorized",
                "promotion_authorized",
            )
        )
        or any(
            type(prepared.get(k)) is not int or prepared[k] != 0
            for k in ("optimizer_updates", "new_physical_steps")
        )
        or subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=pc["source_root"], text=True
        ).strip()
        != pc["source_head"]
        or subprocess.check_output(
            ["git", "status", "--porcelain"], cwd=pc["source_root"], text=True
        ).strip()
    ):
        raise ValueError("original authenticated unadmitted actual-winner batch required")
    for name, h in prepared["input_file_hashes"].items():
        if digest(name) != h or (name in pins and pins[name] != h):
            raise ValueError("original prepared sources or physical evidence changed")
        pins[name] = h
    rows = prepared["selection"]["rows"]
    byrole = {role: [r for r in rows if r["role"] == role] for role in ("zero", "teacher")}
    selection = select_actual_winner_paths(
        parent_model_hash=cfg["parent_model_hash"],
        **{
            role: [
                dict(
                    group=r["group"],
                    role=r["role"],
                    receipt_hash=r["source_receipt_hash"],
                    outcome=r["original_outcome"],
                )
                for r in byrole[key]
            ]
            for role, key in (("zeros", "zero"), ("teachers", "teacher"))
        },
        zero_nonfoot={r["group"]: r["dense_nonfoot_over1n"] for r in byrole["zero"]},
        teacher_nonfoot={r["group"]: r["dense_nonfoot_over1n"] for r in byrole["teacher"]},
    )
    if selection != prepared["selection"]:
        raise ValueError("original full successful/failed path selection must reconstruct")
    zconfig = json.loads(Path(pc["zero_cohort"]["contact_configuration"]).read_text())
    native = json.loads(Path(zconfig["producer_config"]).read_text())
    if (
        native["behavior_model_hash"] != cfg["parent_model_hash"]
        or native["mesh_task_contact_world_hash"] != cfg["world_hash"]
    ):
        raise ValueError("unchanged physical world and parent model required")
    w = prepared["numeric_arrays"]
    if digest(w["path"]) != w["sha256"]:
        raise ValueError("original numeric byte witness required")
    pins[w["path"]] = w["sha256"]
    with np.load(w["path"], allow_pickle=False) as a:
        if set(a.files) != {
            "actual_observations",
            "reference_targets",
            "partitions",
            "source_seeds",
            "eligible",
        }:
            raise ValueError("complete actual-path numeric archive required")
        x, y = a["actual_observations"], a["reference_targets"]
        if x.shape != (16, 300, 135) or y.shape != (16, 300, 12):
            raise ValueError("whole sixteen recorded paths required")
        batch, numeric = prepare_actual_winner_batch(
            x[::2], y[::2], x[1::2], y[1::2], selection=selection
        )
        for key in ("partitions", "source_seeds", "eligible"):
            if a[key].tolist() != batch[key]:
                raise ValueError("original split and positive-weight selection required")
    if numeric != prepared["numeric_batch"] or (
        numeric["train_paths"],
        numeric["validation_paths"],
    ) != (3, 2):
        raise ValueError("exact predeclared three-train/two-validation research objective required")
    return prepared, batch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile") != "one-actual-winner-zero-anchor-fit-v1"
        or cfg.get("mode") != "SIM_ONLY"
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or cfg.get("experimental_numeric_fit_authorized") is not True
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "teacher_auto_admission",
                "private_fresh_accessed",
                "online_rl_claimed",
                "validation_checkpoint_selection",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("fit_seed", 2026101902),
                ("fit_steps", 128),
                ("maximum_numeric_candidates", 1),
                ("new_physical_steps", 0),
            )
        )
        or Path(__file__).resolve() != source / "scripts/rsi_fit_actual_winner_batch.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused frozen one-seed 128-step offline research fit required")
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get(key) != hash_json({k: x for k, x in v.items() if k != key}):
            raise ValueError("original sealed data receipt required")
        pins[str(path)] = digest(path)
        return v

    prepared, batch = load_batch(cfg, sealed=sealed, pins=pins)
    if args.preflight:
        print("ACTUAL_WINNER_ONE_CANDIDATE_READY_ZERO_OPTIMIZER_ALLOCATIONS", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        fit = fit_numeric_reference_student(
            batch,
            world_hash=cfg["world_hash"],
            parent_model_hash=cfg["parent_model_hash"],
            seed=cfg["fit_seed"],
            steps=128,
            ignore_past_reference=True,
        )
        write_once(root / "numeric-fit.json", fit)
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source or original data changed during fit")
        report = dict(
            schema="soccer.rsi.actual_winner_fit_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            prepared_complete_hash=prepared["report_hash"],
            selection_hash=prepared["selection"]["report_hash"],
            numeric_fit_hash=fit["report_hash"],
            status="NUMERIC_CANDIDATE_REQUIRES_OWN_BODY_EXAM",
            optimizer_updates=fit["learning_receipt"]["completed_optimizer_steps"],
            own_body_student_evaluated=False,
            validation_checkpoint_selection=False,
            dataset_admission_authorized=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            new_physical_steps=0,
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(report["status"], report["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
