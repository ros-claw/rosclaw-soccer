"""Authenticate recorded NAV/IK winners into one proposed batch; no optimizer."""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.actual_winner_batch import prepare_actual_winner_batch
from rosclaw_soccer.rsi.actual_winner_supervision import select_actual_winner_paths
from rosclaw_soccer.rsi.positive_lateral_diagnostics import (
    measured_contact_readout,
    require_cohort,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_compare_matched_references import digest, load_cohort, require_terminal
from scripts.rsi_prepare_reference_state_aggregation import original_actor_rows

PROFILE = "proposed-actual-winner-zero-anchor-batch-v1"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        cfg.get("profile") != PROFILE
        or cfg.get("mode") != "SIM_ONLY"
        or root.exists()
        or time.time() >= cfg["deadline_unix"]
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "dataset_admission_authorized",
                "neural_training_authorized",
                "private_fresh_accessed",
            )
        )
        or any(
            type(cfg.get(k)) is not int or cfg[k] != 0
            for k in ("new_physical_steps", "optimizer_updates")
        )
        or Path(__file__).resolve() != source / "scripts/rsi_prepare_actual_winner_batch.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused frozen SIM-only data preparation with zero fitting required")
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get(key) != hash_json({k: x for k, x in v.items() if k != key}):
            raise ValueError("original sealed source evidence required")
        pins[str(path)] = digest(path)
        return v

    def unchanged(owner):
        for name, h in {**owner["input_file_hashes"], **owner.get("prepared_pins", {})}.items():
            if digest(name) != h or (name in pins and pins[name] != h):
                raise ValueError("original source, data or witness bytes changed")
            pins[name] = h

    zero = load_cohort(cfg["zero_cohort"], sealed=sealed, pins=pins)
    prepared = sealed(cfg["original_array_container"])
    if prepared["report_hash"] != cfg["original_array_container_hash"]:
        raise ValueError("fixed original numeric container required")
    unchanged(prepared)
    stage_path = prepared["configuration"]["original_fit_complete"]
    stage, fit = sealed(stage_path), sealed(Path(stage_path).parent / "numeric-fit.json")
    if (
        stage["report_hash"] != prepared["source_fit_complete_hash"]
        or fit["report_hash"] != stage["numeric_fit_hash"]
    ):
        raise ValueError("original teacher numeric fit lineage required")
    unchanged(stage)
    pcpath = Path(stage["configuration"]["positive_configuration"])
    ccpath = Path(stage["configuration"]["positive_contact_configuration"])
    for p, key in ((pcpath, "positive_configuration"), (ccpath, "positive_contact_configuration")):
        if digest(p) != stage["configuration"][key + "_file_hash"]:
            raise ValueError("original teacher producer and contact configuration required")
        pins[str(p)] = digest(p)
    pc, cc = json.loads(pcpath.read_text()), json.loads(ccpath.read_text())
    for unit in (cc["producer_unit"], cc["audit_unit"], cfg["teacher_contact_unit"]):
        require_terminal(unit)
    tsource = Path(pc["source_root"])
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=tsource, text=True).strip()
        != pc["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=tsource, text=True).strip()
    ):
        raise ValueError("unchanged original teacher source required")
    troot = Path(pc["output_root"])
    decl, complete = (sealed(troot / n) for n in ("declaration.json", "complete.json"))
    audit, dense = sealed(cc["readback"]), sealed(cc["complete_path"])
    replay, metrics = (
        sealed(Path(cc["replay_output_root"]) / "complete.json"),
        sealed(cc["metrics_output"]),
    )
    paired = require_cohort(pc, decl, complete, audit)
    measured = measured_contact_readout(decl, complete, replay, metrics)
    if (
        complete["report_hash"] != stage["positive_complete_hash"]
        or dense["report_hash"] != stage["contact_complete_hash"]
        or dense["upstream_readback_hash"] != audit["report_hash"]
        or dense["replay_complete_hash"] != replay["report_hash"]
        or dense["metrics_hash"] != metrics["report_hash"]
        or dense["measured_contact_readout"] != measured
        or dense["paired_development"] != paired
        or audit["reader_source_hash"] != digest(cc["audit_source"])
        or [r["receipt_hash"] for r in complete["results"]] != stage["source_receipt_hashes"]
        or pc["behavior_model_hash"] != zero["config"]["behavior_model_hash"]
        or pc["mesh_task_contact_world_hash"] != zero["config"]["mesh_task_contact_world_hash"]
    ):
        raise ValueError("whole original teacher physics, dense audit and shared world required")
    for owner in (decl, audit):
        unchanged(owner)
    for i, r in enumerate(complete["results"]):
        if sealed(troot / f"receipt-{i:04d}.json", "receipt_hash") != r:
            raise ValueError("all original teacher/control receipt files required")
    teachers = complete["results"][1::2]
    if any(zero["declaration"]["reused_teachers"][str(t["group"])] != t for t in teachers):
        raise ValueError("same exact original measured IK teachers required")
    zeros = zero["complete"]["results"][1:]
    zn = {
        r["job"]["group"]: r["metrics"]["first_nonfoot_robot_contact_substep"] is not None
        for r in zero["measured"]["rows"]
        if r["job"]["role"] == "zero"
    }
    tn = {
        r["job"]["group"]: r["dense_row"]["metrics"]["first_nonfoot_robot_contact_substep"]
        is not None
        for r in measured["rows"]
        if r["job"]["role"] == "teacher"
    }
    selection = select_actual_winner_paths(
        parent_model_hash=pc["behavior_model_hash"],
        zeros=zeros,
        teachers=teachers,
        zero_nonfoot=zn,
        teacher_nonfoot=tn,
    )
    w = prepared["numeric_arrays"]
    if digest(w["path"]) != w["sha256"]:
        raise ValueError("original whole numeric input bytes required")
    pins[w["path"]] = w["sha256"]
    with np.load(w["path"], allow_pickle=False) as a:
        ox, oy = a["actual_observations"][:16], a["reference_proposals"][:16]
    if (
        hash_bytes(ox.tobytes()) != fit["whole_observations_hash"]
        or hash_bytes(oy.tobytes()) != fit["whole_reference_labels_hash"]
    ):
        raise ValueError("byte-exact original own-state IK inputs and labels required")
    observed = []
    for r in zeros:
        w = r["witnesses"]["trace"]
        if pins.get(w["path"]) != w["sha256"]:
            raise ValueError("authenticated original NAV trace required")
        with np.load(w["path"], allow_pickle=False) as a:
            observed.append(original_actor_rows(dict(a), captured_required=True))
    batch, numeric = prepare_actual_winner_batch(
        np.stack([r["observation"] for r in observed]),
        np.stack([r["same_state_reference_target_rad"] for r in observed]),
        ox[1::2],
        oy[1::2],
        selection=selection,
    )
    if args.preflight:
        print("ACTUAL_WINNER_16_PATH_BATCH_READY_ZERO_OPTIMIZER_OR_PHYSICS", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        path = root / "proposed-actual-paths-not-admitted.npz"
        with path.open("xb") as stream:
            np.savez_compressed(
                stream,
                actual_observations=batch["original_observations"],
                reference_targets=batch["targets"],
                partitions=np.asarray(batch["partitions"]),
                source_seeds=np.asarray(batch["source_seeds"]),
                eligible=np.asarray(batch["eligible"]),
            )
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("input source or evidence changed during preparation")
        report = dict(
            schema="soccer.rsi.actual_winner_batch_prepared.v1",
            configuration=cfg,
            input_file_hashes=pins,
            selection=selection,
            numeric_batch=numeric,
            numeric_arrays=dict(path=str(path), sha256=digest(path)),
            zero_complete_hash=zero["complete"]["report_hash"],
            teacher_complete_hash=complete["report_hash"],
            source_authentication_performed=True,
            original_task_labels_unchanged=True,
            dataset_admission_authorized=False,
            neural_training_authorized=False,
            teacher_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            optimizer_updates=0,
            new_physical_steps=0,
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(
            "ACTUAL_WINNER_BATCH_PREPARED_NOT_ADMITTED_OR_TRAINED",
            report["report_hash"],
            flush=True,
        )
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(type=type(error).__name__, message=str(error), automatic_retry_authorized=False),
        )
        raise


if __name__ == "__main__":
    main()
