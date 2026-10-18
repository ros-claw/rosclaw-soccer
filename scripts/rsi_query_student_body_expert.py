"""Query the proven cached IK on all thirteen closed student-exam bodies.

No physical integration, Foundation, optimizer, data admission or promotion.
Uses the exact body that the student actually visited, never another rollout.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import mujoco
import numpy as np

from rosclaw_soccer.rsi.bounded_first_touch_control import BoundedFirstTouchControl
from rosclaw_soccer.rsi.contact_reference_labels import reference_labels
from rosclaw_soccer.rsi.recorded_body_reference_query import query_recorded_body
from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout, require_cohort
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
        cfg.get("profile") != "closed-thirteen-student-own-state-expert-query-v1"
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
            type(cfg.get(k)) is not int or cfg[k] != n
            for k, n in (
                ("body_episodes", 13),
                ("new_physical_steps", 0),
                ("optimizer_updates", 0),
            )
        )
        or Path(__file__).resolve() != source / "scripts/rsi_query_student_body_expert.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("once-only pinned thirteen-body SIM kinematic query required")
    for unit in cfg["upstream_units"]:
        require_terminal(unit)
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        v = json.loads(path.read_text())
        if v.get(key) != hash_json({k: x for k, x in v.items() if k != key}):
            raise ValueError("original complete sealed evidence required")
        pins[str(path)] = digest(path)
        return v

    parity = sealed(cfg["original_expert_query_complete"])
    if (
        parity["report_hash"] != cfg["original_expert_query_complete_hash"]
        or parity["original_eight_teachers_bit_exact"] is not True
        or parity["source_episodes"] != 16
    ):
        raise ValueError("actual original eight-teacher bit-exact query proof required")
    cp = Path(cfg["producer_contact_configuration"])
    if digest(cp) != cfg["producer_contact_configuration_file_hash"]:
        raise ValueError("fixed original whole-exam contact configuration required")
    cc = json.loads(cp.read_text())
    pins[str(cp)] = digest(cp)
    pp = Path(cc["producer_config"])
    if digest(pp) != cc["producer_config_file_hash"]:
        raise ValueError("fixed original producer configuration required")
    pc = json.loads(pp.read_text())
    pins[str(pp)] = digest(pp)
    proot = Path(pc["output_root"])
    declaration, complete = (sealed(proot / n) for n in ("declaration.json", "complete.json"))
    audit, contact = sealed(cc["readback"]), sealed(cc["complete_path"])
    pairs = require_cohort(pc, declaration, complete, audit)
    replay = sealed(Path(cc["replay_output_root"]) / "complete.json")
    metrics = sealed(cc["metrics_output"])
    if (
        audit["reader_source_hash"] != digest(cc["audit_source"])
        or contact["upstream_readback_hash"] != audit["report_hash"]
        or contact["paired_results"] != pairs
        or contact["replay_complete_hash"] != replay["report_hash"]
        or contact["metrics_hash"] != metrics["report_hash"]
        or contact["measured_contact_readout"]
        != measured_readout(declaration, complete, replay, metrics)
    ):
        raise ValueError("whole original own-body and dense outcome audit required")
    for owner in (parity, declaration, audit):
        owned = dict(owner["input_file_hashes"])
        owned.update(owner.get("prepared_pins", {}))
        for name, h in owned.items():
            if digest(name) != h:
                raise ValueError("original closed source or physical evidence changed")
            pins[name] = h
    traces, dense = [], []
    for i, (receipt, row) in enumerate(zip(complete["results"], replay["rows"], strict=True)):
        r = sealed(proot / f"receipt-{i:04d}.json", "receipt_hash")
        if r != receipt or row["source_receipt_hash"] != receipt["receipt_hash"]:
            raise ValueError("all ordered unchanged own-body source receipts required")
        for w, owner in (
            (r["witnesses"]["trace"], proot),
            (row["arrays"], Path(cc["replay_output_root"])),
            (r["witnesses"]["world"], proot),
        ):
            path = Path(w["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(owner.resolve())
                or digest(path) != w["sha256"]
            ):
                raise ValueError("original ordinary owned trace, substeps and world required")
            pins[str(path)] = w["sha256"]
        if r["witnesses"]["world"]["sha256"] != pc["mesh_task_contact_world_hash"]:
            raise ValueError("unchanged original compiled world required")
        with np.load(r["witnesses"]["trace"]["path"], allow_pickle=False) as a:
            traces.append(dict(a))
        with np.load(row["arrays"]["path"], allow_pickle=False) as a:
            dense.append(dict(a))
    control = BoundedFirstTouchControl(**declaration["candidate"]["parameters"])
    if declaration["candidate"]["swing"] != "bounded_contact_diagnostic":
        raise ValueError("original bounded body-origin candidate only required")
    if args.preflight:
        print("THIRTEEN_OWN_STATE_EXPERT_QUERIES_READY_ZERO_MJDATA_OR_DYNAMICS", flush=True)
        return
    root.mkdir()
    write_once(root / "started.json", dict(configuration=cfg, automatic_retry_authorized=False))
    try:
        model = mujoco.MjModel.from_binary_path(
            complete["results"][0]["witnesses"]["world"]["path"]
        )
        rows, arrays = [], {}
        for i, (job, receipt, trace, substeps) in enumerate(
            zip(declaration["scope"]["jobs"], complete["results"], traces, dense, strict=True)
        ):
            queried = query_recorded_body(trace, substeps, model, control=control)
            label = queried["proposed_reference_increment_rad"]
            actual = reference_labels(trace)["pre_parent_reference_increment_rad"][:, 0]
            if job["role"] == "off" and label.tobytes() != actual.tobytes():
                raise ValueError("original whole OFF teacher must reproduce expert labels exactly")
            arrays[f"queried_reference_rad_{i:02d}"] = label
            rows.append(
                dict(
                    job=job,
                    source_receipt_hash=receipt["receipt_hash"],
                    original_outcome=receipt["outcome"],
                    query=queried["diagnostic"],
                    mean_squared_difference_from_executed_student_rad2=float(
                        np.mean((label[30:] - actual[30:]) ** 2)
                    ),
                    expert_success_from_this_body_state_verified=False,
                    eligible_for_training=False,
                )
            )
            print("OWN_BODY_EXPERT_QUERY", job["job"], flush=True)
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("source or physical evidence changed during query")
        path = root / "own-body-proposals-not-admitted.npz"
        with path.open("xb") as stream:
            np.savez_compressed(stream, **arrays)
        report = dict(
            schema="soccer.rsi.student_own_state_expert_query_complete.v1",
            configuration=cfg,
            input_file_hashes=pins,
            source_complete_hash=complete["report_hash"],
            source_audit_hash=audit["report_hash"],
            source_contact_hash=contact["report_hash"],
            original_expert_query_hash=parity["report_hash"],
            rows=rows,
            proposed_arrays=dict(path=str(path), sha256=digest(path)),
            kinematic_queries=3900,
            source_body_episodes=13,
            source_courses=10,
            source_clusters=6,
            all_source_failures_retained=True,
            same_actual_body_and_navigation_only=True,
            new_physical_steps=0,
            new_foundation_inferences=0,
            optimizer_updates=0,
            dagger_training_performed=False,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(
            "OWN_STUDENT_BODY_EXPERT_QUERIES_COMPLETE_NOT_ADMISSION",
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
