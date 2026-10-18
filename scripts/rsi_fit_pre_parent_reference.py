"""Finite, unqualified numeric reference fit after both full physical studies.

No new physics/Foundation calls, implicit retries, candidate activation, or
validation checkpoint selection. Upstream failures remain in the whole bank.
"""

import argparse
import json
import subprocess
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.positive_lateral_diagnostics import measured_contact_readout, require_cohort
from rosclaw_soccer.rsi.pre_parent_reference_features import trace_observations
from rosclaw_soccer.rsi.reference_student_imitation import (
    fit_numeric_reference_student,
    prepare_numeric_batch,
)
from rosclaw_soccer.rsi.reference_student_probe import JOBS, measured_authority, require_parity
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_reference_student_three_pilot import require_scope as require_reference_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal

FIT_SEED = 2026101902
LOCAL_EVIDENCE = "LOCAL_REFERENCE_EVIDENCE_REQUIRES_LABEL_BUILD_AND_OWN_STUDENT_CLOSED_LOOP"
UPSTREAM_UNITS = [
    "rosclaw-positive-lateral-sixteen-native-20261019.service",
    "rosclaw-positive-lateral-sixteen-audit-20261019.service",
    "rosclaw-positive-lateral-sixteen-contact-20261019.service",
    "rosclaw-reference-student-three-native-20261019.service",
    "rosclaw-reference-student-three-audit-20261019.service",
]


def require_scope(cfg):
    if (
        cfg.get("fit_profile") != "positive-sixteen-reference-imitation-v1"
        or cfg.get("upstream_units") != UPSTREAM_UNITS
        or cfg.get("offline_numeric_fit_authorized") is not True
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            type(cfg.get(k)) is not int or cfg[k] != v
            for k, v in (
                ("fit_steps", 128),
                ("fit_seed", FIT_SEED),
                ("new_physical_budget", 0),
                ("new_foundation_inferences", 0),
                ("maximum_optimizer_steps", 128),
            )
        )
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "private_fresh_accessed",
                "teacher_auto_admission",
                "validation_checkpoint_selection",
                "online_rl_claimed",
            )
        )
    ):
        raise ValueError(
            "fixed once-only 128-step offline numeric fit, not RL or promotion, required"
        )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    require_scope(cfg)
    source, root = Path(cfg["source_root"]), Path(cfg["output_root"])
    if (
        root.exists()
        or Path(__file__).resolve() != source / "scripts/rsi_fit_pre_parent_reference.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or time.time() >= cfg["deadline_unix"]
    ):
        raise ValueError("unused output, fixed deadline and exact frozen fit source required")
    for unit in cfg["upstream_units"]:
        require_terminal(unit)
    pins = {str(args.config): digest(args.config)}
    for project in (source, Path(cfg["core_root"])):
        for folder in (project / "src", project / "scripts"):
            pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("whole original evidence seal required")
        pins[str(path)] = digest(path)
        return value

    def original_config(key):
        path = Path(cfg[key])
        if digest(path) != cfg[key + "_file_hash"]:
            raise ValueError("unchanged original upstream configuration required")
        pins[str(path)] = digest(path)
        return json.loads(path.read_text())

    positive = original_config("positive_configuration")
    contact_cfg = original_config("positive_contact_configuration")
    reference = original_config("reference_configuration")
    require_reference_scope(reference)
    proot, rroot = Path(positive["output_root"]), Path(reference["output_root"])
    declaration, complete = (sealed(proot / n) for n in ("declaration.json", "complete.json"))
    audit = sealed(contact_cfg["readback"])
    paired = require_cohort(positive, declaration, complete, audit)
    contact = sealed(contact_cfg["complete_path"])
    replay = sealed(Path(contact_cfg["replay_output_root"]) / "complete.json")
    metrics = sealed(contact_cfg["metrics_output"])
    measured = measured_contact_readout(declaration, complete, replay, metrics)
    if (
        contact_cfg["producer_config"] != cfg["positive_configuration"]
        or contact["upstream_readback_hash"] != audit["report_hash"]
        or contact["replay_complete_hash"] != replay["report_hash"]
        or contact["metrics_hash"] != metrics["report_hash"]
        or contact["measured_contact_readout"] != measured
        or contact["paired_development"] != paired
        or audit["reader_source_hash"] != digest(contact_cfg["audit_source"])
    ):
        raise ValueError(
            "whole original dense replay, measured readout and independent audit required"
        )
    rd, rc = (sealed(rroot / n) for n in ("declaration.json", "complete.json"))
    ra = sealed(cfg["reference_readback"])
    if (
        ra.get("schema") != "soccer.rsi.reference_student_three_terminal_readback.v1"
        or rc.get("schema") != "soccer.rsi.reference_student_three_complete.v1"
        or ra["complete_hash"] != rc["report_hash"]
        or rc["declaration_hash"] != rd["report_hash"]
        or rd["jobs"] != list(JOBS)
        or any(k not in rd or rd[k] != v for k, v in reference.items())
        or ra["reader_source_hash"]
        != digest(Path(reference["source_root"]) / "scripts/rsi_read_reference_student_three.py")
        or any(
            ra.get(k) is not True
            for k in (
                "original_off_all33_arrays_bit_exact",
                "all_pre_action_prefixes_bit_exact",
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
            )
        )
        or any(
            ra.get(k) is not False
            for k in (
                "probe_is_trained_policy",
                "teacher_qualified",
                "dataset_admission_authorized",
                "hardware_authorized",
                "promotion_authorized",
                "fresh_exam",
            )
        )
        or positive["mesh_task_contact_world_hash"] != reference["mesh_task_contact_world_hash"]
        or positive["behavior_model_hash"] != reference["behavior_model_hash"]
    ):
        raise ValueError("complete original three-body control authority readback required")
    for owner in (declaration, rd):
        for name, h in {**owner["prepared_pins"], **owner["input_file_hashes"]}.items():
            if digest(name) != h:
                raise ValueError("upstream pinned source or original input changed")
            pins[name] = h

    def read_trace(owner_root, index, expected):
        receipt = sealed(owner_root / f"receipt-{index:04d}.json", "receipt_hash")
        if receipt != expected:
            raise ValueError("whole original ordered receipt file required")
        witnesses = dict(receipt["witnesses"])
        witnesses.update({k: v for k, v in receipt.items() if k.endswith("_witness")})
        for w in witnesses.values():
            path = Path(w["path"])
            if (
                path.is_symlink()
                or not path.resolve().is_relative_to(owner_root.resolve())
                or digest(path) != w["sha256"]
            ):
                raise ValueError("ordinary unchanged owned physical witnesses required")
            pins[str(path)] = w["sha256"]
        with np.load(witnesses["trace"]["path"], allow_pickle=False) as archive:
            return dict(archive)

    if len(rc["results"]) != 3:
        raise ValueError("all three original reference probes required")
    reference_traces = [read_trace(rroot, i, r) for i, r in enumerate(rc["results"])]
    with np.load(rd["baseline_receipt"]["witnesses"]["trace"]["path"], allow_pickle=False) as a:
        original = dict(a)
    for i, trace in enumerate(reference_traces):
        require_parity(trace, original, 300 if i == 0 else 30)
    authority = measured_authority(reference_traces[1], reference_traces[2])
    if authority != rc["observed_authority"] or authority != ra["observed_authority"]:
        raise ValueError("actual four-level physical control response must reconstruct")
    traces = [read_trace(proot, i, r) for i, r in enumerate(complete["results"])]
    split = {r["group"]: r for r in declaration["prospective_split"]["rows"]}
    allowed = set(measured["successful_candidate_receipt_hashes"])
    eligible = [r["receipt_hash"] in allowed for r in complete["results"]]
    rows = [trace_observations(t) for t in traces]
    proposed = bool(
        authority["reference_control_authority_observed"]
        and measured["research_decision"] == LOCAL_EVIDENCE
    )
    batch = None
    if proposed:
        batch = prepare_numeric_batch(
            np.stack([r["observation"] for r in rows]),
            np.stack([r["same_state_reference_target_rad"] for r in rows]),
            partitions=[split[j["group"]]["partition"] for j in declaration["jobs"]],
            source_seeds=[split[j["group"]]["source_seed"] for j in declaration["jobs"]],
            eligible=eligible,
        )
    if any(digest(path) != h for path, h in pins.items()):
        raise ValueError("original evidence changed during fit preparation")
    if args.preflight:
        print(
            "REFERENCE_FIT_READY_ZERO_OPTIMIZER_ALLOCATIONS"
            if proposed
            else "REFERENCE_FIT_REFUSED_BY_ORIGINAL_EVIDENCE",
            flush=True,
        )
        return
    root.mkdir()
    report = dict(
        schema="soccer.rsi.reference_fit_stage_complete.v1",
        configuration=cfg,
        input_file_hashes=pins,
        positive_complete_hash=complete["report_hash"],
        contact_complete_hash=contact["report_hash"],
        reference_readback_hash=ra["report_hash"],
        eligibility=eligible,
        source_receipt_hashes=[r["receipt_hash"] for r in complete["results"]],
        all_source_failures_retained=True,
        new_physical_steps=0,
        new_foundation_inferences=0,
        own_body_student_evaluated=False,
        dataset_admission_authorized=False,
        teacher_qualified=False,
        promotion_authorized=False,
        hardware_authorized=False,
        fresh_exam=False,
    )
    write_once(root / "started.json", dict(report, automatic_retry_authorized=False))
    try:
        if batch is None:
            report.update(status="REFUSED_BY_ORIGINAL_PHYSICAL_EVIDENCE", optimizer_updates=0)
        else:
            fit = fit_numeric_reference_student(
                batch,
                world_hash=positive["mesh_task_contact_world_hash"],
                parent_model_hash=positive["behavior_model_hash"],
                seed=cfg["fit_seed"],
                steps=cfg["fit_steps"],
            )
            write_once(root / "numeric-fit.json", fit)
            report.update(
                status="NUMERIC_FIT_COMPLETE_REQUIRES_OWN_BODY_EXAM",
                optimizer_updates=fit["learning_receipt"]["completed_optimizer_steps"],
                numeric_fit_hash=fit["report_hash"],
            )
        if any(digest(path) != h for path, h in pins.items()):
            raise ValueError("source or evidence changed during fit")
        report["report_hash"] = hash_json(report)
        write_once(root / "complete.json", report)
        print(report["status"], report["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(
                type=type(error).__name__,
                message=str(error),
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
