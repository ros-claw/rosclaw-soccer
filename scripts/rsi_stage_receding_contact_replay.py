"""Once-only ten-endpoint diagnostics after the complete six-course receding audit.

No NN, optimizer, new policy trials, teacher admission or scientific retry.
Two previously integrated None controls are reused, never re-integrated here.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs
from rosclaw_soccer.rsi.receding_teacher_scope import SOURCE_GROUPS, receding_jobs
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def require_receding_endpoints(config, declaration, complete, audit):
    jobs = receding_jobs()
    if (
        config.get("mode") != "SIM_ONLY"
        or config.get("groups") != list(SOURCE_GROUPS)
        or type(config.get("max_attempts")) is not int
        or config["max_attempts"] != 10
        or type(config.get("new_physical_budget")) is not int
        or config["new_physical_budget"] != 10
        or config.get("selected_option_index") != 2
        or any(
            config.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "neural_training_authorized")
        )
        or declaration.get("schema") != "soccer.rsi.receding_catalog2_declaration.v1"
        or declaration.get("jobs") != jobs
        or complete.get("schema") != "soccer.rsi.receding_catalog2_complete.v1"
        or audit.get("schema") != "soccer.rsi.receding_catalog2_terminal_readback.v1"
        or complete.get("declaration_hash") != declaration["report_hash"]
        or audit.get("complete_hash") != complete["report_hash"]
        or any(
            type(complete.get(k)) is not int or complete[k] != n
            for k, n in (
                ("new_native_bodies", 10),
                ("native_substeps", 30000),
                ("independent_cpu_substeps", 30000),
                ("independent_course_count", 6),
                ("paired_endpoint_count", 12),
                ("new_optimizer_updates", 0),
            )
        )
        or type(audit.get("source_native_bodies")) is not int
        or audit["source_native_bodies"] != 10
        or any(
            audit.get(k) is not True
            for k in (
                "whole_input_and_witness_files_verified",
                "shared_payload_logical_seals_rebuilt",
                "prior_independent_replays_authenticated",
                "archived_c7_full_numeric_reproduction",
                "reused_endpoints_authenticated_via_prior_complete_terminal_readbacks",
                "single_high_x_receding_stratum_only",
            )
        )
        or complete.get("archived_c7_full_numeric_reproduction") is not True
        or any(
            document.get(k) is not False
            for document in (complete, audit)
            for k in ("teacher_qualified", "hardware_authorized", "promotion_authorized")
        )
        or audit.get("fresh_exam") is not False
        or len(complete.get("results", [])) != 10
        or len(audit.get("rows", [])) != 10
        or len(complete.get("reused_endpoints", [])) != 2
        or complete["reused_endpoints"] != declaration.get("reused_endpoints")
    ):
        raise ValueError("whole ten-body audit, reproduction and six-pair scope required")
    controls, teachers = {}, {}
    for group, reused in zip(SOURCE_GROUPS[:2], complete["reused_endpoints"], strict=True):
        if reused["pair_id"] != f"receding-g{group}":
            raise ValueError("exact two original None controls required")
        controls[reused["pair_id"]] = reused["receipt"]["outcome"]
    for job, receipt, row in zip(jobs, complete["results"], audit["rows"], strict=True):
        if (
            any(receipt.get(k) != v for k, v in job.items())
            or row["job"] != job["job"]
            or row["receipt_hash"] != receipt["receipt_hash"]
            or row["outcome"] != receipt["outcome"]
            or row["world_hash"] != config["mesh_task_contact_world_hash"]
        ):
            raise ValueError("all ordered receipts and outcomes, including failures, required")
        (controls if job["role"] == "control" else teachers)[job["pair_id"]] = receipt["outcome"]
    paired = assess_pairs(controls, teachers)
    if paired != complete["paired_development"] or paired != audit["paired_development"]:
        raise ValueError("six complete original pairs must reconstruct without outcome filtering")
    return paired


def require_dense_union(complete, prior, replay, metrics):
    expected = {row["source_receipt_hash"]: row["original_outcome"] for row in prior["rows"]}
    fresh = {r["receipt_hash"]: r["outcome"] for r in complete["results"]}
    if (
        len(expected) != 12
        or len(fresh) != 10
        or expected.keys() & fresh.keys()
        or replay.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or replay.get("integrated_substeps") != 30000
        or replay.get("attempted_step_calls") != 30000
        or [r["source_receipt_hash"] for r in replay.get("rows", [])] != list(fresh)
        or [r["original_outcome"] for r in replay["rows"]] != list(fresh.values())
        or metrics.get("schema") != "soccer.rsi.contact_exit_metrics_readback.v1"
        or len(metrics.get("rows", [])) != 22
    ):
        raise ValueError("exact ten new and twelve old dense endpoints required")
    expected.update(fresh)
    rows = {r["source_receipt_hash"]: r for r in metrics["rows"]}
    if (
        len(rows) != 22
        or rows.keys() != expected.keys()
        or any(rows[h]["original_outcome"] != outcome for h, outcome in expected.items())
    ):
        raise ValueError("dense union must retain all original outcomes without duplication")
    selected = [r["receipt"]["receipt_hash"] for r in complete["reused_endpoints"]] + list(fresh)
    if len(set(selected)) != 12:
        raise ValueError("six paired courses must have twelve distinct original endpoints")
    return [rows[h] for h in selected]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    output_keys = (
        "started_path",
        "summary_output",
        "replay_config",
        "replay_output_root",
        "metrics_output",
        "complete_path",
    )
    if (
        time.time() >= cfg["deadline_unix"]
        or cfg.get("new_substep_budget") != 30000
        or type(cfg.get("new_substep_budget")) is not int
        or any(Path(cfg[k]).exists() for k in output_keys)
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused bounded frozen post-audit stage required")
    require_terminal(cfg["producer_unit"])
    require_terminal(cfg["audit_unit"])
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original sealed evidence changed")
        pins[str(path)] = digest(path)
        return value

    producer = json.loads(Path(cfg["producer_config"]).read_text())
    pins[cfg["producer_config"]] = digest(cfg["producer_config"])
    root = Path(producer["output_root"])
    declaration = sealed(root / "declaration.json")
    complete = sealed(root / "complete.json")
    audit = sealed(cfg["readback"])
    paired = require_receding_endpoints(producer, declaration, complete, audit)
    if audit["reader_source_hash"] != digest(cfg["audit_source"]):
        raise ValueError("original independently frozen full terminal reader required")
    pins[cfg["audit_source"]] = digest(cfg["audit_source"])
    for name, h in declaration["input_file_hashes"].items():
        if digest(name) != h:
            raise ValueError("whole upstream audit input bytes changed")
        pins[name] = h
    prior = sealed(cfg["prior_replay_complete"])
    if (
        prior["report_hash"] != cfg["prior_replay_complete_hash"]
        or prior.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
        or prior.get("integrated_substeps") != 36000
        or len(prior.get("rows", [])) != 12
        or [r["case_id"] for r in prior["rows"]]
        != [f"{f}_context_{i}" for i in range(2, 8) for f in ("none", "option4")]
        or any(
            prior["rows"][index]["source_receipt_hash"] != reused["receipt"]["receipt_hash"]
            for index, reused in zip((8, 10), complete["reused_endpoints"], strict=True)
        )
    ):
        raise ValueError("exact already-replayed None C6/C7 required; never replay twice")
    cases = []
    for index, job in enumerate(receding_jobs()):
        path = root / f"receipt-{index:04d}.json"
        receipt = sealed(path, "receipt_hash")
        if receipt != complete["results"][index]:
            raise ValueError("exact complete-study receipt bytes required")
        cases.append(
            dict(
                case_id=job["job"].replace("-", "_"),
                receipt_path=str(path),
                receipt_file_hash=digest(path),
                receipt_hash=receipt["receipt_hash"],
            )
        )
    replay_cfg = dict(
        replay_profile="receding-catalog2-ten",
        source_head=cfg["source_head"],
        output_root=cfg["replay_output_root"],
        new_replay_budget=10,
        new_substep_budget=30000,
        cases=cases,
        world_hash=producer["mesh_task_contact_world_hash"],
        mujoco_version="3.12.0",
        producer_config=cfg["producer_config"],
        producer_complete=str(root / "complete.json"),
        producer_declaration=str(root / "declaration.json"),
        source_terminal_readback=cfg["readback"],
        source_terminal_readback_hash=audit["report_hash"],
        audit_source=cfg["audit_source"],
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
    )
    write_once(
        Path(cfg["started_path"]),
        dict(
            schema="soccer.rsi.receding_contact_diagnostic_started.v1",
            input_file_hashes=pins,
            upstream_readback_hash=audit["report_hash"],
            automatic_retry_authorized=False,
            new_physical_substep_budget=30000,
        ),
    )
    try:
        summary = dict(
            schema="soccer.rsi.receding_catalog2_measured_summary.v1",
            complete_hash=complete["report_hash"],
            terminal_readback_hash=audit["report_hash"],
            paired_development=paired,
            rows=audit["rows"],
            reused_endpoints=complete["reused_endpoints"],
            independent_course_count=6,
            paired_endpoint_count=12,
            new_optimizer_updates=0,
            archived_c7_full_numeric_reproduction=True,
            teacher_qualified=False,
            decision="SCOPED_RESEARCH_ONLY_REQUIRES_CONTACT_AND_RETENTION_REVIEW",
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        summary["report_hash"] = hash_json(summary)
        write_once(Path(cfg["summary_output"]), summary)
        write_once(Path(cfg["replay_config"]), replay_cfg)
        env = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(source / "src"), str(source), str(Path(producer["core_root"]) / "src"))
            ),
        )

        def child(script, arguments):
            subprocess.run(
                [sys.executable, str(source / "scripts" / script), *arguments],
                cwd=source,
                env=env,
                check=True,
                timeout=2400,
            )

        child(
            "rsi_replay_first_touch_contacts.py", ["--config", cfg["replay_config"], "--preflight"]
        )
        child("rsi_replay_first_touch_contacts.py", ["--config", cfg["replay_config"]])
        child(
            "rsi_read_contact_exit_metrics.py",
            [
                "--replay-complete",
                str(Path(cfg["replay_output_root"]) / "complete.json"),
                "--replay-complete",
                cfg["prior_replay_complete"],
                "--output",
                cfg["metrics_output"],
            ],
        )
        replay = sealed(Path(cfg["replay_output_root"]) / "complete.json")
        metrics = sealed(cfg["metrics_output"])
        scoped_metrics = require_dense_union(complete, prior, replay, metrics)
        if any(digest(name) != h for name, h in pins.items()):
            raise ValueError("stage/source/upstream bytes changed during diagnostics")
        result = dict(
            schema="soccer.rsi.receding_contact_diagnostic_complete.v1",
            upstream_readback_hash=audit["report_hash"],
            summary_hash=summary["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            new_fixed_control_replay_substeps=30000,
            total_metrics_endpoints=22,
            scoped_paired_endpoints=12,
            scoped_independent_courses=6,
            scoped_contact_metrics=scoped_metrics,
            metrics_union_independent_courses=10,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(Path(cfg["complete_path"]), result)
        print("RECEDING_CONTACT_DIAGNOSTIC_COMPLETE", result["report_hash"], flush=True)
    except BaseException as error:
        write_once(
            Path(cfg["started_path"]).with_suffix(".failure.json"),
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
