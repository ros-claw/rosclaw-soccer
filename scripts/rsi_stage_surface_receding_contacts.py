"""Once-only six-case contact diagnostic and three-family research readout.

Run only after both actual whole-stage units terminate successfully. Replay
18000 NEW fixed-control substeps; reuse all prior 22 endpoints without physics.
No NN, fitting, teacher admission, repeat policy trials or automatic retry.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rosclaw_soccer.rsi.surface_receding_diagnostics import (
    require_endpoints,
    research_decision,
    scoped_metrics,
)
from rosclaw_soccer.rsi.surface_receding_evidence import load_baselines, surface_receding_jobs
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_replay_first_touch_contacts import fixed_replay_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    if (
        cfg.get("diagnostic_profile") != "surface-receding-six"
        or type(cfg.get("new_substep_budget")) is not int
        or cfg["new_substep_budget"] != 18000
        or time.time() >= cfg["deadline_unix"]
        or any(
            Path(cfg[k]).exists()
            for k in (
                "started_path",
                "summary_output",
                "replay_config",
                "replay_output_root",
                "metrics_output",
                "complete_path",
            )
        )
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
    ):
        raise ValueError("unused once-only frozen six-case post-audit diagnostic required")
    require_terminal(cfg["producer_unit"])
    require_terminal(cfg["audit_unit"])
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        value = json.loads(Path(path).read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("exact original evidence seal required")
        pins[str(path)] = digest(path)
        return value

    producer = json.loads(Path(cfg["producer_config"]).read_text())
    pins[cfg["producer_config"]] = digest(cfg["producer_config"])
    root = Path(producer["output_root"])
    declaration, complete = (sealed(root / name) for name in ("declaration.json", "complete.json"))
    audit = sealed(cfg["readback"])
    paired, retention = require_endpoints(producer, declaration, complete, audit)
    if audit["reader_source_hash"] != digest(cfg["audit_source"]):
        raise ValueError("exact original whole six-body reader required")
    pins[cfg["audit_source"]] = digest(cfg["audit_source"])
    for name, h in {**declaration["prepared_pins"], **declaration["input_file_hashes"]}.items():
        if digest(name) != h:
            raise ValueError("producer source/prepared-input bytes changed")
        pins[name] = h
    # Whole original controls, default-OFF source identity and clean C6/C7 are
    # independently reopened before any new fixed-control physics is charged.
    load_baselines(producer, Path(producer["source_root"]), sealed, digest, pins)
    old_metrics = sealed(producer["receding_predecessor"]["metrics"])
    previous = cfg.get("prior_replays")
    if type(previous) is not list or len(previous) != 2:
        raise ValueError("exact ten-endpoint and twelve-endpoint old replay sets required")
    old_replays = []
    for item, expected_profile in zip(
        previous, ("receding-catalog2-ten", "context-baseline-twelve"), strict=True
    ):
        path = Path(item["path"])
        if pins.get(str(path)) != digest(path) or digest(path) != item["file_hash"]:
            raise ValueError("old replay must belong to the whole upstream dense audit")
        prior = sealed(path)
        prior_declaration = sealed(path.parent / "declaration.json")
        count, steps = fixed_replay_scope(prior_declaration)
        if (
            prior["report_hash"] != item["report_hash"]
            or prior.get("schema") != "soccer.rsi.fixed_control_contact_replay_complete.v1"
            or prior.get("declaration_hash") != prior_declaration["report_hash"]
            or prior_declaration.get("replay_profile") != expected_profile
            or prior_declaration["world_hash"] != producer["mesh_task_contact_world_hash"]
            or prior.get("integrated_substeps") != steps
            or prior.get("attempted_step_calls") != steps
            or len(prior.get("rows", [])) != count
        ):
            raise ValueError("exact old budgets, world and whole replay membership required")
        old_replays.append(prior)
    historical = {
        r["source_receipt_hash"]: r["original_outcome"] for old in old_replays for r in old["rows"]
    }
    if len(historical) != 22 or historical != {
        r["source_receipt_hash"]: r["original_outcome"] for r in old_metrics["rows"]
    }:
        raise ValueError("both reused replay sets must equal the previously audited whole union")
    cases = []
    for index, job in enumerate(surface_receding_jobs()):
        path = root / f"receipt-{index:04d}.json"
        receipt = sealed(path, "receipt_hash")
        if receipt != complete["results"][index]:
            raise ValueError("all exact original six receipts required")
        cases.append(
            dict(
                case_id=job["job"].replace("-", "_"),
                receipt_path=str(path),
                receipt_file_hash=digest(path),
                receipt_hash=receipt["receipt_hash"],
            )
        )
    replay_cfg = dict(
        replay_profile="surface-receding-six",
        source_head=cfg["source_head"],
        output_root=cfg["replay_output_root"],
        new_replay_budget=6,
        new_substep_budget=18000,
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
            schema="soccer.rsi.surface_receding_diagnostic_started.v1",
            input_file_hashes=pins,
            automatic_retry_authorized=False,
            new_fixed_control_substep_budget=18000,
            upstream_readback_hash=audit["report_hash"],
            hardware_authorized=False,
        ),
    )
    try:
        summary = dict(
            schema="soccer.rsi.surface_receding_measured_summary.v1",
            complete_hash=complete["report_hash"],
            terminal_readback_hash=audit["report_hash"],
            paired_development=paired,
            original_none_context_retention=retention,
            rows=audit["rows"],
            distinct_derived_course_count=6,
            source_scene_clusters=3,
            decision="PENDING_DENSE_REPLAY_NOT_ADMISSION",
            teacher_qualified=False,
            fresh_exam=False,
            new_optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        summary["report_hash"] = hash_json(summary)
        write_once(Path(cfg["summary_output"]), summary)
        write_once(Path(cfg["replay_config"]), replay_cfg)
        environment = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(source / "src"), str(source), str(Path(producer["core_root"]) / "src"))
            ),
        )

        def child(script, arguments):
            subprocess.run(
                [sys.executable, str(source / "scripts" / script), *arguments],
                cwd=source,
                env=environment,
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
                *[arg for item in previous for arg in ("--replay-complete", item["path"])],
                "--output",
                cfg["metrics_output"],
            ],
        )
        replay = sealed(Path(cfg["replay_output_root"]) / "complete.json")
        metrics = sealed(cfg["metrics_output"])
        selected = scoped_metrics(complete, old_replays, replay, metrics)
        decision = research_decision(paired, selected)
        if any(digest(path) != h for path, h in pins.items()):
            raise ValueError("stage/source/original bytes changed during diagnostics")
        result = dict(
            schema="soccer.rsi.surface_receding_contact_diagnostic_complete.v1",
            upstream_readback_hash=audit["report_hash"],
            summary_hash=summary["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            scoped_contact_metrics=selected,
            research_decision=decision,
            paired_development=paired,
            original_none_context_retention=retention,
            new_fixed_control_replay_substeps=18000,
            new_closed_loop_policy_trials=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            total_metrics_endpoints=28,
            metrics_union_distinct_derived_courses=10,
            metrics_union_source_scene_clusters=5,
            scoped_three_family_endpoints=18,
            scoped_distinct_derived_courses=6,
            scoped_source_scene_clusters=3,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            neural_training_authorized=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(Path(cfg["complete_path"]), result)
        print("SURFACE_RECEDING_DIAGNOSTIC_COMPLETE_NOT_TEACHER", result["report_hash"], flush=True)
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
