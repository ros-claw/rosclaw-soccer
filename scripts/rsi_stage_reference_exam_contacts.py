"""Once-only whole student exam dense diagnostics after both real terminal units.

No full NN, fitting, candidate selection, new policy trial or activation. Replays
only the thirteen original recorded 2ms torque histories, including all failures.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

from rosclaw_soccer.rsi.reference_student_exam_diagnostics import measured_readout, require_cohort
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    if (
        cfg.get("diagnostic_profile") != "reference-student-thirteen"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            cfg.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "neural_training_authorized")
        )
        or type(cfg.get("new_substep_budget")) is not int
        or cfg["new_substep_budget"] != 39000
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
        raise ValueError("unused once-only frozen thirteen-case diagnostic required")
    require_terminal(cfg["producer_unit"])
    require_terminal(cfg["audit_unit"])
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original evidence seal required")
        pins[str(path)] = digest(path)
        return value

    pp = Path(cfg["producer_config"])
    if digest(pp) != cfg["producer_config_file_hash"]:
        raise ValueError("exact original producer configuration required")
    producer = json.loads(pp.read_text())
    pins[str(pp)] = digest(pp)
    root = Path(producer["output_root"])
    declaration, complete = (sealed(root / n) for n in ("declaration.json", "complete.json"))
    audit = sealed(cfg["readback"])
    paired = require_cohort(producer, declaration, complete, audit)
    if audit["reader_source_hash"] != digest(cfg["audit_source"]):
        raise ValueError("original whole own-body reader source required")
    pins[cfg["audit_source"]] = digest(cfg["audit_source"])
    for name, h in {
        **declaration["prepared_pins"],
        **declaration["input_file_hashes"],
        **audit["input_file_hashes"],
    }.items():
        if digest(name) != h:
            raise ValueError("original input/witness/source bytes changed")
        pins[name] = h
    cases = []
    for i, job in enumerate(declaration["scope"]["jobs"]):
        path = root / f"receipt-{i:04d}.json"
        receipt = sealed(path, "receipt_hash")
        if receipt != complete["results"][i]:
            raise ValueError("all thirteen exact original receipt files required")
        cases.append(
            dict(
                case_id=job["job"].replace("-", "_"),
                receipt_path=str(path),
                receipt_file_hash=digest(path),
                receipt_hash=receipt["receipt_hash"],
            )
        )
    replay_cfg = dict(
        replay_profile="reference-student-thirteen",
        source_head=cfg["source_head"],
        output_root=cfg["replay_output_root"],
        new_replay_budget=13,
        new_substep_budget=39000,
        cases=cases,
        world_hash=producer["mesh_task_contact_world_hash"],
        behavior_model_hash=producer["behavior_model_hash"],
        mujoco_version="3.12.0",
        producer_config=cfg["producer_config"],
        producer_config_file_hash=cfg["producer_config_file_hash"],
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
            schema="soccer.rsi.reference_exam_contact_started.v1",
            input_file_hashes=pins,
            upstream_readback_hash=audit["report_hash"],
            automatic_retry_authorized=False,
            new_fixed_control_substep_budget=39000,
            hardware_authorized=False,
        ),
    )
    try:
        summary = dict(
            schema="soccer.rsi.reference_exam_measured_summary.v1",
            complete_hash=complete["report_hash"],
            terminal_readback_hash=audit["report_hash"],
            rows=audit["rows"],
            paired_results=paired,
            distinct_derived_course_count=10,
            source_scene_clusters=6,
            decision="PENDING_DENSE_REPLAY_NOT_ADMISSION",
            new_optimizer_updates=0,
            fresh_exam=False,
            teacher_qualified=False,
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

        def child(script, args):
            subprocess.run(
                [sys.executable, str(source / "scripts" / script), *args],
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
                "--output",
                cfg["metrics_output"],
            ],
        )
        replay = sealed(Path(cfg["replay_output_root"]) / "complete.json")
        metrics = sealed(cfg["metrics_output"])
        measured = measured_readout(declaration, complete, replay, metrics)
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source or original evidence changed during diagnostic")
        result = dict(
            schema="soccer.rsi.reference_exam_contact_complete.v1",
            upstream_readback_hash=audit["report_hash"],
            summary_hash=summary["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            measured_contact_readout=measured,
            paired_results=paired,
            total_metrics_endpoints=13,
            distinct_derived_course_count=10,
            source_scene_clusters=6,
            new_fixed_control_replay_substeps=39000,
            new_closed_loop_policy_trials=0,
            new_neural_inferences=0,
            new_optimizer_updates=0,
            teacher_qualified=False,
            dataset_admission_authorized=False,
            neural_training_authorized=False,
            fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        result["report_hash"] = hash_json(result)
        write_once(Path(cfg["complete_path"]), result)
        print(
            "REFERENCE_EXAM_DENSE_DIAGNOSTIC_COMPLETE_NOT_PROMOTION",
            result["report_hash"],
            flush=True,
        )
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
