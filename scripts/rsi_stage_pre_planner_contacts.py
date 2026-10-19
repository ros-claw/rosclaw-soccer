"""Once-only six original pulse recorded-torque diagnostics; no policy search."""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from rosclaw_soccer.rsi.planner_navigation_pulse_cohort import (
    pulse_contact_readout,
    pulse_motion_readout,
)
from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_load_pre_planner_cohort import load_closed_pulse
from scripts.rsi_stage_release_contact_replay import digest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    source = Path(__file__).resolve().parents[1]
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("diagnostic_profile")
        not in (
            "pre-planner-navigation-six",
            "pre-planner-navigation-positive-six",
            "original-ik-pre-planner-six",
            "foot-heading-contact-six",
        )
        or any(
            cfg.get(k) is not False
            for k in ("hardware_authorized", "promotion_authorized", "neural_training_authorized")
        )
        or type(cfg.get("new_substep_budget")) is not int
        or cfg["new_substep_budget"] != 18000
        or time.time() >= cfg["deadline_unix"]
        or any(
            Path(cfg[k]).exists()
            for k in (
                "started_path",
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
        raise ValueError("unused frozen once-only six-endpoint diagnostic required")
    pins = {str(args.config): digest(args.config)}
    for folder in (source / "src", source / "scripts"):
        pins.update({str(p): digest(p) for p in folder.rglob("*.py")})

    def sealed(path, key="report_hash"):
        path = Path(path)
        value = json.loads(path.read_text())
        if value[key] != hash_json({k: v for k, v in value.items() if k != key}):
            raise ValueError("original evidence seal required")
        h = digest(path)
        if str(path) in pins and pins[str(path)] != h:
            raise ValueError("original evidence changed since authentication")
        pins[str(path)] = h
        return value

    # The audit hash is not knowable before the queued producer terminates.
    # Authenticate its sealed whole output before declaring the replay config.
    audit = sealed(cfg["readback"])
    original_cfg = json.loads(Path(cfg["producer_config"]).read_text())
    root = Path(original_cfg["output_root"])
    inputs = dict(
        producer_unit=cfg["producer_unit"],
        audit_unit=cfg["audit_unit"],
        producer_config=cfg["producer_config"],
        producer_config_file_hash=cfg["producer_config_file_hash"],
        producer_declaration=str(root / "declaration.json"),
        producer_complete=str(root / "complete.json"),
        source_terminal_readback=cfg["readback"],
        source_terminal_readback_hash=audit["report_hash"],
        audit_source=cfg["audit_source"],
    )
    if cfg["diagnostic_profile"] == "foot-heading-contact-six":
        from rosclaw_soccer.rsi.foot_heading_diagnostics import (
            heading_contact_readout as contact_readout,
        )
        from rosclaw_soccer.rsi.foot_heading_diagnostics import (
            heading_motion_readout as motion_readout,
        )
        from rosclaw_soccer.rsi.foot_heading_scope import PROFILE as HEADING_PROFILE
        from scripts.rsi_load_heading_cohort import load_closed_heading

        cohort = load_closed_heading(inputs, sealed=sealed, pins=pins)
    elif cfg["diagnostic_profile"] == "original-ik-pre-planner-six":
        from rosclaw_soccer.rsi.original_ik_planner_diagnostics import (
            ik_contact_readout as contact_readout,
        )
        from rosclaw_soccer.rsi.original_ik_planner_diagnostics import (
            ik_motion_readout as motion_readout,
        )
        from rosclaw_soccer.rsi.original_ik_planner_scope import PROFILE as IK_PROFILE
        from scripts.rsi_load_original_ik_cohort import load_closed_original_ik

        cohort = load_closed_original_ik(inputs, sealed=sealed, pins=pins)
    else:
        contact_readout, motion_readout = pulse_contact_readout, pulse_motion_readout
        cohort = load_closed_pulse(inputs, sealed=sealed, pins=pins)
    producer, declaration, complete = (cohort[k] for k in ("producer", "declaration", "complete"))
    from rosclaw_soccer.rsi.planner_navigation_pulse_scope import POSITIVE_PROFILE, PROFILE

    expected = (
        HEADING_PROFILE
        if cfg["diagnostic_profile"] == "foot-heading-contact-six"
        else IK_PROFILE
        if cfg["diagnostic_profile"] == "original-ik-pre-planner-six"
        else POSITIVE_PROFILE
        if cfg["diagnostic_profile"] == "pre-planner-navigation-positive-six"
        else PROFILE
    )
    if producer["exam_profile"] != expected:
        raise ValueError("exact producer and diagnostic polarity pairing required")
    cases = [
        dict(
            case_id=job["job"].replace("-", "_"),
            receipt_path=str(root / f"receipt-{i:04d}.json"),
            receipt_file_hash=digest(root / f"receipt-{i:04d}.json"),
            receipt_hash=r["receipt_hash"],
        )
        for i, (job, r) in enumerate(
            zip(declaration["scope"]["jobs"], complete["results"], strict=True)
        )
    ]
    replay_cfg = dict(
        inputs,
        replay_profile=cfg["diagnostic_profile"],
        source_head=cfg["source_head"],
        output_root=cfg["replay_output_root"],
        new_replay_budget=6,
        new_substep_budget=18000,
        cases=cases,
        world_hash=producer["mesh_task_contact_world_hash"],
        behavior_model_hash=producer["behavior_model_hash"],
        mujoco_version="3.12.0",
        hardware_authorized=False,
        promotion_authorized=False,
        neural_training_authorized=False,
    )
    schema_prefix = (
        "foot_heading"
        if cfg["diagnostic_profile"] == "foot-heading-contact-six"
        else "original_ik_planner"
        if cfg["diagnostic_profile"] == "original-ik-pre-planner-six"
        else "pre_planner_pulse"
    )
    write_once(
        Path(cfg["started_path"]),
        dict(
            schema=f"soccer.rsi.{schema_prefix}_contact_started.v1",
            input_file_hashes=pins,
            upstream_readback_hash=audit["report_hash"],
            automatic_retry_authorized=False,
            new_fixed_control_substep_budget=18000,
            hardware_authorized=False,
        ),
    )
    try:
        write_once(Path(cfg["replay_config"]), replay_cfg)
        environment = dict(
            os.environ,
            PYTHONPATH=os.pathsep.join(
                (str(source / "src"), str(source), str(Path(producer["core_root"]) / "src"))
            ),
        )

        def child(script, argv):
            subprocess.run(
                [sys.executable, str(source / "scripts" / script), *argv],
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
        measured = contact_readout(declaration, complete, replay, metrics)
        motion = []
        for i in range(0, 6, 2):
            traces = []
            for r in complete["results"][i : i + 2]:
                witness = r["witnesses"]["trace"]
                if (
                    pins.get(witness["path"]) != witness["sha256"]
                    or digest(witness["path"]) != witness["sha256"]
                ):
                    raise ValueError("exact original audited pulse traces required")
                with np.load(witness["path"], allow_pickle=False) as arrays:
                    traces.append(dict(arrays))
            motion.append(
                dict(
                    group=declaration["scope"]["jobs"][i]["group"],
                    off_receipt_hash=complete["results"][i]["receipt_hash"],
                    pulse_receipt_hash=complete["results"][i + 1]["receipt_hash"],
                    diagnostic=motion_readout(*traces),
                )
            )
        if any(digest(p) != h for p, h in pins.items()):
            raise ValueError("source or original evidence changed during diagnostic")
        result = dict(
            schema=f"soccer.rsi.{schema_prefix}_contact_complete.v1",
            upstream_readback_hash=audit["report_hash"],
            replay_complete_hash=replay["report_hash"],
            metrics_hash=metrics["report_hash"],
            measured_contact_readout=measured,
            paired_motion_diagnostics=motion,
            input_file_hashes=pins,
            total_metrics_endpoints=6,
            distinct_derived_course_count=3,
            source_scene_clusters=3,
            new_fixed_control_replay_substeps=18000,
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
            "SIX_PULSE_DENSE_DIAGNOSTIC_COMPLETE_NOT_PROMOTION", result["report_hash"], flush=True
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
