"""Once-only fixed diagnostic succession after the whole80ms readback.

Only proof hashes become known at execution time. The frozen worker, six-case
scope, action protocol and budgets are declared beforehand. No policy
selection, optimizer update, retry or promotion is provided here.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_early_impact_pilot_inputs import require_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def validate_declaration(cfg):
    if (
        cfg.get("schema") != "soccer.rsi.deferred_early_h300.v1"
        or cfg.get("mode") != "SIM_ONLY"
        or any(
            cfg.get(k) is not False
            for k in (
                "automatic_retry",
                "hardware_authorized",
                "promotion_authorized",
                "teacher_qualified",
            )
        )
        or type(cfg.get("worker_config_template")) is not dict
    ):
        raise ValueError("fixed no-retry SIM-only H300 successor declaration required")
    template = cfg["worker_config_template"]
    require_scope(template)
    if (
        "screen_readback_hash" in template
        or "screen_readback_file_hash" in template
        or template.get("screen_readback") != cfg.get("readback_path")
        or template.get("screen_readback_unit") != cfg.get("readback_unit")
    ):
        raise ValueError("only actual terminal readback may bind predecessor hashes")


def clean_head(root, head):
    if (
        subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip() != head
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    ):
        raise ValueError("exact clean frozen successor/worker sources required")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_declaration(cfg)
    source = Path(cfg["source_root"])
    template = cfg["worker_config_template"]
    worker = Path(template["source_root"])
    root = Path(cfg["output_root"])
    if Path(__file__).resolve() != source / "scripts/rsi_deferred_early_h300.py" or root.exists():
        raise ValueError("exact successor entry and unused output required")
    clean_head(source, cfg["source_head"])
    clean_head(worker, template["source_head"])
    if args.preflight:
        print("DEFERRED_STATIC_READY_PREDECESSOR_PROOFS_NOT_YET_ADMITTED", flush=True)
        return
    root.mkdir(mode=0o700)  # Exclusive attempt claim; retained on every exit path.
    write_once(root / "declaration.json", dict(cfg, config_file_hash=digest(args.config)))
    child_started = False
    try:
        require_terminal(cfg["readback_unit"])
        file = Path(cfg["readback_path"])
        if file.is_symlink():
            raise ValueError("ordinary independent screen readback required")
        d = json.loads(file.read_text())
        if (
            d.get("schema") != "soccer.rsi.early_feedback_impact_screen_readback.v1"
            or d.get("source_head") != cfg["readback_source_head"]
            or d.get("report_hash") != hash_json({k: v for k, v in d.items() if k != "report_hash"})
        ):
            raise ValueError("same whole independently terminated readback required")
        full = dict(
            template,
            screen_readback_hash=d["report_hash"],
            screen_readback_file_hash=digest(file),
        )
        require_scope(full)
        path = root / "worker-config.json"
        write_once(path, full)
        command = [
            sys.executable,
            str(worker / "scripts/rsi_early_impact_h300.py"),
            "--config",
            str(path),
        ]
        env = dict(os.environ)
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )
        # This preflight authenticates all six actual precursor chains and OFF
        # calibration arrays before permitting any full-model allocation.
        subprocess.run([*command, "--preflight"], cwd=worker, env=env, check=True)
        write_once(
            root / "worker-preflight-complete.json",
            dict(worker_config_hash=digest(path), physical_steps=0, full_model_allocations=0),
        )
        clean_head(source, cfg["source_head"])
        clean_head(worker, template["source_head"])
        child_started = True
        subprocess.run(command, cwd=worker, env=env, check=True)
        write_once(
            root / "command-complete.json",
            dict(
                exit_code=0,
                worker_output_root=template["output_root"],
                command_success_is_not_task_success=True,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except BaseException as error:
        write_once(
            root / "failure.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                physical_worker_started=child_started,
                worker_counters_must_be_read_from=template["output_root"],
                automatic_retry=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
