"""Fixed SIM diagnostic/readback succession, no retry/selection/promotion."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from rosclaw_soccer.sim.contracts import hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_early_feedback_screen_inputs import require_scope
from scripts.rsi_stage_release_contact_replay import digest, require_terminal


def validate_config(cfg):
    if (
        cfg.get("mode") != "SIM_ONLY"
        or cfg.get("profile") != "early80ms-zero-readback-then-fixed-screen-v1"
        or any(
            cfg.get(k) is not False
            for k in (
                "hardware_authorized",
                "promotion_authorized",
                "teacher_qualified",
                "training_authorized",
                "automatic_retry",
                "actor_future_event_input_allowed",
            )
        )
        or type(cfg.get("new_optimizer_updates")) is not int
        or cfg["new_optimizer_updates"] != 0
        or cfg.get("readback_unit") == cfg.get("zero_unit")
        or type(cfg.get("screen_template")) is not dict
    ):
        raise ValueError("fixed no-learning SIM diagnostic succession required")
    require_scope(cfg["screen_template"])
    # These identities are filled only after the independently terminating reader.
    reserved = {
        "source_root",
        "source_head",
        "output_root",
        "early_zero_unit",
        "early_zero_readback_unit",
        "early_zero_readback",
        "early_zero_readback_hash",
        "early_zero_readback_file_hash",
    }
    if reserved.intersection(cfg["screen_template"]):
        raise ValueError("successor source/output/proof identities must not be caller overrides")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--stage", choices=("readback", "screen", "screen-readback"), required=True)
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    cfg = json.loads(args.config.read_text())
    validate_config(cfg)
    source = Path(cfg["source_root"])
    stage_root = Path(cfg["stage_root"])
    if (
        Path(__file__).resolve() != source / "scripts/rsi_early_feedback_successor.py"
        or subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=source, text=True).strip()
        != cfg["source_head"]
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=source, text=True).strip()
        or digest(cfg["zero_config"]) != cfg["zero_config_file_hash"]
    ):
        raise ValueError("exact frozen source and registered zero-calibration bytes required")
    zero_cfg = json.loads(Path(cfg["zero_config"]).read_text())
    if zero_cfg["output_root"] != cfg["zero_root"]:
        raise ValueError("same registered zero study required")
    if args.preflight:
        print("FIXED_EARLY80_SUCCESSOR_READY_NO_MODEL_OR_PHYSICS", flush=True)
        return
    stage_root.mkdir(exist_ok=True)
    failure_path = stage_root / f"{args.stage}-failure.json"
    start_path = stage_root / f"{args.stage}-started.json"
    # An already-begun stage is never silently retried, even after a crash.
    # write_once is intentionally idempotent, so it is NOT an execution lock.
    # mkdir supplies the exclusive atomic claim; retain it on every exit path.
    (stage_root / f"{args.stage}-attempt").mkdir(mode=0o700)
    write_once(
        start_path,
        dict(
            stage=args.stage,
            config_file_hash=digest(args.config),
            source_head=cfg["source_head"],
            automatic_retry=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    try:
        require_terminal(cfg["zero_unit"])
        zero_root = Path(cfg["zero_root"])
        if (zero_root / "failure.json").exists():
            raise ValueError("failed zero study cannot authorize a successor")
        readback_path = stage_root / "zero-readback.json"
        if args.stage == "readback":
            command = [
                sys.executable,
                "scripts/rsi_read_zero_feedback.py",
                "--early80",
                "--root",
                str(zero_root),
                "--unit",
                cfg["zero_unit"],
                "--source-head",
                cfg["source_head"],
                "--complete-file-hash",
                digest(zero_root / "complete.json"),
                "--output",
                str(readback_path),
            ]
        elif args.stage == "screen":
            require_terminal(cfg["readback_unit"])
            d = json.loads(readback_path.read_text())
            if (
                d.get("source_head") != cfg["source_head"]
                or d.get("schema") != "soccer.rsi.zero_early_feedback_calibration_readback.v1"
                or d.get("report_hash")
                != hash_json({k: v for k, v in d.items() if k != "report_hash"})
            ):
                raise ValueError("same frozen independent reader must terminate before screen")
            screen_cfg = dict(
                cfg["screen_template"],
                source_root=str(source),
                source_head=cfg["source_head"],
                output_root=str(stage_root / "screen"),
                early_zero_unit=cfg["zero_unit"],
                early_zero_readback_unit=cfg["readback_unit"],
                early_zero_readback=str(readback_path),
                early_zero_readback_hash=d["report_hash"],
                early_zero_readback_file_hash=digest(readback_path),
            )
            require_scope(screen_cfg)
            config_path = stage_root / "screen-config.json"
            write_once(config_path, screen_cfg)
            command = [
                sys.executable,
                "scripts/rsi_calibrate_feedback_branches.py",
                "--early-screen",
                "--config",
                str(config_path),
            ]
        else:
            require_terminal(cfg["screen_unit"])
            screen_root = Path(cfg["screen_root"])
            if (screen_root / "failure.json").exists():
                raise ValueError("failed screen cannot authorize raw readback")
            command = [
                sys.executable,
                "scripts/rsi_read_feedback_screen.py",
                "--early80",
                "--root",
                str(screen_root),
                "--unit",
                cfg["screen_unit"],
                "--source-head",
                cfg["source_head"],
                "--complete-file-hash",
                digest(screen_root / "complete.json"),
                "--output",
                str(stage_root / "screen-readback.json"),
            ]
        env = dict(os.environ)
        env.update(
            OMP_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            CUDA_VISIBLE_DEVICES="",
        )
        subprocess.run(command, cwd=source, env=env, check=True)
        write_once(
            stage_root / f"{args.stage}-command-complete.json",
            dict(
                stage=args.stage,
                command=command,
                exit_code=0,
                command_success_is_not_task_success=True,
                source_head=cfg["source_head"],
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except BaseException as error:
        write_once(
            failure_path,
            dict(
                stage=args.stage,
                error_type=type(error).__name__,
                error=str(error),
                automatic_retry=False,
                teacher_qualified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
