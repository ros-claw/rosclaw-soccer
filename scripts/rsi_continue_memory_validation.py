"""Bounded SIM_ONLY pilot -> CPU replay -> current-parent bank continuation.

This consumes an atomically published candidate, never a partial model. Each
process must exit successfully before its evidence is read. No fresh exams,
promotion, video, hardware, retries or automatic changes of parent are allowed.
"""

import argparse
import os
import signal
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_protected_phase_bank_validation import validate_bank_models
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_preflight_memory_bank_validation import check_pilot
from scripts.rsi_train_protected_online_motor_v308 import _head


def transport_flags(*, compressed: bool, shared: bool, proof: bool, reserve: bool) -> list[str]:
    if any(type(v) is not bool for v in (compressed, shared, proof, reserve)):
        raise ValueError("explicit boolean validation transport options required")
    if shared and not compressed or proof != reserve or compressed != proof:
        raise ValueError("lossless transport requires paired actual proof and system reserve")
    return (["--compressed-reports"] if compressed else []) + (
        ["--shared-model-reports"] if shared else []
    )


def run_stage(root: Path, name: str, command: list[str], *, timeout: int = 28800) -> None:
    write_once(root / f"{name}-started.json", dict(command=command, timeout_s=timeout))
    with (root / f"{name}.log").open("x") as log:
        process = subprocess.Popen(
            command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            # Only this stage's newly created, owned process group is stopped.
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=15)
            write_once(root / f"{name}-failed.json", dict(reason="OWNED_SIM_STAGE_TIMEOUT"))
            raise
    write_once(root / f"{name}-finished.json", dict(exit_code=code))
    if code:
        raise subprocess.CalledProcessError(code, command)
    print(f"MEMORY_CONTINUATION_STAGE_COMPLETED {name}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "candidate",
        "baseline",
        "baseline-pilot",
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "scene",
        "bank",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--baseline-cpu-root", type=Path, action="append", required=True)
    parser.add_argument(
        "--baseline-cpu-review-name",
        choices=("review.json", "independent_review.json"),
        default="review.json",
    )
    parser.add_argument("--compressed-reports", action="store_true")
    parser.add_argument("--shared-model-reports", action="store_true")
    parser.add_argument("--compressed-transport-review", type=Path)
    parser.add_argument("--system-reserve-path", type=Path)
    args = parser.parse_args()
    if len(args.baseline_cpu_root) != 4:
        parser.error("four ordered actual-parent CPU reference folders required")
    source = Path(__file__).resolve().parent.parent
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
    ).strip():
        raise ValueError("immutable clean execution checkout required")
    flags = transport_flags(
        compressed=args.compressed_reports,
        shared=args.shared_model_reports,
        proof=args.compressed_transport_review is not None,
        reserve=args.system_reserve_path is not None,
    )
    candidate = load_json_artifact(args.candidate)
    baseline = load_json_artifact(args.baseline)
    validate_bank_models(candidate, baseline)
    if candidate["schema"] not in (
        "soccer.rsi.smooth_memory_motor.v1",
        "soccer.rsi.consolidated_smooth_motor.v1",
        "soccer.rsi.current_memory_guarded_motor.v1",
    ):
        raise ValueError("declared smooth or consolidated-memory learning required")
    if (
        candidate["schema"]
        in ("soccer.rsi.consolidated_smooth_motor.v1", "soccer.rsi.current_memory_guarded_motor.v1")
        and not args.shared_model_reports
    ):
        raise ValueError("consolidated continuation requires explicitly proved shared reports")
    root = args.output_root
    root.mkdir(parents=True, exist_ok=False)
    commitment: dict[str, Any] = dict(
        schema="soccer.rsi.memory_validation_continuation.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        candidate_model_hash=candidate["model_hash"],
        parent_model_hash=baseline["model_hash"],
        candidate_file_hash=hash_bytes(args.candidate.read_bytes()),
        baseline_file_hash=hash_bytes(args.baseline.read_bytes()),
        source_hash=hash_bytes(Path(__file__).read_bytes()),
        g1_asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        scene_hash=hash_bytes(args.scene.read_bytes()),
        baseline_pilot_hash=_sealed(args.baseline_pilot)["report_hash"],
        bank_hash=_sealed(args.bank)["report_hash"],
        activation_ceiling="SIM_ONLY",
        partition="TRAIN_CONSUMED",
        physical_executions_planned=172,
        fresh_holdout_open_authorized=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    if args.compressed_reports:
        transport = _sealed(args.compressed_transport_review)
        if (
            transport["schema"] != "soccer.rsi.physical_report_transport_review.v1"
            or transport["model_hash"] != candidate["model_hash"]
            or transport["physical_executions_added"] != 3
            or transport["actual_motor_actions_reconstructed"] != 600
            or transport["complete_payload_equal"] is not True
            or transport["body_and_ball_trace_hashes_equal"] is not True
            or transport["qualification"] != "STORAGE_EQUIVALENCE_ONLY_NOT_LEARNING_GAIN"
            or args.shared_model_reports
            and transport.get("physical_report_representation") != "lossless_shared_model_gzip_json"
            or any(
                transport.get(k) is not False
                for k in ("promotion_authorized", "hardware_authorized")
            )
        ):
            raise ValueError(
                "bound actual shared transport equivalence required before continuation"
            )
        from scripts.rsi_compressed_bank_storage import capacity_check

        budget = (
            args.candidate.stat().st_size * 12 + args.baseline.stat().st_size * 2 + 512 * 1024**2
        )
        capacity = capacity_check(root, args.system_reserve_path, budget)
        write_once(
            root / "transport-preflight.json",
            dict(
                transport_review_hash=transport["report_hash"],
                transport_file_hash=hash_bytes(args.compressed_transport_review.read_bytes()),
                flags=flags,
                capacity=capacity,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        commitment["physical_report_representation"] = (
            "lossless_shared_model_gzip_json" if args.shared_model_reports else "lossless_gzip_json"
        )
        commitment["transport_review_hash"] = transport["report_hash"]
        commitment["transport_file_hash"] = hash_bytes(
            args.compressed_transport_review.read_bytes()
        )
        commitment["transport_preflight_hash"] = hash_json(
            load_json_artifact(root / "transport-preflight.json")
        )
    write_once(root / "commitment.json", commitment)
    common = [
        "--isaac-python",
        str(args.isaac_python),
        "--g1-usd",
        str(args.g1_usd),
        "--model-root",
        str(args.model_root),
        "--late-swing-policy",
        str(args.late_swing_policy),
        "--core-root",
        str(args.core_root),
    ]
    python = sys.executable

    def script(name: str) -> list[str]:
        return [python, "-u", str(source / "scripts" / name)]

    display = ["xvfb-run", "-a", "--server-args=-screen 0 1280x720x24 -nolisten tcp"]
    pilot_root = root / "pilot"
    run_stage(
        root,
        "pilot",
        display
        + script("rsi_collect_online_step_validation.py")
        + common
        + flags
        + [
            "--output-root",
            str(pilot_root),
            "--step-model",
            str(args.baseline),
            "--online-model",
            str(args.candidate),
            "--pilot-summary",
            str(args.baseline_pilot),
        ],
    )
    run_stage(
        root,
        "pilot-review",
        script("rsi_review_step_motor_physics.py")
        + [
            "--root",
            str(pilot_root),
            "--output",
            str(pilot_root / "independent_review.json"),
        ],
    )
    check_pilot(
        _sealed(pilot_root / "validation_summary.json"),
        _sealed(pilot_root / "independent_review.json"),
        candidate["model_hash"],
        baseline_hash=baseline["model_hash"],
    )

    def cpu(index: int) -> None:
        seed, lane = COURSES[index]
        folder = root / f"cpu-case{index}"
        run_stage(
            root,
            f"cpu-{index}",
            script("rsi_mujoco_motor_transfer.py")
            + [
                "--scene",
                str(args.scene),
                "--model-root",
                str(args.model_root),
                "--late-swing-policy",
                str(args.late_swing_policy),
                "--output-root",
                str(folder),
                "--step-model",
                str(args.candidate),
                "--consumed-bank",
                str(args.bank),
                "--seed",
                str(seed),
                "--lane",
                str(lane),
            ]
            + (["--compressed-report"] if args.compressed_reports else []),
            timeout=3600,
        )
        run_stage(
            root,
            f"cpu-{index}-review",
            script("rsi_review_cpu_motor_transfer.py")
            + [
                "--root",
                str(folder),
                "--source",
                str(source / "scripts/rsi_mujoco_motor_transfer.py"),
                "--output",
                str(folder / "independent_review.json"),
            ],
            timeout=3600,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(cpu, range(4)))
    preflight = script("rsi_preflight_memory_bank_validation.py") + [
        "--candidate",
        str(args.candidate),
        "--baseline",
        str(args.baseline),
        "--pilot-root",
        str(pilot_root),
        "--storage-root",
        str(root),
        "--output",
        str(root / "bank-preflight.json"),
        "--baseline-cpu-review-name",
        args.baseline_cpu_review_name,
    ]
    for index, reference in enumerate(args.baseline_cpu_root):
        preflight += [
            "--cpu-root",
            str(root / f"cpu-case{index}"),
            "--baseline-cpu-root",
            str(reference),
        ]
    if args.compressed_reports:
        preflight += [
            "--compressed-transport-review",
            str(args.compressed_transport_review),
            "--system-reserve-path",
            str(args.system_reserve_path),
        ]
    run_stage(root, "bank-preflight", preflight)
    bank_root = root / "bank"
    bank_command = (
        script("rsi_collect_protected_phase_bank_validation.py")
        + common
        + flags
        + [
            "--output-root",
            str(bank_root),
            "--bank-path",
            str(args.bank),
            "--warm-model",
            str(args.baseline),
            "--candidate-model",
            str(args.candidate),
        ]
    )
    run_stage(root, "bank", display + bank_command)
    run_stage(root, "bank-review", bank_command + ["--review-only", "--review-workers", "4"])
    if (
        load_json_artifact(root / "commitment.json") != commitment
        or _head(source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
        or hash_bytes(args.candidate.read_bytes()) != commitment["candidate_file_hash"]
        or hash_bytes(args.baseline.read_bytes()) != commitment["baseline_file_hash"]
        or hash_bytes(args.g1_usd.read_bytes()) != commitment["g1_asset_hash"]
        or hash_bytes(args.scene.read_bytes()) != commitment["scene_hash"]
        or _sealed(args.baseline_pilot)["report_hash"] != commitment["baseline_pilot_hash"]
        or _sealed(args.bank)["report_hash"] != commitment["bank_hash"]
        or subprocess.check_output(
            ["git", "status", "--porcelain", "--untracked-files=no"], cwd=source, text=True
        ).strip()
    ):
        raise ValueError("continuation source or policy drift")
    if args.compressed_reports and (
        hash_bytes(args.compressed_transport_review.read_bytes())
        != commitment["transport_file_hash"]
        or hash_json(load_json_artifact(root / "transport-preflight.json"))
        != commitment["transport_preflight_hash"]
    ):
        raise ValueError("actual transport proof changed during continuation")
    summary = _sealed(bank_root / "validation_summary.json")
    review = _sealed(bank_root / "independent_review.json")
    result = dict(
        schema="soccer.rsi.memory_validation_continuation_result.v1",
        commitment_hash=hash_json(commitment),
        bank_summary_hash=summary["report_hash"],
        bank_review_hash=review["report_hash"],
        consumed_gain=review["candidate_high_quality"] - review["warm_high_quality"],
        current_parent_retained=review["safe_pelvis"] is True
        and all(
            type(review[k]) is int and review[k] == 0
            for k in ("old_high_quality_loss", "old_clean_foot_loss", "new_out_of_play")
        ),
        qualification="COMPLETE_CONSUMED_COMPARISON_ONLY_NOT_RSI_M0_OR_FRESH",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(root / "result.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
