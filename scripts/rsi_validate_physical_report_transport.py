"""Three actual SIM executions verify plain/gzip report transport equivalence.

One fixed consumed course, same runner/model/parent. Complete reports and all
physical traces must agree. This is storage engineering, not a learning gain.
"""

import argparse
import re
import shutil
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.consolidated_smooth_motor import validate_model
from rosclaw_soccer.rsi.failure_curriculum_evidence import _outcome
from rosclaw_soccer.rsi.json_artifact_io import load_json_artifact
from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_train_protected_online_motor_v308 import _head

MODEL_HASH = "sha256:a5c3c21c3c0907d7c4be2da823d5aafaefa15844c5cf34d64718909af2017eb6"
COURSE = (20262103, 2)


def check_declared_model_hash(actual: str, declared: str) -> None:
    if not re.fullmatch(r"sha256:[a-f0-9]{64}", declared) or actual != declared:
        raise ValueError("exact preregistered transport model hash required")


def compare_complete_reports(
    plain: dict[str, Any],
    compressed: dict[str, Any],
    plain_outcome: dict[str, Any],
    compressed_outcome: dict[str, Any],
) -> None:
    if plain != compressed or hash_json(plain) != hash_json(compressed):
        raise ValueError("lossless transport changed the complete numerical report")
    if plain_outcome != compressed_outcome:
        raise ValueError("transport changed independently reconstructed physics")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "model",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "system-reserve-path",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--shared-model-report", action="store_true")
    parser.add_argument(
        "--expected-model-hash",
        default=MODEL_HASH,
        help="Explicit preregistered SIM model; defaults to the original fixed transport model",
    )
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    model = load_json_artifact(args.model)
    validate_model(model)
    check_declared_model_hash(model["model_hash"], args.expected_model_hash)
    expected_model_hash = args.expected_model_hash
    if args.output_root.exists():
        raise ValueError("new external transport directory required; no automatic retries")
    args.output_root.parent.mkdir(parents=True, exist_ok=True)
    system_reserve = 100 * 1024**3
    # Conservative plain-report/log working budget, before measured gzip sizes
    # exist. This is a three-execution transport test, not a full-bank budget.
    budget = args.model.stat().st_size * 10 + 512 * 1024**2
    same_volume = args.system_reserve_path.stat().st_dev == args.output_root.parent.stat().st_dev
    if shutil.disk_usage(args.system_reserve_path).free < system_reserve or shutil.disk_usage(
        args.output_root.parent
    ).free < budget + (system_reserve if same_volume else 1024**3):
        raise ValueError("transport capacity insufficient; preserve system and evidence reserves")
    paths = [
        runner,
        Path(__file__),
        args.model,
        args.g1_usd,
        args.late_swing_policy,
        source / "scripts/rsi_collect_approach_lateral_tracking_v286.py",
        source / "scripts/rsi_atomic_artifacts.py",
        source / "src/rosclaw_soccer/rsi/physical_report_io.py",
    ]
    if args.shared_model_report:
        from rosclaw.growth.shared_proof_payload import detach_payload

        paths.append(Path(detach_payload.__code__.co_filename))
    pins = {str(p): hash_bytes(p.read_bytes()) for p in paths}
    commitment = dict(
        schema="soccer.rsi.physical_report_transport_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=pins[str(runner)],
        asset_hash=pins[str(args.g1_usd)],
        input_source_hashes=pins,
        model_hash=expected_model_hash,
        course=list(COURSE),
        physical_executions_planned=3,
        system_reserve_bytes=system_reserve,
        working_budget_bytes=budget,
        partition="TRAIN_CONSUMED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir()
    (args.output_root / "logs").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    seed, lane = COURSE
    common = dict(
        root=args.output_root,
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        seed=seed,
        lane=lane,
        gpu=0,
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
        execution_timeout_s=600,
    )
    # A compressed parent also exercises the native causal parent reader.
    parent, _ = _run(**common, arm="reproduction", kind="parent", compressed_report=True)
    parent_path = resolve_physical_report(
        args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
    )
    reports, outcomes, folders = [], [], []
    for arm, compressed in (("plain", False), ("compressed", True)):
        raw, _ = _run(
            **common,
            arm=arm,
            kind="actor",
            motor_step=args.model,
            parent_report_override=parent_path,
            compressed_report=compressed,
            shared_model_report=compressed and args.shared_model_report,
        )
        folder = args.output_root / f"seed{seed}-lane{lane}-{arm}-actor"
        checked = _outcome(folder, raw["contact_motor_policy_hash"], commitment)
        if (
            checked["report"] != raw
            or raw["parent_report_hash"] != parent["report_hash"]
            or raw["contact_motor_policy"]["step_motor_proof"]["model"]["model_hash"]
            != expected_model_hash
        ):
            raise ValueError("transport lost exact parent/model bindings")
        reports.append(raw)
        outcomes.append(checked["outcome"])
        folders.append(folder)
    compare_complete_reports(*reports, *outcomes)
    if (
        any(hash_bytes(p.read_bytes()) != pins[str(p)] for p in paths)
        or _head(source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
        or load_json_artifact(args.output_root / "commitment.json") != commitment
    ):
        raise ValueError("source or input changed during actual transport comparison")
    result = dict(
        schema="soccer.rsi.physical_report_transport_review.v1",
        source_commitment_hash=hash_json(commitment),
        model_hash=expected_model_hash,
        complete_report_hash=reports[0]["report_hash"],
        complete_payload_equal=True,
        body_and_ball_trace_hashes_equal=True,
        actual_motor_actions_reconstructed=600,
        physical_executions_added=3,
        measured_outcome=outcomes[0],
        plain_report_bytes=resolve_physical_report(folders[0] / "report.json").stat().st_size,
        compressed_report_bytes=resolve_physical_report(folders[1] / "report.json").stat().st_size,
        plain_log_bytes=(args.output_root / "logs" / f"{folders[0].name}.log").stat().st_size,
        compressed_log_bytes=(args.output_root / "logs" / f"{folders[1].name}.log").stat().st_size,
        qualification="STORAGE_EQUIVALENCE_ONLY_NOT_LEARNING_GAIN",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    if args.shared_model_report:
        result.pop("report_hash")
        store = args.output_root / ".shared-models"
        result["physical_report_representation"] = "lossless_shared_model_gzip_json"
        result["complete_shared_payload_bytes"] = sum(p.stat().st_size for p in store.iterdir())
        result["shared_payload_count"] = len(list(store.iterdir()))
        result["report_hash"] = hash_json(result)
    write_once(args.output_root / "transport_review.json", result)
    print(result, flush=True)


if __name__ == "__main__":
    main()
