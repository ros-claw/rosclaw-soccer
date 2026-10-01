"""Recover only absent fixed exploration jobs after a recorded pre-physics crash."""

import argparse
import importlib.util
import json
import time
from pathlib import Path

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "root",
        "recovery-root",
        "original-source",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "step-model",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    commitment = json.loads((args.root / "commitment.json").read_text())
    runner = args.original_source / "scripts/rsi_isaac_vector_first_touch.py"
    helper = args.original_source / "scripts/rsi_collect_approach_lateral_tracking_v286.py"
    if (
        _head(args.original_source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
        or commitment["courses"] != [list(c) for c in COURSES]
        or commitment["samples_per_course"] != 8
        or commitment["partition"] != "TRAIN_CONSUMED"
        or (args.root / "training_summary.json").exists()
    ):
        parser.error("exact incomplete frozen v316 commitment required")
    spec = importlib.util.spec_from_file_location("frozen_stochastic_helper", helper)
    if spec is None or spec.loader is None:
        raise ValueError("frozen helper cannot be loaded")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    common = dict(
        runner=runner,
        isaac_python=args.isaac_python,
        g1_usd=args.g1_usd,
        model_root=args.model_root,
        actor=args.late_swing_policy,
        gain=1.2,
        negative_only=True,
        core_root=args.core_root,
    )
    # Fixed missing set is based on the failed lane worker, never its outcomes.
    seed, lane = COURSES[1]
    missing = (5, 6, 7)
    failed_log = args.root / "logs" / f"seed{seed}-lane{lane}-sample-5-actor.log"
    message = failed_log.read_text()
    if (
        "XOpenDisplay" not in message
        or "[Fatal]" not in message
        or "RSI_ISAAC_VECTOR_FIRST_TOUCH=" in message
    ):
        raise ValueError("recovery requires the known pre-physics native startup crash")
    for sample in missing:
        folder = args.root / f"seed{seed}-lane{lane}-sample-{sample}-actor"
        if folder.exists():
            raise ValueError("never overwrite an existing physical or partial execution")
    args.recovery_root.mkdir(parents=True, exist_ok=False)
    (args.recovery_root / "logs").mkdir()
    receipt_commitment = dict(
        schema="soccer.rsi.stochastic_resource_recovery_commitment.v1",
        original_commitment_hash=hash_json(commitment),
        original_source_commit=_head(args.original_source),
        frozen_runner_hash=commitment["runner_hash"],
        frozen_helper_hash=hash_bytes(helper.read_bytes()),
        helper_source_hash=hash_bytes(Path(__file__).read_bytes()),
        missing_samples=list(missing),
        failed_log_hash=hash_bytes(failed_log.read_bytes()),
        course=[seed, lane],
        gpu=1,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    write_once(args.recovery_root / "commitment.json", receipt_commitment)
    parent_path = args.root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
    control, _ = module._run(
        **common,
        root=args.recovery_root,
        seed=seed,
        lane=lane,
        gpu=1,
        arm="control",
        kind="actor",
        motor_step=args.step_model,
        parent_report_override=parent_path,
    )
    old = _sealed(args.root / f"seed{seed}-lane{lane}-greedy-actor/report.json")
    if any(
        control[k] != old[k]
        for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
    ):
        raise ValueError("resource recovery control changed body or ball physics")
    archive = failed_log.with_name(failed_log.stem + "-native-startup-crash-attempt1.log")
    if archive.exists():
        raise ValueError("failed startup log archive already exists")
    failed_log.rename(archive)
    recovered = []
    for sample in missing:
        report, _ = module._run(
            **common,
            root=args.root,
            seed=seed,
            lane=lane,
            gpu=1,
            arm=f"sample-{sample}",
            kind="actor",
            motor_step=args.root / "models" / f"sample-{8 + sample}.json",
            parent_report_override=parent_path,
        )
        recovered.append(dict(sample=sample, report_hash=report["report_hash"]))
        print(f"STOCHASTIC_RECOVERED seed={seed} sample={sample}", flush=True)
    # Wait only for immutable report files from other fixed workers; no new jobs.
    deadline = time.monotonic() + 1800
    while not all(
        (args.root / f"seed{course_seed}-lane{course_lane}-sample-{i}-actor/report.json").is_file()
        for course_seed, course_lane in COURSES
        for i in range(8)
    ):
        if time.monotonic() >= deadline:
            raise RuntimeError("original fixed workers remain incomplete; preserve all evidence")
        print("WAITING_FOR_ORIGINAL_FIXED_REPORTS", flush=True)
        time.sleep(10)
    rows = []
    for course_index, (course_seed, course_lane) in enumerate(COURSES):
        parent, _ = module._run(
            **common,
            root=args.root,
            seed=course_seed,
            lane=course_lane,
            gpu=course_index,
            arm="reproduction",
            kind="parent",
            resume=True,
        )
        parent_path = (
            args.root / f"seed{course_seed}-lane{course_lane}-reproduction-parent/report.json"
        )
        greedy, outcome = module._run(
            **common,
            root=args.root,
            seed=course_seed,
            lane=course_lane,
            gpu=course_index,
            arm="greedy",
            kind="actor",
            motor_step=args.step_model,
            parent_report_override=parent_path,
            resume=True,
        )
        samples = []
        for sample in range(8):
            path = args.root / "models" / f"sample-{8 * course_index + sample}.json"
            view = json.loads(path.read_text())
            if view["model_hash"] != commitment["sampling_view_hashes"][8 * course_index + sample]:
                raise ValueError("fixed exploration model changed")
            raw, observed = module._run(
                **common,
                root=args.root,
                seed=course_seed,
                lane=course_lane,
                gpu=course_index,
                arm=f"sample-{sample}",
                kind="actor",
                motor_step=path,
                parent_report_override=parent_path,
                resume=True,
            )
            samples.append(
                dict(
                    sample=sample,
                    view_hash=view["model_hash"],
                    report_hash=raw["report_hash"],
                    high_quality=high_quality(observed),
                    **observed,
                )
            )
        rows.append(
            dict(
                seed=course_seed,
                lane=course_lane,
                parent_report_hash=parent["report_hash"],
                greedy=dict(
                    report_hash=greedy["report_hash"], high_quality=high_quality(outcome), **outcome
                ),
                samples=samples,
            )
        )
    receipt = dict(
        schema="soccer.rsi.stochastic_resource_recovery.v1",
        commitment=receipt_commitment,
        control_report_hash=control["report_hash"],
        control_body_and_ball_byte_equal=True,
        recovered=recovered,
        extra_control_physical_executions=1,
        archived_failed_log_hash=hash_bytes(archive.read_bytes()),
        archived_failed_log=str(archive.relative_to(args.root)),
        promotion_authorized=False,
        hardware_authorized=False,
    )
    receipt["report_hash"] = hash_json(receipt)
    write_once(args.recovery_root / "recovery_summary.json", receipt)
    summary = dict(
        schema="soccer.rsi.stochastic_step_physics.v1",
        commitment=commitment,
        rows=rows,
        scheduled_physical_executions=40,
        physical_executions=41,
        independent_contexts=4,
        exploration_executions=32,
        high_quality_samples=sum(s["high_quality"] for r in rows for s in r["samples"]),
        recovery_receipt_hash=receipt["report_hash"],
        extra_control_physical_executions=1,
        prephysics_native_startup_failures=1,
        qualification="TRAINING_ONLY_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary["report_hash"] = hash_json(summary)
    write_once(args.root / "training_summary.json", summary)
    print(
        json.dumps(dict(report_hash=summary["report_hash"], HQ=summary["high_quality_samples"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
