"""Independent spawned sampling worker; no tensors or policies are pickled."""

from argparse import Namespace
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.physical_report_io import resolve_physical_report
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality


def run_failure_worker(job: dict[str, Any]) -> list[dict[str, Any]]:
    from scripts.rsi_collect_failed_step_courses import sampling_model_path

    args = Namespace(**job["args"])
    gpu, courses, arm, runner = job["gpu"], job["courses"], job["arm"], job["runner"]
    views = job["view_hashes"]
    if type(gpu) is not int or not 0 <= gpu < 4 or arm not in ("warm", "candidate"):
        raise ValueError("bounded declared simulation worker required")
    records = []
    for i in range(gpu, len(courses), 4):
        old = courses[i]
        seed, lane = old["seed"], old["lane"]
        common = dict(
            root=args.output_root,
            runner=runner,
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            actor=args.late_swing_policy,
            seed=seed,
            lane=lane,
            gpu=gpu,
            gain=1.2,
            negative_only=True,
            core_root=args.core_root,
            resume=args.resume,
            execution_timeout_s=600,
        )
        if getattr(args, "shared_sampling_models", False):
            common.update(compressed_report=True, shared_model_report=True)
        for pending_arm, kind in [("reproduction", "parent"), ("greedy", "actor")] + [
            (f"sample-{s}", "actor") for s in range(args.samples_per_course)
        ]:
            stem = f"seed{seed}-lane{lane}-{pending_arm}-{kind}"
            if (args.output_root / "logs" / f"{stem}.log").exists():
                folder = args.output_root / stem
                if not any((folder / n).is_file() for n in ("report.json", "report.json.gz")):
                    raise ValueError(
                        "preserve and explicitly archive failed attempt before recovery"
                    )
                resolve_physical_report(folder / "report.json")
        parent, _ = _run(**common, arm="reproduction", kind="parent")
        parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        if getattr(args, "shared_sampling_models", False):
            parent_path = resolve_physical_report(parent_path)
        greedy, outcome = _run(
            **common,
            arm="greedy",
            kind="actor",
            motor_step=args.warm_model,
            parent_report_override=parent_path,
        )
        old_folder = args.bank_physics_root / f"seed{seed}-lane{lane}-{arm}-actor"
        old_report = _sealed(old_folder / "report.json")
        old_parent = _sealed(
            args.bank_physics_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        )
        if old_report["report_hash"] != old[arm]["report_hash"] or any(
            current[k] != historical[k]
            for current, historical in ((parent, old_parent), (greedy, old_report))
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("expanded training changed frozen warm/parent physical traces")
        sampled = []
        for s in range(args.samples_per_course):
            j = i * args.samples_per_course + s
            raw, result = _run(
                **common,
                arm=f"sample-{s}",
                kind="actor",
                motor_step=sampling_model_path(
                    args.output_root, j, compressed=args.compressed_sampling_models
                ),
                parent_report_override=parent_path,
            )
            sampled.append(
                dict(
                    sample=s,
                    view_hash=views[j],
                    report_hash=raw["report_hash"],
                    high_quality=high_quality(result),
                    **result,
                )
            )
            print(f"FAILURE_EXPLORED course={i} sample={s} HQ={high_quality(result)}", flush=True)
        record = dict(
            index=i,
            seed=seed,
            lane=lane,
            parent_report_hash=parent["report_hash"],
            greedy=dict(
                report_hash=greedy["report_hash"], high_quality=high_quality(outcome), **outcome
            ),
            samples=sampled,
        )
        write_once(args.output_root / f"row-{i}.json", record)
        records.append(record)
    return records
