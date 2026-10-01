"""Actual bounded per-frame exploration on four already consumed motor courses."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.stochastic_step_execution import make_preview, make_sampling_view
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_collect_step_motor_pilot import COURSES
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
        "step-model",
        "pilot-summary",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--samples-per-course", type=int, default=8)
    parser.add_argument("--std", type=float, default=0.1)
    args = parser.parse_args()
    if not 1 <= args.samples_per_course <= 32:
        parser.error("bounded training budget required")
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    model = json.loads(args.step_model.read_text())
    pilot = _sealed(args.pilot_summary)
    rows = {(r["seed"], r["lane"]): r for r in pilot["rows"]}
    if set(rows) != set(COURSES) or pilot["commitment"]["model_hash"] != model["model_hash"]:
        parser.error("only previously consumed matching pilot permitted")
    views = []
    for course in range(4):
        for sample in range(args.samples_per_course):
            view = make_sampling_view(model, seed=202610016 + course * 100 + sample, std=args.std)
            make_preview(view)
            views.append(view)
    commitment = dict(
        schema="soccer.rsi.stochastic_step_training_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        base_model_hash=model["model_hash"],
        pilot_hash=pilot["report_hash"],
        courses=[list(c) for c in COURSES],
        samples_per_course=args.samples_per_course,
        std_raw=args.std,
        sampling_view_hashes=[v["model_hash"] for v in views],
        partition="TRAIN_CONSUMED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    (args.output_root / "models").mkdir()
    write_once(args.output_root / "commitment.json", commitment)
    for index, view in enumerate(views):
        write_once(args.output_root / "models" / f"sample-{index}.json", view)

    def worker(gpu: int) -> dict[str, Any]:
        seed, lane = COURSES[gpu]
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
        )
        parent, _ = _run(**common, arm="reproduction", kind="parent")
        parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        greedy, greedy_outcome = _run(
            **common,
            arm="greedy",
            kind="actor",
            motor_step=args.step_model,
            parent_report_override=parent_path,
        )
        old_path = (
            args.pilot_summary.parent / f"seed{seed}-lane{lane}-step-neural-actor/report.json"
        )
        old = _sealed(old_path)
        if any(
            greedy[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("stochastic dispatcher changed deterministic physical policy")
        sampled = []
        for sample in range(args.samples_per_course):
            index = gpu * args.samples_per_course + sample
            report, outcome = _run(
                **common,
                arm=f"sample-{sample}",
                kind="actor",
                motor_step=args.output_root / "models" / f"sample-{index}.json",
                parent_report_override=parent_path,
            )
            sampled.append(
                dict(
                    sample=sample,
                    view_hash=views[index]["model_hash"],
                    report_hash=report["report_hash"],
                    high_quality=high_quality(outcome),
                    **outcome,
                )
            )
            print(
                f"STOCHASTIC_EXECUTED seed={seed} lane={lane} sample={sample} "
                f"HQ={sampled[-1]['high_quality']}",
                flush=True,
            )
        return dict(
            seed=seed,
            lane=lane,
            parent_report_hash=parent["report_hash"],
            greedy=dict(
                report_hash=greedy["report_hash"],
                high_quality=high_quality(greedy_outcome),
                **greedy_outcome,
            ),
            samples=sampled,
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(worker, range(4)))
    if _head(source) != commitment["source_commit"]:
        raise ValueError("stochastic training source drift")
    summary = dict(
        schema="soccer.rsi.stochastic_step_physics.v1",
        commitment=commitment,
        rows=results,
        physical_executions=4 * (args.samples_per_course + 2),
        independent_contexts=4,
        exploration_executions=4 * args.samples_per_course,
        high_quality_samples=sum(s["high_quality"] for r in results for s in r["samples"]),
        qualification="TRAINING_ONLY_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "training_summary.json", summary)
    print(
        json.dumps(dict(report_hash=summary["report_hash"], HQ=summary["high_quality_samples"])),
        flush=True,
    )


if __name__ == "__main__":
    main()
