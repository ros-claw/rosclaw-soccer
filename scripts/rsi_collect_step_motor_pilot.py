"""Twelve actual executions for a four-course per-frame neural motor pilot."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.step_motor_network import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head

COURSES = ((20261177, 0), (20261282, 0), (20262102, 4), (20262104, 6))


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
        "champion-policy",
        "consumed-bank",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    model = json.loads(args.step_model.read_text())
    validate_model(model)
    bank = _sealed(args.consumed_bank)
    courses = {(r["seed"], r["lane"]): r for r in bank["courses"]}
    if not set(COURSES) <= courses.keys():
        parser.error("pilot cannot open any fresh context")
    commitment = dict(
        schema="soccer.rsi.step_motor_pilot_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        model_hash=model["model_hash"],
        consumed_bank_hash=bank["report_hash"],
        courses=[list(c) for c in COURSES],
        partition="CONSUMED_PILOT",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(parents=True, exist_ok=False)
    (args.output_root / "logs").mkdir()
    write_once(args.output_root / "commitment.json", commitment)

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
        old = _sealed(Path(courses[(seed, lane)]["parent_report"]))
        if any(
            parent[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("per-frame bridge changed physical parent")
        parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        baseline, baseline_outcome = _run(
            **common,
            arm="champion",
            kind="actor",
            motor_policy=args.champion_policy,
            parent_report_override=parent_path,
        )
        candidate, outcome = _run(
            **common,
            arm="step-neural",
            kind="actor",
            motor_step=args.step_model,
            parent_report_override=parent_path,
        )
        baseline_outcome["high_quality"] = high_quality(baseline_outcome)
        outcome["high_quality"] = high_quality(outcome)
        print(
            f"STEP_PILOT_EXECUTED seed={seed} lane={lane} "
            f"parent={baseline_outcome['high_quality']} neural={outcome['high_quality']}",
            flush=True,
        )
        return dict(
            seed=seed,
            lane=lane,
            parent_report_hash=parent["report_hash"],
            baseline=dict(report_hash=baseline["report_hash"], **baseline_outcome),
            neural=dict(report_hash=candidate["report_hash"], **outcome),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(worker, range(4)))
    if _head(source) != commitment["source_commit"]:
        raise ValueError("per-frame source drift")
    result = dict(
        schema="soccer.rsi.step_motor_physical_pilot.v1",
        commitment=commitment,
        rows=rows,
        physical_executions=12,
        independent_contexts=4,
        baseline_high_quality=sum(r["baseline"]["high_quality"] for r in rows),
        neural_high_quality=sum(r["neural"]["high_quality"] for r in rows),
        qualification="PILOT_ONLY_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "pilot_summary.json", result)
    print(
        json.dumps(
            dict(
                report_hash=result["report_hash"], neural_high_quality=result["neural_high_quality"]
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
