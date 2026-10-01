"""Compare updated PPO and frozen warm-start actors in actual consumed physics."""

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.compiled_step_inference import make_preview as compiled_preview
from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.step_motor_network import validate_model as validate_warm
from rosclaw_soccer.rsi.step_motor_ppo import validate_model
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
        "online-model",
        "pilot-summary",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    model = json.loads(args.online_model.read_text())
    warm = json.loads(args.step_model.read_text())
    online_base, warm_base = model, warm
    if model.get("schema") == "soccer.rsi.compiled_step_motor_decoder.v1":
        compiled_preview(model)
        online_base = model["base_model"]
    if warm.get("schema") == "soccer.rsi.compiled_step_motor_decoder.v1":
        compiled_preview(warm)
        warm_base = warm["base_model"]
    if online_base.get("schema") == "soccer.rsi.kernel_guarded_step_actor_critic.v1":
        from rosclaw_soccer.rsi.kernel_guarded_step_network import validate_model as kernel_validate

        kernel_validate(online_base)
        predecessor = online_base["encoder"]["base_model"]
    elif online_base.get("schema") == "soccer.rsi.protected_phase_step_actor_critic.v1":
        from rosclaw_soccer.rsi.protected_phase_step_network import (
            validate_model as protected_validate,
        )

        protected_validate(online_base)
        predecessor = online_base["base_model"]
    else:
        validate_model(online_base)
        predecessor = online_base["warm_start_model"]
    validate_warm(warm_base)
    pilot = _sealed(args.pilot_summary)
    if predecessor != warm_base or pilot["commitment"]["model_hash"] != warm_base["model_hash"]:
        parser.error("online and warm policies must share the same consumed pilot")
    commitment = dict(
        schema="soccer.rsi.online_step_validation_commitment.v1",
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        model_hash=model["model_hash"],
        warm_model_hash=warm["model_hash"],
        warm_base_model_hash=warm_base["model_hash"],
        online_base_model_hash=online_base["model_hash"],
        pilot_hash=pilot["report_hash"],
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
        parent_path = args.output_root / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
        baseline, baseline_outcome = _run(
            **common,
            arm="warm",
            kind="actor",
            motor_step=args.step_model,
            parent_report_override=parent_path,
        )
        old = _sealed(
            args.pilot_summary.parent / f"seed{seed}-lane{lane}-step-neural-actor/report.json"
        )
        if any(
            baseline[k] != old[k]
            for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
        ):
            raise ValueError("online dispatcher changed frozen warm-start physical actor")
        candidate, outcome = _run(
            **common,
            arm="online",
            kind="actor",
            motor_step=args.online_model,
            parent_report_override=parent_path,
        )
        print(
            f"ONLINE_STEP_EXECUTED seed={seed} lane={lane} HQ={high_quality(outcome)}", flush=True
        )
        return dict(
            seed=seed,
            lane=lane,
            parent_report_hash=parent["report_hash"],
            baseline=dict(
                report_hash=baseline["report_hash"],
                high_quality=high_quality(baseline_outcome),
                **baseline_outcome,
            ),
            online=dict(
                report_hash=candidate["report_hash"], high_quality=high_quality(outcome), **outcome
            ),
        )

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = list(pool.map(worker, range(4)))
    if _head(source) != commitment["source_commit"]:
        raise ValueError("online motor source drift")
    summary = dict(
        schema="soccer.rsi.online_step_motor_physical_validation.v1",
        commitment=commitment,
        rows=rows,
        physical_executions=12,
        independent_contexts=4,
        baseline_high_quality=sum(r["baseline"]["high_quality"] for r in rows),
        online_high_quality=sum(r["online"]["high_quality"] for r in rows),
        old_high_quality_loss=sum(
            r["baseline"]["high_quality"] and not r["online"]["high_quality"] for r in rows
        ),
        old_clean_foot_loss=sum(
            r["baseline"]["clean_foot_only"] and not r["online"]["clean_foot_only"] for r in rows
        ),
        safe_pelvis=all(r["online"]["minimum_pelvis_z_m"] >= 0.65 for r in rows),
        new_out_of_play=sum(
            r["baseline"]["maximum_lateral_excursion_m"]
            <= 4
            < r["online"]["maximum_lateral_excursion_m"]
            for r in rows
        ),
        qualification="CONSUMED_PILOT_ONLY_NOT_FRESH_NOT_PROMOTION",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "validation_summary.json", summary)
    print(
        json.dumps({k: v for k, v in summary.items() if k not in ("rows", "commitment")}),
        flush=True,
    )


if __name__ == "__main__":
    main()
