"""Frozen stable-weight actor recalibration then independent physical replay."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed, retention_score
from rosclaw_soccer.rsi.stable_motor_bank import stable_replay
from rosclaw_soccer.rsi.stable_motor_update import stable_update, validate_stable_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
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
        "learning-bank",
        "preview-root",
        "online-root",
        "validation-root",
        "core-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--core-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if (
        _head(source) != args.source_commit
        or _head(args.core_root) != args.core_commit
        or (args.output_root.exists() and not args.resume)
    ):
        parser.error("frozen clean sources and fresh output or audited resume required")
    replay = stable_replay(
        args.online_root, args.learning_bank, args.preview_root, args.validation_root
    )
    summary = _sealed(args.online_root / "training_summary.json")
    best = [g for g in summary["generations"] if g["model_hash"] == summary["best_model_hash"]]
    parent_path = (
        args.online_root / f"models/g{best[-1]['generation']}.json"
        if best
        else args.online_root / "models/initial.json"
    )
    parent = json.loads(parent_path.read_text())
    trained = stable_update(parent, replay["samples"], replay["report_hash"])
    validate_stable_model(trained)
    protocol = json.loads((source / "docs/rsi/protocols/stable-motor-replay-v309.json").read_text())
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    commitment = dict(
        source_commit=args.source_commit,
        core_commit=args.core_commit,
        runner_hash=hash_bytes(runner.read_bytes()),
        collector_hash=hash_bytes(Path(__file__).read_bytes()),
        protocol_hash=hash_json(protocol),
        replay_hash=replay["report_hash"],
        prior_best_model_hash=parent["model_hash"],
        candidate_model_hash=trained["model_hash"],
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        activation_ceiling="SIM_ONLY",
        partition="TRAIN_CONSUMED",
    )
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)
    write_once(args.output_root / "learning_replay.json", replay)
    trained_path = args.output_root / "stable_model.json"
    write_once(trained_path, trained)

    def execute(
        index: int, gpu: int, arm: str, kind: str, model: Path | None = None
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seed, lane = COURSES[index]
        report, outcome = _run(
            root=args.output_root,
            runner=runner,
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            actor=args.late_swing_policy,
            seed=seed,
            lane=lane,
            gpu=gpu,
            arm=arm,
            gain=1.2,
            kind=kind,
            negative_only=True,
            motor_online=model,
            core_root=args.core_root,
            resume=args.resume,
            parent_report_override=args.output_root
            / f"seed{seed}-lane{lane}-reproduction-parent/report.json",
        )
        if kind == "actor":
            outcome["high_quality"] = high_quality(outcome)
        return report, outcome

    def reproduce(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for i in range(gpu, 12, 4):
            execute(i, gpu, "reproduction", "parent")
            report, outcome = execute(i, gpu, "champion", "actor", parent_path)
            seed, lane = COURSES[i]
            historical_arm = f"g{best[-1]['generation']}-neural" if best else "neural-reproduction"
            old = _sealed(
                args.online_root / f"seed{seed}-lane{lane}-{historical_arm}-actor/report.json"
            )
            if any(
                report[k] != old[k]
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError("parent physics changed; stable neural preview prohibited")
            rows.append(dict(course_index=i, report_hash=report["report_hash"], outcome=outcome))
            print(f"STABLE_CHAMPION_REPRODUCED course={i}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        reproduced = sorted(
            [r for batch in pool.map(reproduce, range(4)) for r in batch],
            key=lambda r: r["course_index"],
        )
    if len(reproduced) != 12:
        raise ValueError("all parent courses must reproduce before stable actor evaluation")
    write_once(
        args.output_root / "reproduction.json", dict(rows=reproduced, matched_course_count=12)
    )

    def evaluate(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for i in range(gpu, 12, 4):
            report, outcome = execute(i, gpu, "stable", "actor", trained_path)
            rows.append(dict(course_index=i, **outcome, report_hash=report["report_hash"]))
            print(f"STABLE_NEURAL_EVALUATED course={i} high={outcome['high_quality']}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(
            [r for batch in pool.map(evaluate, range(4)) for r in batch],
            key=lambda r: r["course_index"],
        )
    prior = _sealed(args.validation_root / "validation_summary.json")
    by_course = {(r["seed"], r["lane"]): r for r in prior["rows"]}
    score = retention_score(rows, [by_course[c] for c in COURSES])
    neural_prior = _sealed(args.preview_root / "preview_summary.json")
    loss = sum(
        neural_prior["rows"][i]["neural"]["high_quality"] and not rows[i]["high_quality"]
        for i in range(12)
    )
    if _head(source) != args.source_commit or _head(args.core_root) != args.core_commit:
        raise ValueError("stable experiment source drift")
    result = dict(
        schema="soccer.rsi.stable_motor_consumed_preview.v309",
        commitment=commitment,
        rows=rows,
        score=score,
        previous_nine_high_quality_loss=loss,
        consumed_gate_passed=score["consumed_training_gate_passed"] and loss == 0,
        physical_episode_count=36,
        learning_kind="offline_replay_recalibration",
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "preview_summary.json", result)
    print(
        f"STABLE_PREVIEW_COMPLETE high={score['high_quality_count']} "
        f"gate={result['consumed_gate_passed']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
