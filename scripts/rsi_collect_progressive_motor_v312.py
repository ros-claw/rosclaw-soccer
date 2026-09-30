"""Frozen broader-curriculum neural reproduction and physical evaluation.

No new fresh pool is opened. The learner has 52 consumed contexts, not a
continuous-team qualification. Interrupted work requires an explicit resume.
"""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

from rosclaw_soccer.rsi.failure_curriculum_evidence import _sealed
from rosclaw_soccer.rsi.progressive_motor_actor import validate_model
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once
from scripts.rsi_train_protected_online_motor_v308 import _head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "bank-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "core-root",
    ):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    bank = _sealed(args.bank_root / "learning_bank.json")
    initial_path = args.bank_root / "initial_model.json"
    replay_path = args.bank_root / "replay_model.json"
    initial = json.loads(initial_path.read_text())
    candidate = json.loads(replay_path.read_text())
    validate_model(initial)
    validate_model(candidate)
    if (
        initial["learning_report_hash"] != bank["report_hash"]
        or candidate["parent_model_hash"] != initial["model_hash"]
        or candidate["learning_report_hash"] != bank["report_hash"]
        or bank["partition"] != "TRAIN_CONSUMED"
    ):
        parser.error("sealed consumed replay and its frozen models required")
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    commitment = dict(
        source_commit=_head(source),
        core_commit=_head(args.core_root),
        runner_hash=hash_bytes(runner.read_bytes()),
        asset_hash=hash_bytes(args.g1_usd.read_bytes()),
        learning_bank_hash=bank["report_hash"],
        initial_model_hash=initial["model_hash"],
        candidate_model_hash=candidate["model_hash"],
        partition="TRAIN_CONSUMED",
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)

    def execute(
        row: dict[str, Any], gpu: int, arm: str, path: Path
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        raw, outcome = _run(
            root=args.output_root,
            runner=runner,
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            actor=args.late_swing_policy,
            seed=row["seed"],
            lane=row["lane"],
            gpu=gpu,
            arm=arm,
            kind="actor",
            gain=1.2,
            negative_only=True,
            core_root=args.core_root,
            motor_online=path,
            parent_report_override=Path(row["parent_report"]),
            resume=args.resume,
        )
        outcome["high_quality"] = high_quality(outcome)
        proof = raw["contact_motor_policy"]["progressive_motor_proof"]
        if proof["context"] != row["observation"]:
            raise ValueError("broader curriculum changed causal context before its decision")
        return raw, outcome

    def reproduce(gpu: int) -> list[dict[str, Any]]:
        results = []
        for i in range(gpu, len(bank["courses"]), 4):
            row = bank["courses"][i]
            raw, outcome = execute(row, gpu, "progressive-reproduction", initial_path)
            old = _sealed(Path(row["predecessor_folder"]) / "report.json")
            if any(
                raw[k] != old[k]
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError("expanded zero-head model changed predecessor physics")
            results.append(dict(index=i, report_hash=raw["report_hash"], outcome=outcome))
            print(f"PROGRESSIVE_REPRODUCED i={i}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=4) as pool:
        controls = sorted(
            [r for batch in pool.map(reproduce, range(4)) for r in batch], key=lambda r: r["index"]
        )
    write_once(
        args.output_root / "reproduction.json",
        dict(rows=controls, matched_course_count=len(controls)),
    )

    def evaluate(gpu: int) -> list[dict[str, Any]]:
        results = []
        for i in range(gpu, len(bank["courses"]), 4):
            row = bank["courses"][i]
            raw, outcome = execute(row, gpu, "progressive-replay", replay_path)
            if row["predecessor_outcome"]["high_quality"]:
                old = _sealed(Path(row["predecessor_folder"]) / "report.json")
                if any(raw[k] != old[k] for k in ("body_trace_hash", "trace_hash")):
                    raise ValueError("learned update changed protected successful physics")
            results.append(
                dict(
                    index=i,
                    seed=row["seed"],
                    lane=row["lane"],
                    report_hash=raw["report_hash"],
                    **outcome,
                )
            )
            print(f"PROGRESSIVE_EVALUATED i={i} high_quality={outcome['high_quality']}", flush=True)
        return results

    with ThreadPoolExecutor(max_workers=4) as pool:
        rows = sorted(
            [r for batch in pool.map(evaluate, range(4)) for r in batch], key=lambda r: r["index"]
        )
    if (
        _head(source) != commitment["source_commit"]
        or _head(args.core_root) != commitment["core_commit"]
    ):
        raise ValueError("training source drift")
    result = dict(
        schema="soccer.rsi.progressive_replay_physics.v312",
        commitment=commitment,
        rows=rows,
        physical_episode_count=104,
        distinct_consumed_contexts=52,
        high_quality_count=sum(r["high_quality"] for r in rows),
        protected_quality_loss=sum(
            old["predecessor_outcome"]["high_quality"] and not r["high_quality"]
            for old, r in zip(bank["courses"], rows, strict=True)
        ),
        out_of_play_count=sum(r["maximum_lateral_excursion_m"] > 4 for r in rows),
        safe_pelvis_guardrail=all(r["minimum_pelvis_z_m"] >= 0.65 for r in rows),
        fresh_evaluation_status="NOT_OPENED",
        promotion_authorized=False,
        hardware_authorized=False,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "preview_summary.json", result)
    print(f"PROGRESSIVE_PREVIEW_COMPLETE HQ={result['high_quality_count']}/52", flush=True)


if __name__ == "__main__":
    main()
