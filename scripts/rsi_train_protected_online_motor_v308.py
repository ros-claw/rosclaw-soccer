"""Actual-physics episodic contextual actor-critic, with generic Core protection.

This is not full-trajectory PPO or torque control. No course-ID policy routing,
fresh-test access, or automatic runtime/promotion authority. Resume is audited.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, cast

import numpy as np

from rosclaw_soccer.rsi.contact_motor_phase import make_policy
from rosclaw_soccer.rsi.failure_curriculum_evidence import (
    _sealed,
    retention_score,
    review_curriculum,
)
from rosclaw_soccer.rsi.motor_bootstrap_network import validate_model as validate_base
from rosclaw_soccer.rsi.motor_learning_bank import causal_context
from rosclaw_soccer.rsi.online_motor_actor_critic import (
    actor_parameters,
    make_model,
    terminal_return,
    update_from_physics,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once


def _head(root: Path) -> str:
    if subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip():
        raise ValueError("immutable clean source required")
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()


def _normalize(parameters: Any) -> list[float]:
    values = np.asarray(parameters, dtype=np.float64)
    return cast(
        list[float], np.concatenate((values[:36] / 0.16, [(values[36] + 0.05) / 0.3])).tolist()
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "learning-bank",
        "neural-model",
        "preview-root",
        "training-root",
        "validation-root",
        "core-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--core-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    source = Path(__file__).resolve().parent.parent
    if _head(source) != args.source_commit or _head(args.core_root) != args.core_commit:
        parser.error("frozen Soccer and Core commits required")
    if args.output_root.exists() and not args.resume:
        parser.error("fresh output or explicit audited resume required")
    protocol = json.loads(
        (source / "docs/rsi/protocols/protected-online-motor-v308.json").read_text()
    )
    preview = _sealed(args.preview_root / "preview_summary.json")
    bank = _sealed(args.learning_bank)
    reference_report = _sealed(args.validation_root / "validation_summary.json")
    reviewed = review_curriculum(args.training_root, args.validation_root)
    base = json.loads(args.neural_model.read_text())
    validate_base(base)
    if (
        preview["commitment"]["neural_model_hash"] != base["model_hash"]
        or base["bank_hash"] != bank["report_hash"]
        or bank["independent_review_hash"] != reviewed["report_hash"]
        or not reviewed["training_complete"]
        or [r["course_index"] for r in preview["rows"] if not r["neural"]["high_quality"]]
        != protocol["failed_course_indices"]
    ):
        raise ValueError(
            "independently reviewed predecessor and declared failure curriculum required"
        )
    contexts: dict[tuple[int, int], list[float]] = {}
    for sample in bank["critic_samples"]:
        course = tuple(sample["audit_metadata"]["course"])
        if course in contexts and contexts[course] != sample["observation"]:
            raise ValueError("inconsistent measured replay context")
        contexts[course] = sample["observation"]
    protected = [contexts[COURSES[i]] for i in protocol["protected_course_indices"]]
    initial = make_model(base, protected, preview["report_hash"])
    runner = source / "scripts/rsi_isaac_vector_first_touch.py"
    commitment = {
        "source_commit": args.source_commit,
        "core_commit": args.core_commit,
        "runner_hash": hash_bytes(runner.read_bytes()),
        "trainer_hash": hash_bytes(Path(__file__).read_bytes()),
        "protocol_hash": hash_json(protocol),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "bank_hash": bank["report_hash"],
        "predecessor_preview_hash": preview["report_hash"],
        "initial_model_hash": initial["model_hash"],
        "activation_ceiling": "SIM_ONLY",
        "partition": "TRAIN_CONSUMED",
    }
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    for name in ("logs", "models", "policies"):
        (args.output_root / name).mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)
    write_once(args.output_root / "models/initial.json", initial)

    def execute(
        index: int, gpu: int, arm: str, **backend: Any
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        seed, lane = COURSES[index]
        kind = "parent" if arm == "reproduction-parent" else "actor"
        arm = "reproduction" if kind == "parent" else arm
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
            parent_report_override=args.output_root
            / f"seed{seed}-lane{lane}-reproduction-parent/report.json",
            resume=args.resume,
            core_root=args.core_root,
            **backend,
        )
        if kind == "parent":
            return report, outcome
        outcome["high_quality"] = high_quality(outcome)
        folder = args.output_root / f"seed{seed}-lane{lane}-{arm}-actor"
        if backend.get("motor_bootstrap") or backend.get("motor_online"):
            key = "online_motor_proof" if backend.get("motor_online") else "bootstrap_proof"
            context = report["contact_motor_policy"][key]["context"]
        else:
            with (
                np.load(folder / "body_trace.npz", allow_pickle=False) as body,
                np.load(folder / "contact_motor_trace.npz", allow_pickle=False) as motor,
            ):
                context = list(causal_context(body, motor, outcome["first_contact_frame"]))
        if context != contexts[COURSES[index]]:
            raise ValueError("physical proposal changed its causal context before decision")
        terminal_return(outcome)
        return report, outcome

    def reproduce(gpu: int) -> list[dict[str, Any]]:
        rows = []
        for i in range(gpu, 12, 4):
            execute(i, gpu, "reproduction-parent")
            report, outcome = execute(
                i, gpu, "neural-reproduction", motor_bootstrap=args.neural_model
            )
            seed, lane = COURSES[i]
            old = _sealed(args.preview_root / f"seed{seed}-lane{lane}-neural-actor/report.json")
            if any(
                report[k] != old[k]
                for k in ("body_trace_hash", "trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError("neural predecessor physics drift; exploration prohibited")
            rows.append(
                {"course_index": i, "report_hash": report["report_hash"], "outcome": outcome}
            )
            print(f"ONLINE_PREDECESSOR_REPRODUCED course={i}", flush=True)
        return rows

    with ThreadPoolExecutor(max_workers=4) as pool:
        reproduced = sorted(
            [r for batch in pool.map(reproduce, range(4)) for r in batch],
            key=lambda r: r["course_index"],
        )
    write_once(
        args.output_root / "reproduction.json",
        {"rows": reproduced, "matched_course_count": len(reproduced)},
    )
    if len(reproduced) != 12:
        raise ValueError("all predecessor contexts must reproduce before learning")
    replay = [
        dict(
            observation=s["observation"],
            normalized_action=_normalize(s["motor_parameters"]),
            outcome=s["learning_labels"],
            report_hash=s["audit_metadata"]["source_report_hash"],
        )
        for s in bank["critic_samples"]
    ]
    reference_by_course = {(r["seed"], r["lane"]): r for r in reference_report["rows"]}
    reference = [reference_by_course[c] for c in COURSES]
    best_model = initial
    initial_score = retention_score([r["neural"] for r in preview["rows"]], reference)
    best_rank: tuple[float, ...] = (1.0, *initial_score["rank"])
    replay.extend(
        dict(
            observation=contexts[COURSES[r["course_index"]]],
            normalized_action=_normalize(
                actor_parameters(initial, contexts[COURSES[r["course_index"]]])
            ),
            outcome=r["outcome"],
            report_hash=r["report_hash"],
        )
        for r in reproduced
    )
    generations = []
    for generation in range(protocol["generations"]):
        proposals = []
        rng = np.random.default_rng(protocol["sampling_seed"] + generation)
        for index in protocol["failed_course_indices"]:
            mean = actor_parameters(best_model, contexts[COURSES[index]])
            for candidate in range(protocol["proposals_per_failed_course_per_generation"]):
                noise = np.concatenate(
                    (
                        rng.normal(0, protocol["joint_exploration_std_rad"], 36),
                        rng.normal(0, protocol["duration_exploration_std_m"], 1),
                    )
                )
                parameters = mean if candidate == 0 else mean + noise
                parameters = np.clip(parameters, [-0.16] * 36 + [-0.35], [0.16] * 36 + [0.25])
                policy = make_policy(
                    parameters[:36].reshape(3, 12), float(parameters[36]), best_model["model_hash"]
                )
                path = args.output_root / f"policies/g{generation}-i{index}-c{candidate}.json"
                write_once(path, policy)
                proposals.append((index, candidate, parameters, path))

        def explore(
            gpu: int,
            proposals: list[tuple[int, int, Any, Path]] = proposals,
            generation: int = generation,
        ) -> list[dict[str, Any]]:
            collected = []
            for n in range(gpu, len(proposals), 4):
                index, candidate, parameters, path = proposals[n]
                report, outcome = execute(
                    index, gpu, f"g{generation}-c{candidate}-explore", motor_policy=path
                )
                collected.append(
                    dict(
                        observation=contexts[COURSES[index]],
                        normalized_action=_normalize(parameters),
                        outcome=outcome,
                        report_hash=report["report_hash"],
                        course_index=index,
                        candidate=candidate,
                    )
                )
                print(
                    f"ONLINE_PHYSICS_SAMPLE g={generation} course={index} c={candidate} "
                    f"high={outcome['high_quality']}",
                    flush=True,
                )
            return collected

        with ThreadPoolExecutor(max_workers=4) as pool:
            samples = sorted(
                [r for batch in pool.map(explore, range(4)) for r in batch],
                key=lambda r: (r["course_index"], r["candidate"]),
            )
        feedback = {
            "generation": generation,
            "parent_model_hash": best_model["model_hash"],
            "samples": samples,
            "runtime_selection_authorized": False,
        }
        feedback["report_hash"] = hash_json(feedback)
        write_once(args.output_root / f"feedback-g{generation}.json", feedback)
        replay.extend(samples)
        trained = update_from_physics(best_model, replay, feedback["report_hash"])
        model_path = args.output_root / f"models/g{generation}.json"
        write_once(model_path, trained)

        def evaluate(
            gpu: int, generation: int = generation, model_path: Path = model_path
        ) -> list[dict[str, Any]]:
            rows = []
            for i in range(gpu, 12, 4):
                report, outcome = execute(i, gpu, f"g{generation}-neural", motor_online=model_path)
                rows.append({"course_index": i, **outcome, "report_hash": report["report_hash"]})
                print(
                    f"ONLINE_NEURAL_EVALUATED g={generation} course={i} "
                    f"high={outcome['high_quality']}",
                    flush=True,
                )
            return rows

        with ThreadPoolExecutor(max_workers=4) as pool:
            rows = sorted(
                [r for batch in pool.map(evaluate, range(4)) for r in batch],
                key=lambda r: r["course_index"],
            )
        score = retention_score(rows, reference)
        loss = sum(
            preview["rows"][i]["neural"]["high_quality"] and not rows[i]["high_quality"]
            for i in range(12)
        )
        current_rank = (float(loss == 0), *score["rank"])
        ledger = dict(
            generation=generation,
            model_hash=trained["model_hash"],
            model_path=str(model_path),
            rows=rows,
            score=score,
            previous_nine_high_quality_loss=loss,
            consumed_gate_passed=score["consumed_training_gate_passed"] and loss == 0,
            promotion_authorized=False,
        )
        ledger["report_hash"] = hash_json(ledger)
        write_once(args.output_root / f"generation-{generation}.json", ledger)
        generations.append(ledger)
        if current_rank > best_rank:
            best_model, best_rank = trained, current_rank
        print(
            f"ONLINE_GENERATION_COMPLETE g={generation} high={score['high_quality_count']} "
            f"old_clean_loss={score['old_clean_foot_loss']} gate={ledger['consumed_gate_passed']}",
            flush=True,
        )
        if ledger["consumed_gate_passed"]:
            break
    if _head(source) != args.source_commit or _head(args.core_root) != args.core_commit:
        raise ValueError("experiment source drift")
    result = dict(
        schema="soccer.rsi.protected_online_motor_training.v308",
        commitment=commitment,
        generations=generations,
        best_model_hash=best_model["model_hash"],
        total_replay_records=len(replay),
        activation_ceiling="SIM_ONLY",
        promotion_authorized=False,
        fresh_holdout_open_authorized=False,
        training_complete=True,
    )
    result["report_hash"] = hash_json(result)
    write_once(args.output_root / "training_summary.json", result)


if __name__ == "__main__":
    main()
