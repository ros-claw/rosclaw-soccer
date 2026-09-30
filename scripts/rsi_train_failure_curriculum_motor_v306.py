"""Train the next motor generation on actual failures AND protected successes.

SIM_ONLY: one plastic 37-parameter motor primitive; no policy activation.
Frozen-model reproduction must match every old body/ball trace before fitting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_phase import END_GAP_LIMITS_M, make_policy
from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, load_policy
from rosclaw_soccer.rsi.core_motor_growth import export_motor_growth
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_bilateral_motor_validation_v305 import COURSES, POLICY_HASH
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality
from scripts.rsi_train_bilateral_contact_motor_v303 import write_once

VALIDATION_HASH = "sha256:df875a1d41cd33e5883d6ec9c52417aed6a2d8ca95637b7a130e6d6d8c3b8abf"
POPULATION = 8
GENERATIONS = 4


def rank(rows: list[dict[str, Any]], reference: list[dict[str, Any]]) -> tuple[float, ...]:
    if len(rows) != 12 or len(reference) != 12:
        raise ValueError("complete matched 12-course curriculum required")
    pairs = list(zip(rows, reference, strict=True))
    return (
        float(all(r["minimum_pelvis_z_m"] >= 0.65 for r in rows)),
        -float(
            sum(b["arms"]["learned"]["high_quality"] and not r["high_quality"] for r, b in pairs)
        ),
        -float(
            sum(b["arms"]["gain_12"]["high_quality"] and not r["high_quality"] for r, b in pairs)
        ),
        -float(
            sum(
                b["arms"]["gain_08"]["clean_foot_only"] and not r["clean_foot_only"]
                for r, b in pairs
            )
        ),
        -float(
            sum(
                b["arms"]["gain_08"]["maximum_lateral_excursion_m"]
                <= 4
                < r["maximum_lateral_excursion_m"]
                for r, b in pairs
            )
        ),
        -float(
            sum(b["arms"]["gain_08"]["high_quality"] and not r["high_quality"] for r, b in pairs)
        ),
        float(sum(r["high_quality"] for r in rows)),
        float(sum(r["reward"] for r in rows)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "output-root",
        "isaac-python",
        "g1-usd",
        "model-root",
        "late-swing-policy",
        "initial-policy",
        "zero-parent-policy",
        "validation-root",
    ):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    runner = root / "scripts/rsi_isaac_vector_first_touch.py"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    if (
        head != args.source_commit
        or subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        or (args.output_root.exists() and not args.resume)
    ):
        parser.error("frozen clean source and fresh output or explicit resume required")
    prior = json.loads((args.validation_root / "validation_summary.json").read_text())
    if prior.get("report_hash") != VALIDATION_HASH or prior.get("report_hash") != hash_json(
        {k: v for k, v in prior.items() if k != "report_hash"}
    ):
        raise ValueError("unsealed predecessor failure curriculum")
    core = export_motor_growth(
        validation_root=args.validation_root,
        policy_path=args.initial_policy,
        protocol_path=root / "docs/rsi/protocols/bilateral-motor-validation-v305.json",
        zero_parent_policy_path=args.zero_parent_policy,
    )
    initial, knots = load_policy(args.initial_policy)
    if initial["policy_hash"] != POLICY_HASH:
        raise ValueError("unexpected frozen predecessor")
    protocol = json.loads(
        (root / "docs/rsi/protocols/failure-curriculum-motor-v306.json").read_text()
    )
    if (
        protocol["courses"] != [list(c) for c in COURSES]
        or protocol["generations"] != GENERATIONS
        or protocol["population"] != POPULATION
    ):
        raise ValueError("failure curriculum protocol drift")
    by_course = {(r["seed"], r["lane"]): r for r in prior["rows"]}
    reference = [by_course[c] for c in COURSES]
    commitment = {
        "source_commit": head,
        "runner_hash": hash_bytes(runner.read_bytes()),
        "trainer_hash": hash_bytes(Path(__file__).read_bytes()),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "initial_policy_hash": POLICY_HASH,
        "prior_validation_hash": VALIDATION_HASH,
        "protocol_hash": hash_json(protocol),
        "core_growth_report_hash": core["report_hash"],
        "activation_ceiling": "SIM_ONLY",
    }
    commitment_hash = hash_json(commitment)
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    (args.output_root / "policies").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)
    write_once(args.output_root / "core-predecessor-growth.json", core)
    write_once(
        args.output_root / "curriculum.json",
        {
            "failed_courses": [
                list(c) for c in COURSES if not by_course[c]["arms"]["learned"]["high_quality"]
            ],
            "protected_courses": [
                list(c) for c in COURSES if by_course[c]["arms"]["learned"]["high_quality"]
            ],
            "reference_hash": VALIDATION_HASH,
        },
    )

    def run(index: int, arm: str, kind: str, policy: Path | None = None):
        seed, lane = COURSES[index]
        return _run(
            root=args.output_root,
            runner=runner,
            isaac_python=args.isaac_python,
            g1_usd=args.g1_usd,
            model_root=args.model_root,
            actor=args.late_swing_policy,
            seed=seed,
            lane=lane,
            gpu=index % 4,
            arm=arm,
            gain=1.2,
            kind=kind,
            negative_only=True,
            motor_policy=policy,
            parent_report_override=args.output_root
            / f"seed{seed}-lane{lane}-reproduction-parent/report.json"
            if policy is not None
            else None,
            resume=args.resume,
        )

    def reproduce(gpu: int) -> list[dict[str, Any]]:
        result = []
        for index in range(gpu, 12, 4):
            run(index, "reproduction", "parent")
            report, outcome = run(index, "reproduction", "actor", args.initial_policy)
            seed, lane = COURSES[index]
            previous = json.loads(
                (
                    args.validation_root / f"seed{seed}-lane{lane}-learned-actor/report.json"
                ).read_text()
            )
            if any(
                report[k] != previous[k]
                for k in ("trace_hash", "body_trace_hash", "asset_hash", "sonic_qualification_hash")
            ):
                raise ValueError(
                    "predecessor body/ball reproduction diverged; training not authorized"
                )
            result.append(
                {
                    "course_index": index,
                    "report_hash": report["report_hash"],
                    "previous_report_hash": previous["report_hash"],
                    "outcome": {**outcome, "high_quality": high_quality(outcome)},
                }
            )
            print(f"MOTOR_CURRICULUM_REPRODUCED seed={seed} lane={lane}", flush=True)
        return result

    with ThreadPoolExecutor(max_workers=4) as pool:
        reproduction = [r for batch in pool.map(reproduce, range(4)) for r in batch]
    write_once(
        args.output_root / "reproduction.json",
        {
            "matched_course_count": len(reproduction),
            "rows": reproduction,
            "reference_source_commit": prior["source_commit"],
            "current_source_commit": head,
        },
    )
    center = np.concatenate((knots.ravel(), [0.25]))
    best_vector = center.copy()
    best = None
    all_candidates = []
    rng = np.random.default_rng(20261001306)
    for generation in range(GENERATIONS):
        scales = np.concatenate(
            (
                np.full(36, (0.012, 0.010, 0.008, 0.006)[generation]),
                [(0.05, 0.04, 0.03, 0.02)[generation]],
            )
        )
        noise = rng.normal(size=(3, 37)) * scales
        proposals = np.vstack((best_vector, center, center + noise, center - noise))
        proposals[:, :36] = np.clip(proposals[:, :36], -CAP_RAD, CAP_RAD)
        proposals[:, 36] = np.clip(proposals[:, 36], *END_GAP_LIMITS_M)
        for index, vector in enumerate(proposals):
            write_once(
                args.output_root / f"policies/g{generation}-c{index}.json",
                make_policy(vector[:36].reshape(3, 12), float(vector[36]), commitment_hash),
            )

        def worker(gpu: int, generation: int = generation) -> list[dict[str, Any]]:
            result = []
            for candidate in range(POPULATION):
                policy_path = args.output_root / f"policies/g{generation}-c{candidate}.json"
                for index in range(gpu, 12, 4):
                    report, outcome = run(
                        index, f"g{generation}-c{candidate}", "actor", policy_path
                    )
                    if (
                        report["source_hash"] != commitment["runner_hash"]
                        or report["asset_hash"] != commitment["asset_hash"]
                    ):
                        raise ValueError("training runner or asset drift")
                    if generation == 0 and candidate == 0:
                        seed, lane = COURSES[index]
                        frozen = json.loads(
                            (
                                args.output_root
                                / f"seed{seed}-lane{lane}-reproduction-actor/report.json"
                            ).read_text()
                        )
                        if any(report[k] != frozen[k] for k in ("trace_hash", "body_trace_hash")):
                            raise ValueError(
                                "initial learned-phase clone changed predecessor physics"
                            )
                    result.append(
                        {
                            "candidate": candidate,
                            "course_index": index,
                            "report_hash": report["report_hash"],
                            "outcome": {**outcome, "high_quality": high_quality(outcome)},
                        }
                    )
                    print(
                        f"MOTOR_CURRICULUM_AUDITED g={generation} c={candidate} course={index}",
                        flush=True,
                    )
            return result

        with ThreadPoolExecutor(max_workers=4) as pool:
            batches = [r for batch in pool.map(worker, range(4)) for r in batch]
        candidates = []
        for candidate in range(POPULATION):
            selected = sorted(
                (r for r in batches if r["candidate"] == candidate), key=lambda r: r["course_index"]
            )
            rows = [r["outcome"] for r in selected]
            entry = {
                "generation": generation,
                "candidate": candidate,
                "policy": str(args.output_root / f"policies/g{generation}-c{candidate}.json"),
                "rows": rows,
                "reports": [r["report_hash"] for r in selected],
                "score": list(rank(rows, reference)),
            }
            write_once(args.output_root / f"g{generation}-c{candidate}-result.json", entry)
            candidates.append(entry)
        candidates.sort(key=lambda r: (r["score"], -r["candidate"]), reverse=True)
        if best is None or candidates[0]["score"] > best["score"]:
            best = candidates[0]
            selected_policy = json.loads(Path(best["policy"]).read_text())
            best_vector = np.concatenate(
                (
                    np.asarray(selected_policy["knots_rad"]).ravel(),
                    [selected_policy["phase_gap_end_m"]],
                )
            )
        elite = [proposals[r["candidate"]] for r in candidates[:2]]
        center = 0.25 * center + 0.75 * np.mean(elite, axis=0)
        all_candidates.extend(candidates)
        write_once(
            args.output_root / f"generation-{generation}.json",
            {
                "candidates": candidates,
                "next_center": center.tolist(),
                "best_vector": best_vector.tolist(),
                "best": best,
            },
        )
        print(f"MOTOR_CURRICULUM_GENERATION g={generation} best={best['score']}", flush=True)
    if (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
    ):
        raise ValueError("source drift during curriculum")
    assert best is not None
    passed = bool(
        best["score"][0] == 1 and all(v == 0 for v in best["score"][1:6]) and best["score"][6] >= 10
    )
    summary = {
        "schema": "rsi_failure_curriculum_motor_training_v306",
        "activation_ceiling": "SIM_ONLY",
        "partition": "TRAIN_CONSUMED",
        "commitment": commitment,
        "candidates": all_candidates,
        "best": best,
        "independent_physical_episode_count": 24 + GENERATIONS * POPULATION * 12,
        "distinct_consumed_course_count": 12,
        "consumed_training_gate_passed": passed,
        "fresh_holdout_open_authorized": False,
        "promotion_authorized": False,
    }
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "training_summary.json", summary)
    print(json.dumps({"gate": passed, "report_hash": summary["report_hash"]}), flush=True)


if __name__ == "__main__":
    main()
