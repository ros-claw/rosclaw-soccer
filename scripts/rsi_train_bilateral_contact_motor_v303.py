"""SIM_ONLY independent-physics CEM training, with resumable audited evidence.

One plastic 36-parameter bilateral primitive; SONIC and swing actor stay frozen.
Consumed three-course pilot only: no holdout or team/video promotion authority.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.rsi.contact_motor_primitive import CAP_RAD, make_policy
from rosclaw_soccer.rsi.contact_motor_strike import make_policy as make_strike_policy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from scripts.rsi_collect_approach_lateral_tracking_v286 import _run
from scripts.rsi_collect_negative_side_approach_fresh_v287 import high_quality

COURSES = ((20261227, 4), (20261446, 0), (20261095, 2))
POPULATION = 8
GENERATIONS = 3


def write_once(path: Path, value: dict[str, Any]) -> None:
    if path.exists():
        if hash_json(json.loads(path.read_text(encoding="utf-8"))) != hash_json(value):
            raise ValueError(f"resume commitment differs: {path}")
        return
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def score(rows: list[dict[str, Any]], baseline: list[dict[str, Any]]) -> tuple[float, ...]:
    if len(rows) != 3 or len(baseline) != 3:
        raise ValueError("three paired consumed courses required")
    safe = all(r["minimum_pelvis_z_m"] >= 0.65 for r in rows)
    retention = bool(rows[2]["high_quality"])
    foot_loss = sum(
        b["clean_foot_only"] and not r["clean_foot_only"]
        for b, r in zip(baseline, rows, strict=True)
    )
    new_out = sum(
        b["maximum_lateral_excursion_m"] <= 4 < r["maximum_lateral_excursion_m"]
        for b, r in zip(baseline, rows, strict=True)
    )
    risk_rescued = sum(
        r["clean_foot_only"] and r["maximum_lateral_excursion_m"] <= 4 for r in rows[:2]
    )
    return (
        float(safe),
        float(retention),
        -float(foot_loss + new_out),
        float(risk_rescued),
        float(sum(r["high_quality"] for r in rows)),
        float(sum(r["reward"] for r in rows)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output-root", "isaac-python", "g1-usd", "model-root", "late-swing-policy"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--source-commit", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--phase-profile", choices=("preparation", "strike"), default="preparation")
    args = parser.parse_args()
    root = Path(__file__).resolve().parent.parent
    runner = root / "scripts/rsi_isaac_vector_first_touch.py"
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
    if head != args.source_commit or dirty or (args.output_root.exists() and not args.resume):
        parser.error("clean frozen commit and fresh output, or explicit resume required")
    version = 303 if args.phase_profile == "preparation" else 304
    protocol = json.loads(
        (root / f"docs/rsi/protocols/bilateral-contact-motor-v{version}.json").read_text(
            encoding="utf-8"
        )
    )
    if (
        protocol["courses"] != [list(c) for c in COURSES]
        or protocol["population"] != POPULATION
        or protocol["generations"] != GENERATIONS
    ):
        raise ValueError("training protocol drift")
    commitment = {
        "source_commit": head,
        "runner_hash": hash_bytes(runner.read_bytes()),
        "trainer_hash": hash_bytes(Path(__file__).read_bytes()),
        "asset_hash": hash_bytes(args.g1_usd.read_bytes()),
        "frozen_actor_file_hash": hash_bytes(args.late_swing_policy.read_bytes()),
        "protocol_hash": hash_json(protocol),
        "phase_profile": args.phase_profile,
        "activation_ceiling": "SIM_ONLY",
    }
    commitment_hash = hash_json(commitment)
    args.output_root.mkdir(parents=True, exist_ok=args.resume)
    (args.output_root / "logs").mkdir(exist_ok=args.resume)
    (args.output_root / "policies").mkdir(exist_ok=args.resume)
    write_once(args.output_root / "commitment.json", commitment)

    def run(
        seed: int, lane: int, gpu: int, arm: str, kind: str, gain: float, policy: Path | None = None
    ):
        parent = args.output_root / f"seed{seed}-lane{lane}-gain_12-parent/report.json"
        return _run(
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
            gain=gain,
            kind=kind,
            negative_only=True,
            motor_policy=policy,
            parent_report_override=parent if policy is not None else None,
            resume=args.resume,
        )

    def baseline_course(index: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        seed, lane = COURSES[index]
        rows, hashes = [], []
        for name, gain in (("gain_08", 0.8), ("gain_12", 1.2)):
            parent, _ = run(seed, lane, index, name, "parent", gain)
            report, outcome = run(seed, lane, index, name, "actor", gain)
            rows.append({**outcome, "high_quality": high_quality(outcome)})
            hashes.append({"parent": parent["report_hash"], "actor": report["report_hash"]})
        return rows, hashes

    with ThreadPoolExecutor(max_workers=3) as pool:
        base = list(pool.map(baseline_course, range(3)))
    baseline_08 = [item[0][0] for item in base]
    baseline_12 = [item[0][1] for item in base]
    write_once(
        args.output_root / "baseline.json", {"rows": base, "commitment_hash": commitment_hash}
    )
    rng = np.random.default_rng(20261001303)
    center = np.zeros((3, 12))
    all_candidates: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation, sigma in enumerate((0.04, 0.03, 0.022)):
        noise = rng.normal(size=(POPULATION // 2, 3, 12))
        proposals = np.clip(center + sigma * np.concatenate((noise, -noise)), -CAP_RAD, CAP_RAD)
        # The unmodified center is a retention anchor, not an independent sample.
        proposals[0] = center
        for index, knots in enumerate(proposals):
            write_once(
                args.output_root / f"policies/g{generation}-c{index}.json",
                (make_policy if args.phase_profile == "preparation" else make_strike_policy)(
                    knots, commitment_hash
                ),
            )

        def worker(gpu: int, generation: int = generation) -> list[dict[str, Any]]:
            result = []
            seed, lane = COURSES[gpu]
            for index in range(POPULATION):
                policy = args.output_root / f"policies/g{generation}-c{index}.json"
                report, outcome = run(
                    seed, lane, gpu, f"g{generation}-c{index}", "actor", 1.2, policy
                )
                if (
                    report["source_hash"] != commitment["runner_hash"]
                    or report["asset_hash"] != commitment["asset_hash"]
                ):
                    raise ValueError("frozen source or asset drift")
                if generation == 0 and index == 0:
                    original = json.loads(
                        (
                            args.output_root / f"seed{seed}-lane{lane}-gain_12-actor/report.json"
                        ).read_text()
                    )
                    if any(
                        report[key] != original[key] for key in ("trace_hash", "body_trace_hash")
                    ):
                        raise ValueError("zero motor primitive changed the physical baseline")
                result.append(
                    {
                        "candidate": index,
                        "row": {**outcome, "high_quality": high_quality(outcome)},
                        "report_hash": report["report_hash"],
                    }
                )
                print(
                    f"MOTOR_TRAIN_AUDITED g={generation} c={index} seed={seed} lane={lane}",
                    flush=True,
                )
            return result

        # Each course remains on the same physical GPU for parent and candidates.
        with ThreadPoolExecutor(max_workers=3) as pool:
            batches = list(pool.map(worker, range(3)))
        candidates = []
        for index in range(POPULATION):
            rows = [batch[index]["row"] for batch in batches]
            row = {
                "generation": generation,
                "candidate": index,
                "policy": str(args.output_root / f"policies/g{generation}-c{index}.json"),
                "rows": rows,
                "reports": [batch[index]["report_hash"] for batch in batches],
                "score": list(score(rows, baseline_08)),
            }
            write_once(args.output_root / f"g{generation}-c{index}-result.json", row)
            candidates.append(row)
        candidates.sort(key=lambda row: (row["score"], -row["candidate"]), reverse=True)
        all_candidates.extend(candidates)
        if best is None or candidates[0]["score"] > best["score"]:
            best = candidates[0]
        elite = [
            np.asarray(json.loads(Path(r["policy"]).read_text())["knots_rad"])
            for r in candidates[:2]
        ]
        center = 0.25 * center + 0.75 * np.mean(elite, axis=0)
        write_once(
            args.output_root / f"generation-{generation}.json",
            {"candidates": candidates, "next_center": center.tolist(), "best": best},
        )
    if (
        subprocess.check_output(["git", "status", "--porcelain"], cwd=root, text=True).strip()
        or hash_bytes(runner.read_bytes()) != commitment["runner_hash"]
    ):
        raise ValueError("source drift during physics training")
    assert best is not None
    passed = bool(
        best["score"][0] == 1
        and best["score"][1] == 1
        and best["score"][2] == 0
        and best["score"][3] == 2
        and (args.phase_profile != "strike" or best["score"][4] == 3)
    )
    summary = {
        "schema": f"rsi_bilateral_contact_motor_training_v{version}",
        "activation_ceiling": "SIM_ONLY",
        "commitment": commitment,
        "baseline_08": baseline_08,
        "baseline_12": baseline_12,
        "candidates": all_candidates,
        "best": best,
        "consumed_development_gate_passed": passed,
        "fresh_holdout_open_authorized": False,
        "promotion_authorized": False,
        "independent_physical_episode_count": 12 + GENERATIONS * POPULATION * len(COURSES),
        "distinct_consumed_course_count": 3,
    }
    summary["report_hash"] = hash_json(summary)
    write_once(args.output_root / "training_summary.json", summary)
    print(json.dumps({"gate": passed, "report_hash": summary["report_hash"]}), flush=True)


if __name__ == "__main__":
    main()
