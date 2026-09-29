"""SIM_ONLY native eight-G1 left receiving CEM with hard contact retention."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_precontact_expert_world_v150 import run
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json

SCHEMA = "rosclaw_soccer.rsi.r1_native_left_contact_cem_v154.result.v1"
POPULATION = 12
GENERATIONS = 2


def rank(summary: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(summary["safe"] and not summary["fault_agents"]),
        float(summary["first_foot_frame"] is not None),
        float(not summary["own_nonfoot_frames"]),
        float(summary["controlled_reception"]),
        -float(len(summary["own_nonfoot_frames"])),
        -float(summary["tail_maximum_foot_distance_m"]),
        -float(summary["tail_maximum_ball_speed_mps"]),
    )


def train(
    asset_root: Path,
    policy: Path,
    parent_report: Path,
    experts_report: Path,
    right_report: Path,
    authority_report: Path,
    output: Path,
    *,
    seed: int,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output.exists() or output.resolve().is_relative_to(root) or not 0 <= seed < 2**31:
        raise ValueError("new bounded external SIM_ONLY native training evidence required")
    parent = json.loads(parent_report.read_text())
    experts = json.loads(experts_report.read_text())
    right = json.loads(right_report.read_text())
    authority = json.loads(authority_report.read_text())
    if any(
        item["report_hash"] != hash_json({k: v for k, v in item.items() if k != "report_hash"})
        for item in (parent, experts, right, authority)
    ) or (
        authority["status"] != "REJECTED_EIGHT_G1_CONTROLLED_GATE"
        or not all(row["zero_physics_equal"] for row in authority["rows"])
        or experts["parent_report_hash"] != right["report_hash"]
        or parent["policy_hash"] != hash_bytes(policy.read_bytes())
    ):
        raise ValueError("sealed zero-authority parent and expert reports required")
    coordination = tuple(parent["selected"]["weights"])
    base = np.asarray(experts["best_left"]["weights"], dtype=np.float64) * 0.55
    right_weights = tuple(
        float(v) for v in np.asarray(right["best_right"]["weights"], dtype=np.float64) * 0.7
    )
    source_hashes = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_native_left_contact_cem_v154.py",
            "scripts/rsi_r1_precontact_expert_world_v150.py",
            "src/rosclaw_soccer/rsi/receiving_precontact_expert.py",
            "src/rosclaw_soccer/training/receiving_oracle_schedule.py",
        )
    }
    output.mkdir(parents=True)
    rng = np.random.default_rng(seed)
    mean = base.copy()
    std = np.full(12, 0.18, dtype=np.float64)
    generations: list[dict[str, Any]] = []
    best: dict[str, Any] | None = None
    for generation in range(GENERATIONS):
        weights = np.clip(rng.normal(mean, std, size=(POPULATION, 12)), -1, 1)
        weights[0] = base
        rows: list[dict[str, Any]] = []
        for index, candidate in enumerate(weights):
            left = tuple(float(v) for v in candidate)
            summary, _ = run(
                asset_root,
                policy,
                COURSES[0],
                coordination,
                left,
                (0.0,) * 12,
            )
            row = {
                "generation": generation + 1,
                "candidate": index,
                "weights": left,
                "summary": summary,
            }
            rows.append(row)
            if best is None or rank(summary) > rank(best["summary"]):
                best = row
            (output / "progress.json").write_text(
                json.dumps(
                    [*generations, {"generation": generation + 1, "rows": rows}],
                    indent=2,
                    allow_nan=False,
                )
                + "\n"
            )
            print(
                json.dumps({"generation": generation + 1, "candidate": index, "summary": summary}),
                flush=True,
            )
        rows.sort(key=lambda row: rank(row["summary"]), reverse=True)
        elite = np.stack([np.asarray(row["weights"]) for row in rows[:4]])
        mean = elite.mean(axis=0)
        std = np.maximum(0.05, elite.std(axis=0))
        generations.append(
            {
                "generation": generation + 1,
                "clean_count": sum(
                    row["summary"]["safe"]
                    and row["summary"]["first_foot_frame"] is not None
                    and not row["summary"]["own_nonfoot_frames"]
                    for row in rows
                ),
                "controlled_count": sum(row["summary"]["controlled_reception"] for row in rows),
                "rows": rows,
            }
        )
    assert best is not None
    paired = None
    if best["summary"]["safe"] and not best["summary"]["own_nonfoot_frames"]:
        left = tuple(best["weights"])
        paired = [
            run(asset_root, policy, course, coordination, left, right_weights)[0]
            for course in COURSES
        ]
    paired_controlled = paired is not None and all(
        row["safe"]
        and not row["fault_agents"]
        and row["first_foot_frame"] is not None
        and not row["own_nonfoot_frames"]
        and row["controlled_reception"]
        for row in paired
    )
    report = {
        "schema": SCHEMA,
        "authority_report_hash": authority["report_hash"],
        "source_hashes": source_hashes,
        "partition": "CONSUMED_NATIVE_EIGHT_G1_LEFT_CONTACT_CEM",
        "seed": seed,
        "rollout_count": POPULATION * GENERATIONS + (2 if paired is not None else 0),
        "generations": generations,
        "best": best,
        "paired": paired,
        "status": "DEVELOPMENT_NATIVE_PAIRED_CONTROLLED_UNVALIDATED"
        if paired_controlled
        else "REJECTED_NATIVE_PAIRED_CONTROLLED_GATE",
        "fresh_exam_required": True,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "report.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(
        hash_bytes((root / name).read_bytes()) != digest for name, digest in source_hashes.items()
    ):
        raise ValueError("source drift during native left contact learning")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--parent-report", type=Path, required=True)
    parser.add_argument("--experts-report", type=Path, required=True)
    parser.add_argument("--right-report", type=Path, required=True)
    parser.add_argument("--authority-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.parent_report,
        args.experts_report,
        args.right_report,
        args.authority_report,
        args.output,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
