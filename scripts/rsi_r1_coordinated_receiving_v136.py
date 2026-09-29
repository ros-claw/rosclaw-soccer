"""SIM_ONLY paired search over a mirrored full-body receiving synergy policy."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_r1_taskspace_feedback_v125 import COURSES

from rosclaw_soccer.rsi.receiving_coordinated_feedback import ReceivingCoordinatedFeedback
from rosclaw_soccer.rsi.team_receive_contact_evidence import (
    ReceiveContactEvidence,
    ReceiveContactMailbox,
)
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_rollout import explain_receiving_window, receiving_window
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

SCHEMA = "rosclaw_soccer.rsi.r1_coordinated_receiving_v136.result.v1"
START_FRAME = 15


def evaluate(
    asset_root: Path, policy: Path, course: ReceivingCourse, weights: tuple[float, ...]
) -> dict[str, Any]:
    schedule = ReceivingOracleSchedule(
        course.agent_id, "A1_body29", START_FRAME, 10, ((0.0,) * 29,)
    )
    mailbox = ReceiveContactMailbox(course.agent_id)
    actor = ReceivingCoordinatedFeedback(
        course.agent_id,
        schedule.contract_hash,
        mailbox,
        0.35,
        0.0,
        0.0,
        target_depth_m=0.25,
        target_lateral_m=0.12,
        coordination=weights,
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=course,
        scenario_id=f"s199.rsi.r1.coordinated-receiving.{course.seed}",
        configuration_profile="R1_CONTACT_TAP",
        oracle=schedule,
        feedback_provider=actor,
        physics_evidence_consumers={course.agent_id: ReceiveContactEvidence(mailbox)},
    )
    info = result.to_dict()
    ids = tuple(sorted(row["agent_id"] for row in info["qualities"]))
    _, outcome = receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    detail = explain_receiving_window(
        trace, agent_ids=ids, agent_id=course.agent_id, start=20, frames=100
    )
    code = ids.index(course.agent_id) + 1
    foot = np.asarray(trace["ball_contact_agent_code"])
    effector = np.asarray(trace["ball_contact_effector_code"])
    force = np.asarray(trace["ball_contact_force_n"])
    nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
    nonfoot_force = np.asarray(trace["ball_nonfoot_contact_force_n"])
    first = next(
        (i for i in range(20, 120) if foot[i] == code and effector[i] in (1, 2) and force[i] > 0),
        None,
    )
    shin = (
        []
        if first is None
        else [i for i in range(first, 120) if nonfoot[i] == code and nonfoot_force[i] > 0]
    )
    return {
        "course": vars(course),
        "safe": info["safe"],
        "physics_evidence_fault_agents": info["physics_evidence_fault_agents"],
        "first_own_foot_time_sec": mailbox.snapshot.first_own_foot_time_sec,
        "prefoot_nonfoot_count": mailbox.snapshot.prefoot_nonfoot_count,
        "own_shin_frames": shin,
        "outcome": outcome,
        "tail_maximum_foot_distance_m": detail["tail_maximum_foot_distance_m"],
        "tail_maximum_ball_speed_mps": detail["tail_maximum_ball_speed_mps"],
        "peak_actual_residual_rad": float(
            np.max(np.abs(np.asarray(trace["receiving_oracle_delta_rad"])))
        ),
        "result_hash": hash_json(info),
    }


def summarize(
    index: int, generation: int, weights: tuple[float, ...], trials: list[dict[str, Any]]
) -> dict[str, Any]:
    return {
        "candidate": index,
        "generation": generation,
        "weights": weights,
        "safe": all(t["safe"] and not t["physics_evidence_fault_agents"] for t in trials),
        "all_clean_first_foot": all(
            t["first_own_foot_time_sec"] is not None and t["prefoot_nonfoot_count"] == 0
            for t in trials
        ),
        "controlled_count": sum(t["outcome"]["controlled_reception"] for t in trials),
        "own_shin_frame_count": sum(len(t["own_shin_frames"]) for t in trials),
        "tail_distance_sum_m": sum(t["tail_maximum_foot_distance_m"] for t in trials),
        "tail_speed_sum_mps": sum(t["tail_maximum_ball_speed_mps"] for t in trials),
        "trials": trials,
    }


def rank(row: dict[str, Any]) -> tuple[float, ...]:
    return (
        float(row["safe"]),
        float(row["all_clean_first_foot"]),
        float(row["controlled_count"]),
        -float(row["own_shin_frame_count"]),
        -float(row["tail_distance_sum_m"]),
        -float(row["tail_speed_sum_mps"]),
    )


def train(
    asset_root: Path, policy: Path, output: Path, *, generations: int, population: int, seed: int
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if (
        output.exists()
        or output.resolve().is_relative_to(root)
        or not 1 <= generations <= 4
        or not 4 <= population <= 24
        or population % 2
        or not 0 <= seed < 2**31
    ):
        raise ValueError("new bounded external SIM_ONLY learning evidence required")
    sources = {
        name: hash_bytes((root / name).read_bytes())
        for name in (
            "scripts/rsi_r1_coordinated_receiving_v136.py",
            "src/rosclaw_soccer/rsi/receiving_coordinated_feedback.py",
            "src/rosclaw_soccer/rsi/receiving_taskspace_feedback.py",
            "src/rosclaw_soccer/training/receiving_feedback.py",
            "src/rosclaw_soccer/skills/team/independent_team_world.py",
        )
    }
    output.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "partition": "CONSUMED_BILATERAL_EIGHT_G1_DEVELOPMENT",
        "source_hashes": sources,
        "policy_hash": hash_bytes(policy.read_bytes()),
        "courses": [vars(c) for c in COURSES],
        "generations": generations,
        "population": population,
        "seed": seed,
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    rng = np.random.default_rng(seed)
    mean = np.zeros(8)
    sigma = np.full(8, 0.45)
    rows: list[dict[str, Any]] = []
    for generation in range(generations):
        candidates = [np.zeros(8)] if generation == 0 else [mean.copy()]
        for _ in range(population - 1):
            candidates.append(np.clip(mean + rng.normal(size=8) * sigma, -1, 1))
        generation_rows = []
        for candidate in candidates:
            weights = tuple(float(x) for x in candidate)
            trials = [evaluate(asset_root, policy, course, weights) for course in COURSES]
            row = summarize(len(rows), generation, weights, trials)
            rows.append(row)
            generation_rows.append(row)
            (output / "progress.json").write_text(
                json.dumps(rows, indent=2, allow_nan=False) + "\n"
            )
            print(json.dumps({k: v for k, v in row.items() if k != "trials"}), flush=True)
        elite = sorted(generation_rows, key=rank, reverse=True)[: max(2, population // 4)]
        mean = np.mean(np.asarray([row["weights"] for row in elite]), axis=0)
        sigma = np.maximum(
            0.1, 0.7 * sigma + 0.3 * np.std(np.asarray([row["weights"] for row in elite]), axis=0)
        )
    parent = rows[0]
    eligible = [row for row in rows if row["safe"] and row["all_clean_first_foot"]]
    selected = max(eligible, key=rank) if eligible else None
    report = {
        "schema": SCHEMA,
        "protocol_hash": protocol["protocol_hash"],
        "source_hashes": sources,
        "policy_hash": protocol["policy_hash"],
        "candidate_count": len(rows),
        "rollout_count": len(rows) * len(COURSES),
        "parent": parent,
        "selected": selected,
        "all_candidates": rows,
        "status": "DEVELOPMENT_CONTROLLED_GAIN_UNVALIDATED"
        if selected is not None and selected["controlled_count"] > parent["controlled_count"]
        else "REJECTED_NO_CONTROLLED_GAIN",
        "promotion_authorized": False,
        "video_authorized": False,
        "activation_ceiling": "SIM_ONLY",
    }
    report["report_hash"] = hash_json(report)
    (output / "selection.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if any(hash_bytes((root / name).read_bytes()) != digest for name, digest in sources.items()):
        raise ValueError("source drift during coordinated receiving development")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    report = train(
        args.asset_root,
        args.policy,
        args.output,
        generations=args.generations,
        population=args.population,
        seed=args.seed,
    )
    print(json.dumps({"status": report["status"], "report_hash": report["report_hash"]}))


if __name__ == "__main__":
    main()
