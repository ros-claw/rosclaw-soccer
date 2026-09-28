"""SIM_ONLY paired eight-G1 current-ball navigation probe on a consumed course."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse


def probe(*, asset_root: Path, sonic_model_root: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY output directory required")
    fixture = collection_fixture(asset_root)
    ids = tuple(sorted(cell.self_model.agent_id for cell in fixture.cells))
    policy = NearBallResidualPolicy.initialize(
        ids, fixture.cells[0].growth_scope.body_hash, seed=92801
    )
    course = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
    output_dir.mkdir(parents=True)
    policy_path = output_dir / "zero-near-ball-parent.npz"
    policy.save(policy_path)
    rows = []
    for label, gain in (
        ("parent", None),
        ("follow025", 0.25),
        ("follow050", 0.5),
        ("follow075", 0.75),
    ):
        result, trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy_path,
            course=course,
            scenario_id="s199.rsi.ball.follow.92801",
            sonic_model_root=sonic_model_root,
            sonic_start_frame=0,
            sonic_ball_follow_gain=gain,
        )
        numeric = {
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        }
        trajectory_path = output_dir / f"{label}.npz"
        np.savez_compressed(trajectory_path, **numeric)  # type: ignore[arg-type]
        result_dict = result.to_dict()
        agents = [row["agent_id"] for row in result_dict["qualities"]]
        code = agents.index(course.agent_id) + 1
        foot = np.asarray(trace["ball_contact_agent_code"])
        foot_code = np.asarray(trace["ball_contact_foot_code"])
        nonfoot = np.asarray(trace["ball_nonfoot_contact_agent_code"])
        own_foot = bool(np.any((foot == code) & (foot_code > 0)))
        own_nonfoot = bool(np.any(nonfoot == code))
        row = {
            "label": label,
            "gain": gain,
            "safe": all(quality["safe"] for quality in result_dict["qualities"]),
            "motor_fault_agents": result_dict.get("motor_fault_agents"),
            "foot_seen": own_foot,
            "nonfoot_seen": own_nonfoot,
            "clean_foot": own_foot and not own_nonfoot,
            "trace_hash": hash_bytes(trajectory_path.read_bytes()),
            "result": result_dict,
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    key: row[key]
                    for key in ("label", "safe", "foot_seen", "nonfoot_seen", "clean_foot")
                }
            ),
            flush=True,
        )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("ball-follow source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_ball_follow_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "policy_hash": policy.policy_hash,
        "course": {
            "agent_id": course.agent_id,
            "seed": course.seed,
            "speed_mps": course.speed_mps,
            "lateral_m": course.lateral_m,
        },
        "rows": rows,
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--sonic-model-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(probe(**vars(parser.parse_args()))["report_hash"])


if __name__ == "__main__":
    main()
