"""SIM_ONLY paired receiving transfer probe for a frozen shooting leg actor.

Transfer is exploratory. No receiver skill is promoted from a shooter exam.
"""

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


def probe(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    actor_path: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY transfer directory required")
    fixture = collection_fixture(asset_root)
    ids = tuple(sorted(cell.self_model.agent_id for cell in fixture.cells))
    body_hash = fixture.cells[0].growth_scope.body_hash
    policy = NearBallResidualPolicy.initialize(ids, body_hash, seed=92801)
    output_dir.mkdir(parents=True)
    policy_path = output_dir / "zero-near-ball-parent.npz"
    policy.save(policy_path)
    course = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
    rows: list[dict[str, Any]] = []
    for label, actor in (("parent", None), ("candidate", actor_path)):
        result, trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy_path,
            course=course,
            scenario_id="s199.rsi.feedback.receiving.92801",
            sonic_model_root=sonic_model_root,
            sonic_start_frame=0,
            feedback_actor_path=actor,
        )
        numeric = {
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        }
        trajectory_path = output_dir / f"{label}.npz"
        np.savez_compressed(trajectory_path, **numeric)  # type: ignore[arg-type]
        result_dict = result.to_dict()
        row: dict[str, Any] = {
            "label": label,
            "result": result_dict,
            "trace_hash": hash_bytes(trajectory_path.read_bytes()),
            "trace_keys": sorted(numeric),
            "feedback_actor_hash": (
                trace["feedback_actor_hash"].tolist() if actor is not None else None
            ),
            "feedback_foot_seen": (
                bool(trace["feedback_actor_foot_seen"][0]) if actor is not None else None
            ),
            "feedback_nonfoot_seen": (
                bool(trace["feedback_actor_nonfoot_seen"][0]) if actor is not None else None
            ),
        }
        rows.append(row)
        print(
            json.dumps(
                {
                    "label": label,
                    "safe": all(q["safe"] for q in result_dict["qualities"]),
                    "motor_fault_agents": result_dict.get("motor_fault_agents"),
                    "foot_seen": row["feedback_foot_seen"],
                    "nonfoot_seen": row["feedback_nonfoot_seen"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("transfer probe source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.feedback_receiving_transfer_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "actor_file_hash": hash_bytes(actor_path.read_bytes()),
        "policy_hash": policy.policy_hash,
        "course": {
            "agent_id": course.agent_id,
            "seed": course.seed,
            "speed_mps": course.speed_mps,
            "lateral_m": course.lateral_m,
        },
        "transfer_qualified": False,
        "promotion_authorized": False,
        "rows": rows,
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
    parser.add_argument("--actor-path", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = probe(**vars(parser.parse_args()))
    print(report["report_hash"])


if __name__ == "__main__":
    main()
