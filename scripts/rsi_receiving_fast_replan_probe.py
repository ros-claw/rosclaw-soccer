"""SIM_ONLY single-course fast-replan follow-up to a sealed ball-follow baseline."""

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
    *, asset_root: Path, sonic_model_root: Path, parent_dir: Path, output_dir: Path
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY output required")
    prior: dict[str, Any] = json.loads((parent_dir / "report.json").read_text(encoding="utf-8"))
    parent_hash = prior.pop("report_hash")
    if parent_hash != hash_json(prior) or prior["promotion_authorized"] is not False:
        raise ValueError("sealed unpromoted baseline required")
    selected = next(row for row in prior["rows"] if row["label"] == "follow075")
    if not selected["clean_foot"] or selected["gain"] != 0.75:
        raise ValueError("measured clean-foot development baseline required")
    fixture = collection_fixture(asset_root)
    ids = tuple(sorted(cell.self_model.agent_id for cell in fixture.cells))
    policy = NearBallResidualPolicy.initialize(
        ids, fixture.cells[0].growth_scope.body_hash, seed=92801
    )
    if policy.policy_hash != prior["policy_hash"]:
        raise ValueError("same frozen team policy required")
    course = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
    output_dir.mkdir(parents=True)
    policy_path = output_dir / "zero-near-ball-parent.npz"
    policy.save(policy_path)
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=course,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        sonic_ball_follow_fast_replan=True,
    )
    numeric = {
        key: value
        for key, value in trace.items()
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
    }
    trajectory_path = output_dir / "fast-replan.npz"
    np.savez_compressed(trajectory_path, **numeric)  # type: ignore[arg-type]
    result_dict = result.to_dict()
    code = [row["agent_id"] for row in result_dict["qualities"]].index(course.agent_id) + 1
    own_foot_frames = np.flatnonzero(
        (trace["ball_contact_agent_code"] == code) & (trace["ball_contact_foot_code"] > 0)
    )
    own_nonfoot = bool(np.any(trace["ball_nonfoot_contact_agent_code"] == code))
    first = int(own_foot_frames[0]) if len(own_foot_frames) else None
    after = min(299, first + 25) if first is not None else None
    ball_speed = (
        float(np.linalg.norm(trace["ball_velocity"][after, :2])) if after is not None else None
    )
    distance = (
        float(
            np.linalg.norm(
                trace["ball_pose"][after, :2] - trace["red_finisher_pelvis_pose"][after, :2]
            )
        )
        if after is not None
        else None
    )
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("fast-replan source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_fast_replan_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "parent_report_hash": parent_hash,
        "parent_trace_hash": selected["trace_hash"],
        "trace_hash": hash_bytes(trajectory_path.read_bytes()),
        "policy_hash": policy.policy_hash,
        "first_foot_frame": first,
        "nonfoot_seen": own_nonfoot,
        "clean_foot": first is not None and not own_nonfoot,
        "post_half_second_ball_speed_mps": ball_speed,
        "post_half_second_ball_pelvis_distance_m": distance,
        "safe": all(row["safe"] for row in result_dict["qualities"]),
        "motor_fault_agents": result_dict.get("motor_fault_agents"),
        "result": result_dict,
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
    parser.add_argument("--parent-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    print(json.dumps(probe(**vars(parser.parse_args())), sort_keys=True))


if __name__ == "__main__":
    main()
