"""SIM_ONLY exact observation/target capture from one consumed 8-G1 receive course."""

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


def capture(*, asset_root: Path, sonic_model_root: Path, output_dir: Path) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY motor trace directory required")
    fixture = collection_fixture(asset_root)
    ids = tuple(sorted(cell.self_model.agent_id for cell in fixture.cells))
    policy = NearBallResidualPolicy.initialize(
        ids, fixture.cells[0].growth_scope.body_hash, seed=92801
    )
    output_dir.mkdir(parents=True)
    policy_path = output_dir / "zero-near-ball-parent.npz"
    policy.save(policy_path)
    course = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=course,
        scenario_id="s199.rsi.feedback.receiving.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        capture_sonic_targets=True,
    )
    keys = (
        "sonic_recorded_qpos",
        "sonic_recorded_qvel",
        "sonic_recorded_target",
        "sonic_recorded_kp",
        "sonic_recorded_kd",
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
    )
    arrays = {key: np.asarray(trace[key]) for key in keys}
    if (
        arrays["sonic_recorded_qpos"].shape != (300, 43)
        or arrays["sonic_recorded_qvel"].shape != (300, 41)
        or arrays["sonic_recorded_target"].shape != (300, 29)
        or any(not np.isfinite(array).all() for array in arrays.values())
    ):
        raise ValueError("complete finite measured receiving motor trace required")
    path = output_dir / "motor-trace.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("trace collector source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_motor_trace_capture.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "course": {
            "agent_id": course.agent_id,
            "seed": course.seed,
            "speed_mps": course.speed_mps,
            "lateral_m": course.lateral_m,
        },
        "policy_hash": policy.policy_hash,
        "world_result": result.to_dict(),
        "trace_hash": hash_bytes(path.read_bytes()),
        "trace_keys": list(keys),
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
    report = capture(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("trace_hash", "report_hash")}, sort_keys=True))


if __name__ == "__main__":
    main()
