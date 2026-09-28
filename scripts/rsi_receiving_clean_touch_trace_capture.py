"""SIM_ONLY exact motor trace of the unpromoted clean-foot receiving teacher."""

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


def capture(
    *, asset_root: Path, sonic_model_root: Path, teacher_dir: Path, output_dir: Path
) -> dict[str, Any]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    if output_dir.exists() or output_dir.resolve().is_relative_to(source.resolve().parents[1]):
        raise ValueError("new external SIM_ONLY capture directory required")
    teacher_report: dict[str, Any] = json.loads(
        (teacher_dir / "report.json").read_text(encoding="utf-8")
    )
    commitment = teacher_report.pop("report_hash")
    if (
        commitment != hash_json(teacher_report)
        or teacher_report["promotion_authorized"] is not False
    ):
        raise ValueError("sealed unpromoted clean-touch teacher required")
    teacher = next(row for row in teacher_report["rows"] if row["label"] == "follow075")
    teacher_path = teacher_dir / "follow075.npz"
    if (
        teacher["gain"] != 0.75
        or not teacher["clean_foot"]
        or teacher["trace_hash"] != hash_bytes(teacher_path.read_bytes())
    ):
        raise ValueError("physical clean-foot teacher evidence required")
    fixture = collection_fixture(asset_root)
    ids = tuple(sorted(cell.self_model.agent_id for cell in fixture.cells))
    policy = NearBallResidualPolicy.initialize(
        ids, fixture.cells[0].growth_scope.body_hash, seed=92801
    )
    if policy.policy_hash != teacher_report["policy_hash"]:
        raise ValueError("same frozen team policy required")
    output_dir.mkdir(parents=True)
    policy_path = output_dir / "zero-near-ball-parent.npz"
    policy.save(policy_path)
    course = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=course,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        capture_ball_follow_targets=True,
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
        "red_finisher_pelvis_pose",
    )
    arrays = {key: np.asarray(trace[key]) for key in keys}
    if (
        arrays["sonic_recorded_qpos"].shape != (300, 43)
        or arrays["sonic_recorded_qvel"].shape != (300, 41)
        or any(arrays[key].shape != (300, 29) for key in keys[2:5])
        or any(not np.isfinite(array).all() for array in arrays.values())
    ):
        raise ValueError("finite complete measured clean-touch motor trace required")
    with np.load(teacher_path, allow_pickle=False) as previous:
        exact_arrays = all(np.array_equal(arrays[key], previous[key]) for key in keys[5:])
    same_world = result.to_dict()["trajectory_hash"] == teacher["result"]["trajectory_hash"]
    if not exact_arrays or not same_world:
        raise ValueError("read-only teacher recorder changed physical trajectory")
    path = output_dir / "motor-trace.npz"
    np.savez_compressed(path, **arrays)  # type: ignore[arg-type]
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("clean-touch capture source changed during physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_clean_touch_trace_capture.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "teacher_report_hash": commitment,
        "teacher_trace_hash": teacher["trace_hash"],
        "policy_hash": policy.policy_hash,
        "exact_physical_arrays": exact_arrays,
        "same_world_trajectory": same_world,
        "trace_hash": hash_bytes(path.read_bytes()),
        "world_result": result.to_dict(),
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
    parser.add_argument("--teacher-dir", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = capture(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("trace_hash", "report_hash")}, sort_keys=True))


if __name__ == "__main__":
    main()
