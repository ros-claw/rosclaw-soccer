"""SIM_ONLY read-only frame-45 eight-G1 motor and integration-state capture."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_student_shared_world_exam import COURSE

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course


def capture(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    prior: Path,
    output_dir: Path,
    student_bundle: QualifiedReceivingStudent | None = None,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving_experiment": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "checkpoint": root / "src/rosclaw_soccer/sim/physical_checkpoint.py",
    }
    source_hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY team-motor capture directory required")
    previous: dict[str, Any] = json.loads(prior.read_text(encoding="utf-8"))
    previous_hash = previous.pop("report_hash")
    prior_index = 1 if student_bundle is not None else 0
    prior_label = "student" if student_bundle is not None else "parent"
    if (
        previous_hash != hash_json(previous)
        or previous["schema"] != "rosclaw_soccer.rsi.receiving_student_shared_world_exam.v1"
        or previous["historical_parent_same_physics"] is not True
        or previous["rows"][prior_index]["label"] != prior_label
        or student_bundle is not None
        and previous["model_hash"] != student_bundle.model_hash
    ):
        raise ValueError("sealed unchanged eight-G1 parent or qualified student required")
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=captured / "zero-near-ball-parent.npz",
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        checkpoint_frame=45,
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        capture_team_motor_targets=True,
        receiving_student=student_bundle,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    historical_path = prior.parent / f"{prior_label}.npz"
    if previous["rows"][prior_index]["trace_hash"] != hash_bytes(historical_path.read_bytes()):
        raise ValueError("sealed historical eight-G1 trajectory required")
    physical_keys = (
        "time",
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "ball_nonfoot_contact_agent_code",
        *(
            f"{agent.replace('.', '_')}_{suffix}"
            for agent in agent_ids
            for suffix in ("pelvis_pose", "joint_position", "joint_velocity", "joint_torque")
        ),
    )
    with np.load(historical_path, allow_pickle=False) as historical:
        physical_arrays_exact = all(
            np.array_equal(np.asarray(trace[key]), np.asarray(historical[key]))
            for key in physical_keys
        )
    if not physical_arrays_exact:
        raise ValueError("read-only motor capture changed historical physical arrays")
    for agent in agent_ids:
        key = agent.replace(".", "_")
        for suffix, shape in (
            ("captured_pd_target", (300, 29)),
            ("captured_pd_kp", (300, 29)),
            ("captured_pd_kd", (300, 29)),
            ("captured_extra_torque_nm", (3000, 29)),
            ("captured_executed_torque_nm", (3000, 29)),
        ):
            values = np.asarray(trace[f"{key}_{suffix}"])
            if values.shape != shape or not np.isfinite(values).all():
                raise ValueError("complete finite per-player 500 Hz motor record required")
    checkpoint = np.asarray(trace["initial_integration_state"], dtype=np.float64)
    if (
        len(trace["initial_control_frame"]) != 1
        or int(trace["initial_control_frame"][0]) != 45
        or checkpoint.ndim != 1
        or not np.isfinite(checkpoint).all()
        or hash_bytes(checkpoint.astype("<f8").tobytes())
        != str(trace["initial_integration_hash"][0])
    ):
        raise ValueError("complete frame-45 MuJoCo integration state required")
    numeric = {
        key: value
        for key, value in trace.items()
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
    }
    output_dir.mkdir(parents=True)
    path = output_dir / "team-motor-trace.npz"
    np.savez_compressed(path, **numeric)  # type: ignore[arg-type]
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("team motor capture source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": (
            "rosclaw_soccer.rsi.receiving_team_student_motor_capture.v1"
            if student_bundle is not None
            else "rosclaw_soccer.rsi.receiving_team_motor_capture.v1"
        ),
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "prior_report_hash": previous_hash,
        "student_model_hash": None if student_bundle is None else student_bundle.model_hash,
        "student_training_report_hash": (
            None if student_bundle is None else student_bundle.training_report_hash
        ),
        "student_fresh_report_hash": (
            None if student_bundle is None else student_bundle.fresh_report_hash
        ),
        "policy_hash": result_dict["motor_policy_hashes"][COURSE.agent_id],
        "agent_ids": list(agent_ids),
        "course": previous["course"],
        "checkpoint_frame": 45,
        "model_hash": str(trace["initial_physics_hash"][0]),
        "integration_hash": str(trace["initial_integration_hash"][0]),
        "mujoco_version": str(trace["initial_mujoco_version"][0]),
        "trace_hash": hash_bytes(path.read_bytes()),
        "physical_keys_compared": list(physical_keys),
        "physical_arrays_exact": physical_arrays_exact,
        "world_result": result_dict,
        "read_only_replay_exact": True,
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
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--prior", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = capture(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: report[key] for key in ("report_hash", "trace_hash", "integration_hash")},
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
