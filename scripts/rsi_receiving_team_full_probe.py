"""SIM_ONLY full-eight-G1 off-trajectory check of one frozen receiving student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_student_shared_world_exam import COURSE, _measurement

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course


def examine(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    student_tape: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
    focal_probe_nm: float = 0.75,
    left_hip_roll_offset_rad: float = 0.0,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "exam": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "student": root / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py",
        "bridge": root / "src/rosclaw_soccer/providers/g1/qualified_receiving_student.py",
    }
    source_hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY probe directory required")
    if (
        type(focal_probe_nm) is not float
        or not np.isfinite(focal_probe_nm)
        or not 0 <= focal_probe_nm <= 1
        or type(left_hip_roll_offset_rad) is not float
        or not np.isfinite(left_hip_roll_offset_rad)
        or abs(left_hip_roll_offset_rad) > 0.08
        or (focal_probe_nm == 0.0) == (left_hip_roll_offset_rad == 0.0)
    ):
        raise ValueError("exactly one bounded SIM_ONLY probe required")
    tape = json.loads((student_tape / "report.json").read_text(encoding="utf-8"))
    tape_hash = tape.pop("report_hash")
    if (
        tape_hash != hash_json(tape)
        or tape["schema"] != "rosclaw_soccer.rsi.receiving_team_student_motor_capture.v1"
        or tape["promotion_authorized"] is not False
        or tape["trace_hash"] != hash_bytes((student_tape / "team-motor-trace.npz").read_bytes())
    ):
        raise ValueError("sealed student motor tape required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if (
        tape["student_model_hash"] != student.model_hash
        or tape["student_training_report_hash"] != student.training_report_hash
        or tape["student_fresh_report_hash"] != student.fresh_report_hash
    ):
        raise ValueError("student motor tape must bind frozen student")
    policy_path = captured / "zero-near-ball-parent.npz"
    if not policy_path.is_file():
        raise ValueError("frozen full-world team policy required")
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy_path,
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        receiving_student=student,
        receiving_student_probe_torque_nm=focal_probe_nm,
        receiving_student_hip_roll_offset_rad=left_hip_roll_offset_rad,
    )
    agent_ids = tuple(sorted(row["agent_id"] for row in result.to_dict()["qualities"]))
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    output_dir.mkdir(parents=True)
    trace_path = output_dir / "full-probe.npz"
    np.savez_compressed(
        trace_path,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    with np.load(student_tape / "team-motor-trace.npz", allow_pickle=False) as reference:
        parent_speed = float(np.linalg.norm(reference["ball_velocity"][86, :2]))
        parent_distance = float(
            np.linalg.norm(
                reference["ball_pose"][86, :2] - reference["red_finisher_pelvis_pose"][86, :2]
            )
        )
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("probe source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": (
            "rosclaw_soccer.rsi.receiving_team_full_hip_probe.v1"
            if left_hip_roll_offset_rad != 0.0
            else "rosclaw_soccer.rsi.receiving_team_full_probe.v1"
        ),
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "student_tape_report_hash": tape_hash,
        "student_model_hash": student.model_hash,
        "focal_probe_nm": focal_probe_nm,
        "focal_probe_frames": [55, 75] if focal_probe_nm != 0.0 else None,
        "left_hip_roll_offset_rad": left_hip_roll_offset_rad,
        "left_hip_roll_probe_frames": [46, 61] if left_hip_roll_offset_rad != 0.0 else None,
        "full_trace_hash": hash_bytes(trace_path.read_bytes()),
        "result": result.to_dict(),
        "measurement": measurement,
        "student_tape_frame86_speed_mps": parent_speed,
        "student_tape_frame86_distance_m": parent_distance,
        "full_frame86_speed_delta_mps": measurement["frame86_ball_speed_mps"] - parent_speed,
        "full_frame86_distance_delta_m": measurement["frame86_ball_pelvis_distance_m"]
        - parent_distance,
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    (output_dir / "report.json").write_text(
        json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "asset-root",
        "sonic-model-root",
        "captured",
        "student-tape",
        "warm-start",
        "training",
        "fresh",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--focal-probe-nm", type=float, default=0.75)
    parser.add_argument("--left-hip-roll-offset-rad", type=float, default=0.0)
    report = examine(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "full_frame86_speed_delta_mps": report["full_frame86_speed_delta_mps"],
                "full_frame86_distance_delta_m": report["full_frame86_distance_delta_m"],
                "controlled_reception": report["measurement"]["authoritative_window"][
                    "controlled_reception"
                ],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
