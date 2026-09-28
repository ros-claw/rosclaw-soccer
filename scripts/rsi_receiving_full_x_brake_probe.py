"""SIM_ONLY eight-G1 causal test: longitudinal-only versus two-axis early brake."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_student_shared_world_exam import COURSE, _measurement

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.providers.g1.receiving_foot_capture import ReceivingFootCaptureTeacher
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_velocity_effects import receiving_velocity_effects


def evaluate(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    baseline: Path,
    output_dir: Path,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY longitudinal brake directory required")
    parent = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    parent_hash = parent.pop("report_hash")
    if (
        parent_hash != hash_json(parent)
        or parent["schema"] != "rosclaw_soccer.rsi.receiving_full_foot_brake_probe.v1"
        or parent["early_brake_distance_m"] != 0.65
        or parent["fast_replan"] is not True
        or parent["result"]["safe"] is not True
        or parent["promotion_authorized"] is not False
        or parent["trace_hash"] != hash_bytes((baseline / "full-foot-brake.npz").read_bytes())
    ):
        raise ValueError("sealed two-axis full-eight-G1 brake parent required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    teacher = ReceivingFootCaptureTeacher(0.3, 0.3)
    if (
        student.model_hash != parent["student_model_hash"]
        or teacher.contract_hash != parent["foot_teacher_hash"]
    ):
        raise ValueError("exact frozen motor student and foot teacher required")
    source_paths = {
        "probe": Path(__file__),
        "experiment": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "sonic": root / "src/rosclaw_soccer/providers/g1/receiving_sonic.py",
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "foot_teacher": root / "src/rosclaw_soccer/providers/g1/receiving_foot_capture.py",
    }
    source_hashes = {key: hash_bytes(path.read_bytes()) for key, path in source_paths.items()}
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=captured / "zero-near-ball-parent.npz",
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        sonic_ball_follow_fast_replan=True,
        sonic_ball_follow_brake_distance_m=0.65,
        sonic_ball_follow_brake_axis="x",
        receiving_student=student,
        receiving_foot_capture_teacher=teacher,
        research_control_frame_limit=130,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    if len(agent_ids) != 8 or COURSE.agent_id not in agent_ids:
        raise ValueError("complete eight-G1 roster required")
    code = agent_ids.index(COURSE.agent_id) + 1
    foot_frames = np.flatnonzero(
        (trace["ball_contact_agent_code"] == code) & (trace["ball_contact_foot_code"] > 0)
    ).tolist()
    nonfoot_frames = np.flatnonzero(trace["ball_nonfoot_contact_agent_code"] == code).tolist()
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    effects = (
        receiving_velocity_effects(trace, agent_id=COURSE.agent_id, agent_code=code)
        if foot_frames
        else None
    )
    output_dir.mkdir(parents=True)
    trajectory = output_dir / "full-x-brake.npz"
    np.savez_compressed(
        trajectory,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    if {key: hash_bytes(path.read_bytes()) for key, path in source_paths.items()} != source_hashes:
        raise RuntimeError("source changed during eight-G1 x-only brake physics")
    explanation = measurement["authoritative_explanation"]
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_full_x_brake_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "parent_report_hash": parent_hash,
        "student_model_hash": student.model_hash,
        "foot_teacher_hash": teacher.contract_hash,
        "brake_distance_m": 0.65,
        "brake_axis": "x",
        "trace_hash": hash_bytes(trajectory.read_bytes()),
        "result": result_dict,
        "measurement": measurement,
        "velocity_effects": effects,
        "own_foot_contact_frames": foot_frames,
        "own_nonfoot_contact_frames": nonfoot_frames,
        "tail_maximum_ball_speed_mps": explanation["tail_maximum_ball_speed_mps"],
        "tail_maximum_foot_distance_m": explanation["tail_maximum_foot_distance_m"],
        "controlled_reception": explanation["controlled_reception"],
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
        "warm-start",
        "training",
        "fresh",
        "baseline",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    result = evaluate(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": result["report_hash"],
                "safe": result["result"]["safe"],
                "foot_frames": result["own_foot_contact_frames"],
                "nonfoot_frames": result["own_nonfoot_contact_frames"],
                "tail_ball_speed_mps": result["tail_maximum_ball_speed_mps"],
                "tail_foot_distance_m": result["tail_maximum_foot_distance_m"],
                "controlled_reception": result["controlled_reception"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
