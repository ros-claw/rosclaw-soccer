"""SIM_ONLY eight-G1 causal check of pre-foot geometry plus early navigation brake."""

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
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "sonic": root / "src/rosclaw_soccer/providers/g1/receiving_sonic.py",
        "foot_teacher": root / "src/rosclaw_soccer/providers/g1/receiving_foot_capture.py",
    }
    hashes = {key: hash_bytes(path.read_bytes()) for key, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY brake probe directory required")
    parent: dict[str, Any] = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    parent_hash = parent.pop("report_hash")
    if (
        parent_hash != hash_json(parent)
        or parent["schema"] != "rosclaw_soccer.rsi.receiving_full_prepost_capture_probe.v1"
        or parent["precontact_foot_gain"] != 0.3
        or parent["postcontact_foot_gain"] != 0.3
        or parent["foot_offset_x_m"] != -0.18
        or parent["foot_offset_y_m"] != 0.03
        or parent["result"]["safe"] is not True
        or parent["own_nonfoot_contact_frames"]
        or parent["promotion_authorized"] is not False
        or parent["trace_hash"] != hash_bytes((baseline / "full-prepost.npz").read_bytes())
    ):
        raise ValueError("sealed safe full-eight-G1 pre/post geometry parent required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    teacher = ReceivingFootCaptureTeacher(0.3, 0.3)
    if (
        student.model_hash != parent["student_model_hash"]
        or teacher.contract_hash != parent["foot_teacher_hash"]
    ):
        raise ValueError("frozen student and exact teacher required")
    policy = captured / "zero-near-ball-parent.npz"
    if not policy.is_file():
        raise ValueError("frozen eight-G1 team policy required")
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        sonic_ball_follow_fast_replan=True,
        sonic_ball_follow_brake_distance_m=0.65,
        receiving_student=student,
        receiving_foot_capture_teacher=teacher,
        research_control_frame_limit=130,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    output_dir.mkdir(parents=True)
    trajectory_path = output_dir / "full-foot-brake.npz"
    np.savez_compressed(
        trajectory_path,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    own_code = agent_ids.index(COURSE.agent_id) + 1
    foot = np.flatnonzero(
        (trace["ball_contact_agent_code"] == own_code) & (trace["ball_contact_foot_code"] > 0)
    ).tolist()
    nonfoot = np.flatnonzero(trace["ball_nonfoot_contact_agent_code"] == own_code).tolist()
    if {key: hash_bytes(path.read_bytes()) for key, path in paths.items()} != hashes:
        raise RuntimeError("source changed during live eight-G1 physics")
    parent_explanation = parent["measurement"]["authoritative_explanation"]
    explanation = measurement["authoritative_explanation"]
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_full_foot_brake_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": hashes,
        "parent_report_hash": parent_hash,
        "student_model_hash": student.model_hash,
        "foot_teacher_hash": teacher.contract_hash,
        "early_brake_distance_m": 0.65,
        "fast_replan": True,
        "trace_hash": hash_bytes(trajectory_path.read_bytes()),
        "result": result_dict,
        "measurement": measurement,
        "own_foot_contact_frames": foot,
        "own_nonfoot_contact_frames": nonfoot,
        "parent_tail_ball_speed_mps": parent_explanation["tail_maximum_ball_speed_mps"],
        "parent_tail_foot_distance_m": parent_explanation["tail_maximum_foot_distance_m"],
        "candidate_tail_ball_speed_mps": explanation["tail_maximum_ball_speed_mps"],
        "candidate_tail_foot_distance_m": explanation["tail_maximum_foot_distance_m"],
        "candidate_controlled_reception": explanation["controlled_reception"],
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
    report = evaluate(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "safe": report["result"]["safe"],
                "foot_frames": report["own_foot_contact_frames"],
                "nonfoot_frames": report["own_nonfoot_contact_frames"],
                "tail_ball_speed_mps": report["candidate_tail_ball_speed_mps"],
                "tail_foot_distance_m": report["candidate_tail_foot_distance_m"],
                "controlled_reception": report["candidate_controlled_reception"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
