"""SIM_ONLY paired eight-G1 audit of a bounded live foot-capture teacher."""

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


def _sealed(path: Path, schema: str) -> tuple[dict[str, Any], str]:
    value: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = value.pop("report_hash")
    if commitment != hash_json(value) or value.get("schema") != schema:
        raise ValueError(f"sealed {schema} required")
    return value, str(commitment)


def evaluate(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    proxy_gate: Path,
    proxy_candidate: Path,
    baseline: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
    precontact_foot_gain: float = 0.3,
    postcontact_foot_gain: float = 0.3,
    foot_offset_x_m: float = -0.18,
    foot_offset_y_m: float = 0.03,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "sonic": root / "src/rosclaw_soccer/providers/g1/receiving_sonic.py",
        "proxy": root / "scripts/rsi_receiving_single_live_sonic_proxy.py",
        "foot_teacher": root / "src/rosclaw_soccer/providers/g1/receiving_foot_capture.py",
    }
    source_hashes = {key: hash_bytes(path.read_bytes()) for key, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY eight-G1 probe directory required")
    teacher = ReceivingFootCaptureTeacher(
        pre_gain=precontact_foot_gain,
        post_gain=postcontact_foot_gain,
        offset_x_m=foot_offset_x_m,
        offset_y_m=foot_offset_y_m,
    )
    gate, gate_hash = _sealed(
        proxy_gate / "report.json", "rosclaw_soccer.rsi.receiving_single_live_proxy_gate.v1"
    )
    candidate, candidate_hash = _sealed(
        proxy_candidate / "report.json",
        "rosclaw_soccer.rsi.receiving_single_live_precontact_foot.v1",
    )
    parent, parent_hash = _sealed(
        baseline / "report.json", "rosclaw_soccer.rsi.receiving_live_short_fidelity.v1"
    )
    if (
        gate["local_training_proxy_authorized"] is not True
        or candidate["source_hashes"]["source"] != source_hashes["proxy"]
        or candidate["foot_capture_gain"] != postcontact_foot_gain
        or candidate["precontact_foot_gain"] != precontact_foot_gain
        or candidate["foot_teacher_hash"] != teacher.contract_hash
        or candidate["foot_offset_x_m"] != foot_offset_x_m
        or candidate["foot_offset_y_m"] != foot_offset_y_m
        or candidate["capture_report_hash"] != gate["capture_report_hash"]
        or candidate["trajectory_hash"]
        != hash_bytes((proxy_candidate / "single-live.npz").read_bytes())
        or parent["physical_prefix_exact"] is not True
        or parent["research_control_frame_limit"] != 130
    ):
        raise ValueError("matched sealed live proxy, candidate, and parent required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if student.model_hash != candidate["student_model_hash"]:
        raise ValueError("same frozen student required")
    policy = captured / "zero-near-ball-parent.npz"
    if not policy.is_file():
        raise ValueError("frozen shared-world team policy required")
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        receiving_student=student,
        receiving_foot_capture_teacher=teacher,
        research_control_frame_limit=130,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    output_dir.mkdir(parents=True)
    trajectory_path = output_dir / "full-prepost.npz"
    np.savez_compressed(
        trajectory_path,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    full_speed = float(measurement["frame86_ball_speed_mps"])
    full_distance = float(measurement["frame86_ball_pelvis_distance_m"])
    with np.load(proxy_candidate / "single-live.npz", allow_pickle=False) as proxy_trace:
        proxy_ball = proxy_trace["ball_pose"]
        proxy_velocity = proxy_trace["ball_velocity"]
        proxy_pelvis = proxy_trace["pelvis_pose"]
        proxy_speed = float(np.linalg.norm(proxy_velocity[86, :2]))
        proxy_distance = float(np.linalg.norm(proxy_ball[86, :2] - proxy_pelvis[86, :2]))
        ball_position_max_error = float(np.max(np.abs(proxy_ball - trace["ball_pose"][:130])))
        ball_velocity_max_error = float(
            np.max(np.abs(proxy_velocity - trace["ball_velocity"][:130]))
        )
    own_code = agent_ids.index(COURSE.agent_id) + 1
    foot_frames = np.flatnonzero(
        (trace["ball_contact_agent_code"] == own_code) & (trace["ball_contact_foot_code"] > 0)
    ).tolist()
    nonfoot_frames = np.flatnonzero(trace["ball_nonfoot_contact_agent_code"] == own_code).tolist()
    if {key: hash_bytes(path.read_bytes()) for key, path in paths.items()} != source_hashes:
        raise RuntimeError("source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_full_prepost_capture_probe.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "proxy_gate_report_hash": gate_hash,
        "proxy_candidate_report_hash": candidate_hash,
        "baseline_report_hash": parent_hash,
        "student_model_hash": student.model_hash,
        "precontact_foot_gain": precontact_foot_gain,
        "postcontact_foot_gain": postcontact_foot_gain,
        "foot_offset_x_m": foot_offset_x_m,
        "foot_offset_y_m": foot_offset_y_m,
        "foot_teacher_hash": teacher.contract_hash,
        "trace_hash": hash_bytes(trajectory_path.read_bytes()),
        "result": result_dict,
        "measurement": measurement,
        "own_foot_contact_frames": foot_frames,
        "own_nonfoot_contact_frames": nonfoot_frames,
        "proxy_foot_contact_frames": candidate["proxy_foot_frames"],
        "proxy_nonfoot_contact_frames": candidate["proxy_nonfoot_frames"],
        "full_frame86_ball_speed_mps": full_speed,
        "proxy_frame86_ball_speed_mps": proxy_speed,
        "frame86_ball_speed_error_mps": abs(full_speed - proxy_speed),
        "full_frame86_ball_pelvis_distance_m": full_distance,
        "proxy_frame86_ball_pelvis_distance_m": proxy_distance,
        "frame86_ball_pelvis_distance_error_m": abs(full_distance - proxy_distance),
        "ball_position_max_error_m": ball_position_max_error,
        "ball_velocity_max_error_mps": ball_velocity_max_error,
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
        "proxy-gate",
        "proxy-candidate",
        "baseline",
        "warm-start",
        "training",
        "fresh",
        "output-dir",
    ):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--precontact-foot-gain", type=float, default=0.3)
    parser.add_argument("--postcontact-foot-gain", type=float, default=0.3)
    parser.add_argument("--foot-offset-x-m", type=float, default=-0.18)
    parser.add_argument("--foot-offset-y-m", type=float, default=0.03)
    report = evaluate(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "full_frame86_ball_speed_mps": report["full_frame86_ball_speed_mps"],
                "proxy_frame86_ball_speed_mps": report["proxy_frame86_ball_speed_mps"],
                "frame86_ball_speed_error_mps": report["frame86_ball_speed_error_mps"],
                "foot_frames": report["own_foot_contact_frames"],
                "nonfoot_frames": report["own_nonfoot_contact_frames"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
