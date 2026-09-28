"""SIM_ONLY one live-feedback eight-G1 receiving candidate on a consumed course."""

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

FRAMES = 130


def evaluate(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    fidelity: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
    left_hip_roll_offset_rad: float,
    contact_impedance_scale: float = 1.0,
    posttouch_brake_nm: float = 0.0,
    research_handoff: bool = False,
    research_handoff_target_distance_m: float | None = None,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "student": root / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py",
        "bridge": root / "src/rosclaw_soccer/providers/g1/qualified_receiving_student.py",
        "sonic": root / "src/rosclaw_soccer/providers/g1/receiving_sonic.py",
    }
    hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY live candidate directory required")
    if (
        type(left_hip_roll_offset_rad) is not float
        or not np.isfinite(left_hip_roll_offset_rad)
        or not -0.08 <= left_hip_roll_offset_rad < 0.0
    ):
        raise ValueError("bounded negative left-hip-roll research candidate required")
    if (
        type(contact_impedance_scale) is not float
        or contact_impedance_scale not in (0.70, 0.85, 1.0)
        or contact_impedance_scale != 1.0
        and left_hip_roll_offset_rad != -0.06
    ):
        raise ValueError("predeclared bounded contact impedance requires fixed hip parent")
    if (
        type(posttouch_brake_nm) is not float
        or posttouch_brake_nm not in (-2.0, 0.0, 2.0)
        or posttouch_brake_nm != 0.0
        and (left_hip_roll_offset_rad != -0.06 or contact_impedance_scale != 1.0)
    ):
        raise ValueError("predeclared bounded post-touch brake requires fixed hip parent")
    if (
        type(research_handoff) is not bool
        or research_handoff
        and (
            left_hip_roll_offset_rad != -0.06
            or contact_impedance_scale != 1.0
            or posttouch_brake_nm != 0.0
        )
    ):
        raise ValueError("predeclared SIM_ONLY phase handoff requires fixed hip parent")
    if research_handoff_target_distance_m is not None and (
        type(research_handoff_target_distance_m) is not float
        or research_handoff_target_distance_m not in (0.42, 0.45)
        or not research_handoff
    ):
        raise ValueError("predeclared SIM_ONLY guarded chase radius required")
    gate: dict[str, Any] = json.loads(fidelity.read_text(encoding="utf-8"))
    gate_hash = gate.pop("report_hash")
    if (
        gate_hash != hash_json(gate)
        or gate["schema"] != "rosclaw_soccer.rsi.receiving_live_short_fidelity.v1"
        or gate["physical_prefix_exact"] is not True
        or gate["research_control_frame_limit"] != FRAMES
        or gate["promotion_authorized"] is not False
    ):
        raise ValueError("sealed live eight-G1 training environment required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if student.model_hash != gate["student_model_hash"]:
        raise ValueError("candidate must reuse the same frozen receiving actor")
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
        sonic_ball_follow_fast_replan=research_handoff,
        sonic_ball_follow_post_touch_chase=research_handoff,
        sonic_ball_follow_brake_distance_m=0.65 if research_handoff else None,
        sonic_ball_follow_post_touch_target_distance_m=research_handoff_target_distance_m,
        sonic_ball_follow_post_touch_speed_limit_mps=(
            0.35 if research_handoff_target_distance_m is not None else None
        ),
        research_student_handoff=research_handoff,
        receiving_student=student,
        receiving_student_hip_roll_offset_rad=left_hip_roll_offset_rad,
        receiving_student_contact_impedance_scale=contact_impedance_scale,
        receiving_student_posttouch_brake_nm=posttouch_brake_nm,
        research_control_frame_limit=FRAMES,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    output_dir.mkdir(parents=True)
    trajectory_path = output_dir / "live-candidate.npz"
    np.savez_compressed(
        trajectory_path,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    speed = np.linalg.norm(np.asarray(trace["ball_velocity"])[80:101, :2], axis=1)
    distance = np.linalg.norm(
        np.asarray(trace["ball_pose"])[80:101, :2]
        - np.asarray(trace["red_finisher_pelvis_pose"])[80:101, :2],
        axis=1,
    )
    own_code = agent_ids.index(COURSE.agent_id) + 1
    foot_frames = np.flatnonzero(
        (np.asarray(trace["ball_contact_agent_code"]) == own_code)
        & (np.asarray(trace["ball_contact_foot_code"]) > 0)
    )
    nonfoot_frames = np.flatnonzero(
        np.asarray(trace["ball_nonfoot_contact_agent_code"]) == own_code
    )
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != hashes:
        raise RuntimeError("live candidate source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_live_short_candidate.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": hashes,
        "live_fidelity_report_hash": gate_hash,
        "student_model_hash": student.model_hash,
        "course": {
            "agent_id": COURSE.agent_id,
            "seed": COURSE.seed,
            "speed_mps": COURSE.speed_mps,
            "lateral_m": COURSE.lateral_m,
        },
        "research_control_frame_limit": FRAMES,
        "left_hip_roll_offset_rad": left_hip_roll_offset_rad,
        "contact_impedance_scale": contact_impedance_scale,
        "posttouch_brake_nm": posttouch_brake_nm,
        "research_handoff": research_handoff,
        "research_handoff_target_distance_m": research_handoff_target_distance_m,
        "posttouch_brake_active_substeps": (
            int(np.sum(trace["receiving_student_brake_active_substeps"]))
            if posttouch_brake_nm != 0.0
            else 0
        ),
        "trace_hash": hash_bytes(trajectory_path.read_bytes()),
        "result": result_dict,
        "measurement": measurement,
        "own_foot_contact_frames": foot_frames.tolist(),
        "own_nonfoot_contact_frames": nonfoot_frames.tolist(),
        "frame86_ball_vy_mps": float(np.asarray(trace["ball_velocity"])[86, 1]),
        "tail_mean_speed_mps": float(np.mean(speed)),
        "tail_mean_pelvis_distance_m": float(np.mean(distance)),
        "development_score": float(np.mean(speed) + 0.4 * np.mean(distance)),
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
        "fidelity",
        "warm-start",
        "training",
        "fresh",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--left-hip-roll-offset-rad", type=float, required=True)
    parser.add_argument("--contact-impedance-scale", type=float, default=1.0)
    parser.add_argument("--posttouch-brake-nm", type=float, default=0.0)
    parser.add_argument("--research-handoff", action="store_true")
    parser.add_argument("--research-handoff-target-distance-m", type=float)
    report = evaluate(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "hip_roll_rad": report["left_hip_roll_offset_rad"],
                "frame86_speed_mps": report["measurement"]["frame86_ball_speed_mps"],
                "foot_contact_frames": report["own_foot_contact_frames"],
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
