"""SIM_ONLY eight-G1 causal reachability test for a privileged coupled receiving teacher."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from rsi_receiving_student_shared_world_exam import COURSE, _measurement

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course


def probe(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    prior: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving_experiment": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "authoritative_evaluator": root / "src/rosclaw_soccer/training/receiving_rollout.py",
        "teacher": root / "src/rosclaw_soccer/growth/locomotion_contact_teacher.py",
    }
    source_hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY shared-teacher directory required")
    previous: dict[str, Any] = json.loads(prior.read_text(encoding="utf-8"))
    prior_hash = previous.pop("report_hash")
    parent_path = prior.parent / "parent.npz"
    if (
        prior_hash != hash_json(previous)
        or previous["schema"] != "rosclaw_soccer.rsi.receiving_student_shared_world_exam.v1"
        or previous["historical_parent_same_physics"] is not True
        or previous["unchanged_prefix_through_frame45"] is not True
        or previous["shared_world_authoritative_reception"] is not False
        or previous["rows"][0]["label"] != "parent"
        or previous["rows"][0]["trace_hash"] != hash_bytes(parent_path.read_bytes())
        or previous["course"]
        != {
            "agent_id": COURSE.agent_id,
            "seed": COURSE.seed,
            "speed_mps": COURSE.speed_mps,
            "lateral_m": COURSE.lateral_m,
        }
    ):
        raise ValueError("sealed consumed eight-G1 parent required")
    with np.load(parent_path, allow_pickle=False) as payload:
        parent = {key: np.asarray(payload[key]) for key in payload.files}
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=captured / "zero-near-ball-parent.npz",
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        research_coupled_teacher=True,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    parent_measurement = previous["rows"][0]["measurement"]
    prefix_keys = (
        "time",
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "red_finisher_pelvis_pose",
    )
    prefix_unchanged = all(
        np.array_equal(parent[key][:46], np.asarray(trace[key])[:46]) for key in prefix_keys
    )
    teacher_code = agent_ids.index(COURSE.agent_id) + 1
    own_foot_frames = np.flatnonzero(
        (np.asarray(trace["ball_contact_agent_code"]) == teacher_code)
        & (np.asarray(trace["ball_contact_foot_code"]) > 0)
    )
    own_nonfoot = bool(np.any(np.asarray(trace["ball_nonfoot_contact_agent_code"]) == teacher_code))
    first_foot_force = (
        None
        if not len(own_foot_frames)
        else float(trace["ball_contact_force_n"][own_foot_frames[0]])
    )
    speed_ratio = (
        measurement["frame86_ball_speed_mps"] / parent_measurement["frame86_ball_speed_mps"]
    )
    safe = bool(
        all(row["safe"] for row in result_dict["qualities"])
        and not result_dict.get("motor_fault_agents")
        and result_dict["robot_robot_contact_count"] == 0
    )
    diagnostic_gate = bool(
        safe
        and prefix_unchanged
        and len(own_foot_frames) > 0
        and not own_nonfoot
        and speed_ratio <= 0.90
    )
    numeric = {
        key: value
        for key, value in trace.items()
        if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
    }
    output_dir.mkdir(parents=True)
    trajectory_path = output_dir / "teacher.npz"
    np.savez_compressed(trajectory_path, **numeric)  # type: ignore[arg-type]
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("shared-teacher source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_shared_teacher_reachability.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "prior_report_hash": prior_hash,
        "parent_trace_hash": previous["rows"][0]["trace_hash"],
        "teacher_trace_hash": hash_bytes(trajectory_path.read_bytes()),
        "course": previous["course"],
        "teacher_window_frames": [46, 100],
        "teacher_is_privileged": True,
        "teacher_deployable": False,
        "prefix_unchanged": prefix_unchanged,
        "own_foot_frames": own_foot_frames.tolist(),
        "own_nonfoot_contact": own_nonfoot,
        "first_foot_force_n": first_foot_force,
        "teacher_active_substeps": int(np.sum(trace["research_receiving_teacher_active_substeps"])),
        "teacher_peak_torque_nm": float(np.max(trace["research_receiving_teacher_peak_torque_nm"])),
        "teacher_label_samples": len(trace["research_receiving_sample_label_nm"]),
        "frame86_ball_speed_ratio": speed_ratio,
        "all_bodies_safe": safe,
        "measurement": measurement,
        "result": result_dict,
        "diagnostic_training_source_gate_passed": diagnostic_gate,
        "shared_world_authoritative_reception": bool(
            measurement["authoritative_window"]["controlled_reception"]
        ),
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
    report = probe(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                key: report[key]
                for key in (
                    "report_hash",
                    "prefix_unchanged",
                    "own_foot_frames",
                    "own_nonfoot_contact",
                    "frame86_ball_speed_ratio",
                    "all_bodies_safe",
                    "diagnostic_training_source_gate_passed",
                    "shared_world_authoritative_reception",
                )
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
