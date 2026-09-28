"""SIM_ONLY paired eight-G1 shared-world exam of a Fresh8-qualified receiving student."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_rollout import (
    explain_receiving_window,
    receiving_window,
)
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

COURSE = ReceivingCourse("red.finisher", 92801, 1.25, 0.08)
WINDOW_START = 20
WINDOW_FRAMES = 100


def _measurement(
    trace: dict[str, Any], *, agent_ids: tuple[str, ...], agent_id: str
) -> dict[str, Any]:
    _, outcome = receiving_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=WINDOW_START,
        frames=WINDOW_FRAMES,
    )
    explanation = explain_receiving_window(
        trace,
        agent_ids=agent_ids,
        agent_id=agent_id,
        start=WINDOW_START,
        frames=WINDOW_FRAMES,
    )
    ball_velocity = np.asarray(trace["ball_velocity"], dtype=np.float64)
    ball_pose = np.asarray(trace["ball_pose"], dtype=np.float64)
    pelvis_pose = np.asarray(trace["red_finisher_pelvis_pose"], dtype=np.float64)
    return {
        "authoritative_window": outcome,
        "authoritative_explanation": explanation,
        "frame86_ball_speed_mps": float(np.linalg.norm(ball_velocity[86, :2])),
        "frame86_ball_pelvis_distance_m": float(
            np.linalg.norm(ball_pose[86, :2] - pelvis_pose[86, :2])
        ),
    }


def examine(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "exam": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving_experiment": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "student_bridge": root / "src/rosclaw_soccer/providers/g1/qualified_receiving_student.py",
        "student_actor": root / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py",
        "authoritative_evaluator": root / "src/rosclaw_soccer/training/receiving_rollout.py",
    }
    source_hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY shared-world student directory required")
    capture: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    capture_hash = capture.pop("report_hash")
    if (
        capture_hash != hash_json(capture)
        or capture["schema"] != "rosclaw_soccer.rsi.receiving_clean_touch_trace_capture.v1"
        or capture["exact_physical_arrays"] is not True
        or capture["same_world_trajectory"] is not True
        or capture["promotion_authorized"] is not False
    ):
        raise ValueError("sealed clean-foot eight-G1 foundation required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    policy_path = captured / "zero-near-ball-parent.npz"
    if not policy_path.is_file():
        raise ValueError("same frozen team policy file required")
    output_dir.mkdir(parents=True)
    rows = []
    traces = {}
    for label, bundle in (("parent", None), ("student", student)):
        result, trace = simulate_r0_receiving_course(
            asset_root=asset_root,
            reference_policy_path=policy_path,
            course=COURSE,
            scenario_id="s199.rsi.ball.follow.92801",
            sonic_model_root=sonic_model_root,
            sonic_start_frame=0,
            sonic_ball_follow_gain=0.75,
            receiving_student=bundle,
        )
        result_dict = result.to_dict()
        agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
        measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
        numeric = {
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        }
        trace_path = output_dir / f"{label}.npz"
        np.savez_compressed(trace_path, **numeric)  # type: ignore[arg-type]
        traces[label] = trace
        rows.append(
            {
                "label": label,
                "trace_hash": hash_bytes(trace_path.read_bytes()),
                "result": result_dict,
                "measurement": measurement,
                "student_active_substeps": (
                    int(np.sum(trace["receiving_student_active_substeps"]))
                    if bundle is not None
                    else 0
                ),
            }
        )
        print(
            json.dumps(
                {
                    "label": label,
                    "trajectory_hash": result_dict["trajectory_hash"],
                    "controlled_reception": measurement["authoritative_window"][
                        "controlled_reception"
                    ],
                    "frame86_ball_speed_mps": measurement["frame86_ball_speed_mps"],
                },
                sort_keys=True,
            ),
            flush=True,
        )
    parent, child = traces["parent"], traces["student"]
    unchanged_prefix_keys = (
        "time",
        "ball_pose",
        "ball_velocity",
        "ball_contact_agent_code",
        "ball_contact_foot_code",
        "red_finisher_pelvis_pose",
    )
    prefix_unchanged = all(
        np.array_equal(np.asarray(parent[key])[:46], np.asarray(child[key])[:46])
        for key in unchanged_prefix_keys
    )
    with np.load(captured / "motor-trace.npz", allow_pickle=False) as historical:
        historical_parent_same = all(
            np.array_equal(np.asarray(parent[key]), np.asarray(historical[key]))
            for key in (
                "ball_pose",
                "ball_velocity",
                "ball_contact_agent_code",
                "ball_contact_foot_code",
                "red_finisher_pelvis_pose",
            )
        )
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != source_hashes:
        raise RuntimeError("shared-world student source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_student_shared_world_exam.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "capture_report_hash": capture_hash,
        "training_report_hash": student.training_report_hash,
        "fresh_report_hash": student.fresh_report_hash,
        "model_hash": student.model_hash,
        "course": {
            "agent_id": COURSE.agent_id,
            "seed": COURSE.seed,
            "speed_mps": COURSE.speed_mps,
            "lateral_m": COURSE.lateral_m,
        },
        "student_window_frames": [46, 100],
        "unchanged_prefix_keys": list(unchanged_prefix_keys),
        "unchanged_prefix_through_frame45": prefix_unchanged,
        "historical_parent_same_physics": historical_parent_same,
        "rows": rows,
        "shared_world_authoritative_reception": bool(
            rows[1]["measurement"]["authoritative_window"]["controlled_reception"]
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
    parser.add_argument("--warm-start", required=True, type=Path)
    parser.add_argument("--training", required=True, type=Path)
    parser.add_argument("--fresh", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    report = examine(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "prefix_unchanged": report["unchanged_prefix_through_frame45"],
                "historical_parent_same_physics": report["historical_parent_same_physics"],
                "authoritative_reception": report["shared_world_authoritative_reception"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
