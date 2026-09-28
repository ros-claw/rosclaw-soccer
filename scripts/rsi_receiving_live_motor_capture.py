"""SIM_ONLY read-only live SONIC observations on an exact 130-frame eight-G1 course."""

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
    student_tape: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    output_dir: Path,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "sonic": root / "src/rosclaw_soccer/providers/g1/receiving_sonic.py",
        "student": root / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py",
    }
    hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY live motor tape required")
    parent: dict[str, Any] = json.loads((student_tape / "report.json").read_text(encoding="utf-8"))
    parent_hash = parent.pop("report_hash")
    parent_path = student_tape / "team-motor-trace.npz"
    if (
        parent_hash != hash_json(parent)
        or parent["schema"] != "rosclaw_soccer.rsi.receiving_team_student_motor_capture.v1"
        or parent["trace_hash"] != hash_bytes(parent_path.read_bytes())
        or parent["promotion_authorized"] is not False
    ):
        raise ValueError("sealed full-world student physical tape required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if student.model_hash != parent["student_model_hash"]:
        raise ValueError("frozen student identity differs from world tape")
    policy = captured / "zero-near-ball-parent.npz"
    if not policy.is_file():
        raise ValueError("frozen eight-player tactical policy required")
    result, trace = simulate_r0_receiving_course(
        asset_root=asset_root,
        reference_policy_path=policy,
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=sonic_model_root,
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        receiving_student=student,
        research_control_frame_limit=130,
        capture_live_motor_observations=True,
    )
    compared = tuple(parent["physical_keys_compared"])
    errors = {}
    with np.load(parent_path, allow_pickle=False) as reference:
        for key in compared:
            measured = np.asarray(trace[key])
            expected = np.asarray(reference[key])[:130]
            if measured.shape != expected.shape:
                raise ValueError(f"live motor capture physical shape mismatch: {key}")
            errors[key] = float(np.max(np.abs(measured - expected)))
    if any(error != 0.0 for error in errors.values()):
        raise ValueError("read-only live motor capture changed measured eight-G1 physics")
    for key, shape in (
        ("sonic_recorded_qpos", (130, 43)),
        ("sonic_recorded_qvel", (130, 41)),
        ("sonic_recorded_navigation_command", (130, 3)),
        ("sonic_recorded_target", (130, 29)),
    ):
        if np.asarray(trace[key]).shape != shape:
            raise ValueError(f"complete live SONIC motor observation required: {key}")
    output_dir.mkdir(parents=True)
    tape_path = output_dir / "live-motor-tape.npz"
    np.savez_compressed(
        tape_path,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != hashes:
        raise RuntimeError("live motor capture source changed during eight-G1 physics")
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_live_motor_capture.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": hashes,
        "parent_report_hash": parent_hash,
        "student_model_hash": student.model_hash,
        "research_control_frame_limit": 130,
        "physical_keys_compared": list(compared),
        "physical_max_errors": errors,
        "physical_prefix_exact": True,
        "trace_hash": hash_bytes(tape_path.read_bytes()),
        "result": result.to_dict(),
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
        parser.add_argument("--" + name, type=Path, required=True)
    report = capture(**vars(parser.parse_args()))
    print(json.dumps({"report_hash": report["report_hash"], "trace_hash": report["trace_hash"]}))


if __name__ == "__main__":
    main()
