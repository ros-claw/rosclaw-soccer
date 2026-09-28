"""SIM_ONLY native live-controller 130-frame prefix fidelity for all eight G1s."""

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
    left_hip_roll_offset_rad: float = 0.0,
    reference_full: Path | None = None,
) -> dict[str, Any]:
    source = Path(__file__)
    root = source.parents[1]
    paths = {
        "source": source,
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "receiving": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "student": root / "src/rosclaw_soccer/providers/g1/receiving_torque_student.py",
        "bridge": root / "src/rosclaw_soccer/providers/g1/qualified_receiving_student.py",
    }
    hashes = {name: hash_bytes(path.read_bytes()) for name, path in paths.items()}
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY live short-course directory required")
    if (
        type(left_hip_roll_offset_rad) is not float
        or not np.isfinite(left_hip_roll_offset_rad)
        or abs(left_hip_roll_offset_rad) > 0.08
        or (left_hip_roll_offset_rad == 0.0) != (reference_full is None)
    ):
        raise ValueError("paired bounded SIM_ONLY hip intervention and full reference required")
    tape: dict[str, Any] = json.loads((student_tape / "report.json").read_text(encoding="utf-8"))
    tape_hash = tape.pop("report_hash")
    tape_path = student_tape / "team-motor-trace.npz"
    if (
        tape_hash != hash_json(tape)
        or tape["schema"] != "rosclaw_soccer.rsi.receiving_team_student_motor_capture.v1"
        or tape["trace_hash"] != hash_bytes(tape_path.read_bytes())
        or tape["promotion_authorized"] is not False
    ):
        raise ValueError("sealed student full-world physical reference required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if tape["student_model_hash"] != student.model_hash:
        raise ValueError("student actor differs from frozen full-world tape")
    full_hash = None
    if reference_full is not None:
        full: dict[str, Any] = json.loads(
            (reference_full / "report.json").read_text(encoding="utf-8")
        )
        full_hash = full.pop("report_hash")
        if (
            full_hash != hash_json(full)
            or full["schema"] != "rosclaw_soccer.rsi.receiving_team_full_hip_probe.v1"
            or full["student_tape_report_hash"] != tape_hash
            or full["student_model_hash"] != student.model_hash
            or full["left_hip_roll_offset_rad"] != left_hip_roll_offset_rad
            or full["promotion_authorized"] is not False
            or full["full_trace_hash"]
            != hash_bytes((reference_full / "full-probe.npz").read_bytes())
        ):
            raise ValueError("matched sealed full-eight-G1 hip course required")
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
        receiving_student=student,
        receiving_student_hip_roll_offset_rad=left_hip_roll_offset_rad,
        research_control_frame_limit=FRAMES,
    )
    compared = tuple(tape["physical_keys_compared"])
    errors = {}
    reference_path = tape_path if reference_full is None else reference_full / "full-probe.npz"
    with np.load(reference_path, allow_pickle=False) as reference:
        for key in compared:
            candidate = np.asarray(trace[key])
            expected = np.asarray(reference[key])[:FRAMES]
            if candidate.shape != expected.shape:
                raise ValueError(f"research prefix physical shape mismatch: {key}")
            errors[key] = float(np.max(np.abs(candidate - expected)))
    exact = all(error == 0.0 for error in errors.values())
    output_dir.mkdir(parents=True)
    trace_path = output_dir / "live-short.npz"
    np.savez_compressed(
        trace_path,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    agent_ids = tuple(sorted(row["agent_id"] for row in result.to_dict()["qualities"]))
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    if {name: hash_bytes(path.read_bytes()) for name, path in paths.items()} != hashes:
        raise RuntimeError("live short-course source changed during physics")
    report: dict[str, Any] = {
        "schema": (
            "rosclaw_soccer.rsi.receiving_live_short_hip_fidelity.v1"
            if reference_full is not None
            else "rosclaw_soccer.rsi.receiving_live_short_fidelity.v1"
        ),
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": hashes,
        "student_tape_report_hash": tape_hash,
        "student_model_hash": student.model_hash,
        "reference_full_report_hash": full_hash,
        "left_hip_roll_offset_rad": left_hip_roll_offset_rad,
        "research_control_frame_limit": FRAMES,
        "physical_keys_compared": list(compared),
        "physical_max_errors": errors,
        "physical_prefix_exact": exact,
        "trace_hash": hash_bytes(trace_path.read_bytes()),
        "result": result.to_dict(),
        "measurement": measurement,
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
    parser.add_argument("--left-hip-roll-offset-rad", type=float, default=0.0)
    parser.add_argument("--reference-full", type=Path)
    report = examine(**vars(parser.parse_args()))
    print(
        json.dumps(
            {
                "report_hash": report["report_hash"],
                "physical_prefix_exact": report["physical_prefix_exact"],
                "frame86_ball_speed_mps": report["measurement"]["frame86_ball_speed_mps"],
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
