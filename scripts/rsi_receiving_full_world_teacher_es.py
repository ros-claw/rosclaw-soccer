"""Train a bounded receiving contact teacher against the live eight-G1 world."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsi_receiving_student_shared_world_exam import COURSE, _measurement

from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.providers.g1.receiving_foot_capture import ReceivingFootCaptureTeacher
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.training.receiving_experiment import simulate_r0_receiving_course
from rosclaw_soccer.training.receiving_velocity_effects import receiving_velocity_effects

LOW = np.asarray((0.10, 0.00, -0.28, -0.08), dtype=np.float64)
HIGH = np.asarray((0.65, 0.65, -0.10, 0.08), dtype=np.float64)
SEED = 20260928


def _candidate(
    paths: dict[str, Path], parameters: NDArray[np.float64], output_dir: Path
) -> dict[str, Any]:
    if output_dir.exists():
        raise ValueError("new candidate evidence directory required")
    student = QualifiedReceivingStudent.load(
        warm_start=paths["warm_start"], training=paths["training"], fresh=paths["fresh"]
    )
    if parameters.shape != (4,) or not np.isfinite(parameters).all():
        raise ValueError("finite four-parameter foot teacher required")
    teacher = ReceivingFootCaptureTeacher(
        pre_gain=float(parameters[0]),
        post_gain=float(parameters[1]),
        offset_x_m=float(parameters[2]),
        offset_y_m=float(parameters[3]),
    )
    result, trace = simulate_r0_receiving_course(
        asset_root=paths["asset_root"],
        reference_policy_path=paths["captured"] / "zero-near-ball-parent.npz",
        course=COURSE,
        scenario_id="s199.rsi.ball.follow.92801",
        sonic_model_root=paths["sonic_model_root"],
        sonic_start_frame=0,
        sonic_ball_follow_gain=0.75,
        receiving_student=student,
        receiving_foot_capture_teacher=teacher,
        research_control_frame_limit=130,
    )
    result_dict = result.to_dict()
    agent_ids = tuple(sorted(row["agent_id"] for row in result_dict["qualities"]))
    if len(agent_ids) != 8 or COURSE.agent_id not in agent_ids:
        raise ValueError("complete eight-G1 roster required")
    own_code = agent_ids.index(COURSE.agent_id) + 1
    foot_frames = np.flatnonzero(
        (trace["ball_contact_agent_code"] == own_code) & (trace["ball_contact_foot_code"] > 0)
    ).tolist()
    nonfoot_frames = np.flatnonzero(trace["ball_nonfoot_contact_agent_code"] == own_code).tolist()
    measurement = _measurement(trace, agent_ids=agent_ids, agent_id=COURSE.agent_id)
    diagnostic = (
        receiving_velocity_effects(trace, agent_id=COURSE.agent_id, agent_code=own_code)
        if foot_frames
        else None
    )
    explanation = measurement["authoritative_explanation"]
    speed = float(explanation["tail_maximum_ball_speed_mps"])
    distance = float(explanation["tail_maximum_foot_distance_m"])
    safe = bool(result_dict["safe"] and foot_frames and not nonfoot_frames)
    objective = (
        -max(speed / 0.35, distance / 0.35)
        - 0.25 * abs(diagnostic["outgoing_lateral_mps"]) / 0.35
        - 0.10 * abs(diagnostic["pelvis_x_displacement_first_to_tail_m"]) / 0.35
        if safe and diagnostic is not None
        else -1_000_000.0
    )
    output_dir.mkdir(parents=True)
    trajectory = output_dir / "full-world.npz"
    np.savez_compressed(
        trajectory,
        **{  # type: ignore[arg-type]
            key: value
            for key, value in trace.items()
            if isinstance(value, np.ndarray) and value.dtype.kind in "biufU"
        },
    )
    row: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_full_world_teacher_candidate.v1",
        "activation_ceiling": "SIM_ONLY",
        "parameters": parameters.tolist(),
        "teacher_hash": teacher.contract_hash,
        "student_model_hash": student.model_hash,
        "trace_hash": hash_bytes(trajectory.read_bytes()),
        "result": result_dict,
        "measurement": measurement,
        "diagnostic": diagnostic,
        "own_foot_contact_frames": foot_frames,
        "own_nonfoot_contact_frames": nonfoot_frames,
        "safe_clean_foot": safe,
        "tail_maximum_ball_speed_mps": speed,
        "tail_maximum_foot_distance_m": distance,
        "controlled_reception": bool(explanation["controlled_reception"]),
        "objective": float(objective),
        "promotion_authorized": False,
    }
    row["report_hash"] = hash_json(row)
    (output_dir / "report.json").write_text(
        json.dumps(row, sort_keys=True, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return {
        "parameters": row["parameters"],
        "teacher_hash": row["teacher_hash"],
        "report_hash": row["report_hash"],
        "trace_hash": row["trace_hash"],
        "safe_clean_foot": safe,
        "controlled_reception": row["controlled_reception"],
        "objective": row["objective"],
        "tail_maximum_ball_speed_mps": speed,
        "tail_maximum_foot_distance_m": distance,
        "output_dir": str(output_dir),
    }


def train(
    *,
    asset_root: Path,
    sonic_model_root: Path,
    captured: Path,
    warm_start: Path,
    training: Path,
    fresh: Path,
    baseline: Path,
    output_dir: Path,
    generations: int = 2,
    population: int = 8,
    workers: int = 4,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[1]
    if output_dir.exists() or output_dir.resolve().is_relative_to(root):
        raise ValueError("new external SIM_ONLY training directory required")
    if not (1 <= generations <= 3 and 4 <= population <= 12 and 1 <= workers <= 4):
        raise ValueError("bounded full-world teacher budget required")
    parent = json.loads((baseline / "report.json").read_text(encoding="utf-8"))
    parent_hash = parent.pop("report_hash")
    if (
        parent_hash != hash_json(parent)
        or parent["schema"] != "rosclaw_soccer.rsi.receiving_full_prepost_capture_probe.v1"
        or parent["precontact_foot_gain"] != 0.3
        or parent["postcontact_foot_gain"] != 0.3
        or parent["foot_offset_x_m"] != -0.18
        or parent["foot_offset_y_m"] != 0.03
        or parent["trace_hash"] != hash_bytes((baseline / "full-prepost.npz").read_bytes())
        or parent["result"]["safe"] is not True
        or parent["promotion_authorized"] is not False
    ):
        raise ValueError("sealed full-eight-G1 development parent required")
    student = QualifiedReceivingStudent.load(warm_start=warm_start, training=training, fresh=fresh)
    if student.model_hash != parent["student_model_hash"]:
        raise ValueError("same frozen receiving student required")
    source_paths = {
        "train": Path(__file__),
        "experiment": root / "src/rosclaw_soccer/training/receiving_experiment.py",
        "world": root / "src/rosclaw_soccer/skills/team/independent_team_world.py",
        "teacher": root / "src/rosclaw_soccer/providers/g1/receiving_foot_capture.py",
        "diagnostic": root / "src/rosclaw_soccer/training/receiving_velocity_effects.py",
    }
    source_hashes = {key: hash_bytes(path.read_bytes()) for key, path in source_paths.items()}
    paths = {
        "asset_root": asset_root,
        "sonic_model_root": sonic_model_root,
        "captured": captured,
        "warm_start": warm_start,
        "training": training,
        "fresh": fresh,
    }
    rng = np.random.default_rng(SEED)
    mean = np.asarray((0.30, 0.30, -0.18, 0.03), dtype=np.float64)
    std = np.asarray((0.10, 0.12, 0.035, 0.035), dtype=np.float64)
    incumbent = mean.copy()
    output_dir.mkdir(parents=True)
    records: list[dict[str, Any]] = []
    for generation in range(generations):
        candidates = np.clip(rng.normal(mean, std, size=(population, 4)), LOW, HIGH)
        candidates[0] = incumbent
        if generation == 0:
            candidates[1] = np.asarray((0.3, 0.3, -0.18, -0.03))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [
                pool.submit(
                    _candidate,
                    paths,
                    np.asarray(parameters, dtype=np.float64),
                    output_dir / f"generation-{generation:02d}" / f"candidate-{index:02d}",
                )
                for index, parameters in enumerate(candidates)
            ]
            rows = [future.result() for future in futures]
        for index, row in enumerate(rows):
            row["generation"] = generation
            row["index"] = index
        records.extend(rows)
        ranked = sorted(rows, key=lambda row: row["objective"], reverse=True)
        elite = np.asarray([row["parameters"] for row in ranked[: max(2, population // 4)]])
        mean = np.mean(elite, axis=0)
        std = np.maximum(np.asarray((0.02, 0.02, 0.008, 0.008)), np.std(elite, axis=0) * 1.2)
        winner = max(records, key=lambda row: row["objective"])
        incumbent = np.asarray(winner["parameters"], dtype=np.float64)
        print(
            json.dumps(
                {
                    "generation": generation,
                    "actual_rollouts": len(records),
                    "best_speed_mps": winner["tail_maximum_ball_speed_mps"],
                    "best_foot_distance_m": winner["tail_maximum_foot_distance_m"],
                    "any_controlled_reception": any(r["controlled_reception"] for r in records),
                },
                sort_keys=True,
            ),
            flush=True,
        )
        if any(row["controlled_reception"] for row in records):
            break
    if {key: hash_bytes(path.read_bytes()) for key, path in source_paths.items()} != source_hashes:
        raise RuntimeError("source changed during full-world teacher training")
    winner = max(records, key=lambda row: row["objective"])
    report: dict[str, Any] = {
        "schema": "rosclaw_soccer.rsi.receiving_full_world_teacher_es.v1",
        "activation_ceiling": "SIM_ONLY",
        "source_hashes": source_hashes,
        "parent_report_hash": parent_hash,
        "student_model_hash": student.model_hash,
        "seed": SEED,
        "generation_budget": generations,
        "population": population,
        "actual_rollouts": len(records),
        "winner": winner,
        "candidates": records,
        "development_task_gate_passed": bool(winner["controlled_reception"]),
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
    parser.add_argument("--generations", type=int, default=2)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    report = train(**vars(parser.parse_args()))
    print(json.dumps({"report_hash": report["report_hash"], "winner": report["winner"]}))


if __name__ == "__main__":
    main()
