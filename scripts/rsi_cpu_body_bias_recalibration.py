"""SIM_ONLY exact CPU CEM recalibration of a frozen 10-joint receiving actor.

Only ten feedback bias weights are plastic. The measured SONIC trajectory,
left-leg/body matrices, contact safety and reserved Fresh8 remain untouched.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from rsi_mjx_clean_touch_body_coord_es import FULL_FEATURES, FULL_JOINTS, _validate_body_joints
from rsi_mjx_clean_touch_control_es import COURSES, FRESH8, _capture
from rsi_receiving_contact_dynamics_audit import _run

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_model

SCHEMA = "rosclaw_soccer.rsi.cpu_body_bias_recalibration.v1"
WEIGHTS = len(FULL_JOINTS) * (FULL_FEATURES + 1)
BIAS_START = len(FULL_JOINTS) * FULL_FEATURES


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    clean = [
        row["first_contact"] is not None
        and row["first_contact"]["kind"] == "foot"
        and row["first_nonfoot_contact"] is None
        and row["foot_normal_impulse_ns"] > 0
        for row in rows
    ]
    safe = [
        row["minimum_pelvis_height_m"] >= 0.65 and row["maximum_tilt_rad"] < 0.30 for row in rows
    ]
    return {
        "clean_foot_count": sum(clean),
        "safe_count": sum(safe),
        "nonfoot_count": sum(row["first_nonfoot_contact"] is not None for row in rows),
        "mean_ball_speed_mps": float(np.mean([row["exam_ball_speed_mps"] for row in rows])),
        "mean_ball_pelvis_distance_m": float(
            np.mean([row["exam_ball_pelvis_distance_m"] for row in rows])
        ),
        "minimum_pelvis_height_m": min(row["minimum_pelvis_height_m"] for row in rows),
        "maximum_tilt_rad": max(row["maximum_tilt_rad"] for row in rows),
    }


def train(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    gpu_candidate: Path,
    cpu_center: Path,
    output_dir: Path,
    generations: int,
    population: int,
    seed: int,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_receiving_contact_dynamics_audit.py")
    source_hash, helper_hash = hash_bytes(source.read_bytes()), hash_bytes(helper.read_bytes())
    if (
        output_dir.exists()
        or output_dir.resolve().is_relative_to(source.resolve().parents[1])
        or not 1 <= generations <= 16
        or not 4 <= population <= 32
        or population % 2
        or not 0 <= seed < 2**31
    ):
        raise ValueError("bounded external SIM_ONLY CPU training required")
    candidate: dict[str, Any] = json.loads(gpu_candidate.read_text(encoding="utf-8"))
    candidate_hash = candidate.pop("report_hash")
    center: dict[str, Any] = json.loads(cpu_center.read_text(encoding="utf-8"))
    center_hash = center.pop("report_hash")
    weights = np.asarray(candidate["full_weights"], dtype=np.float32)
    if (
        candidate_hash != hash_json(candidate)
        or center_hash != hash_json(center)
        or center["candidate_hash"] != candidate_hash
        or center["cpu_center_gate_passed"] is not False
        or center["fresh8_opened"] is not False
        or weights.shape != (WEIGHTS,)
        or not np.isfinite(weights).all()
        or candidate["full_weights_hash"] != hash_bytes(weights.tobytes())
    ):
        raise ValueError("sealed failed-center GPU warm start required")
    arrays = _capture(captured, fidelity)
    model = build_g1_stadium_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    original = weights.astype(np.float64)
    zero = np.zeros(WEIGHTS, dtype=np.float64)

    def evaluate(full: NDArray[np.float64]) -> list[dict[str, Any]]:
        return [_run(model, arrays, full, FULL_JOINTS, course) for course in COURSES]

    parent_rows = evaluate(zero)
    warm_rows = evaluate(original)
    parent = _summary(parent_rows)
    warm = _summary(warm_rows)
    output_dir.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "compiled_model_hash": compiled_model_hash(model),
        "capture_hash": hash_bytes((captured / "motor-trace.npz").read_bytes()),
        "gpu_candidate_hash": candidate_hash,
        "cpu_center_hash": center_hash,
        "train_courses": [list(row) for row in COURSES],
        "reserved_fresh8": [list(row) for row in FRESH8],
        "plastic_coordinates": list(range(BIAS_START, WEIGHTS)),
        "generations": generations,
        "population": population,
        "seed": seed,
        "promotion_authorized": False,
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )

    def fitness(rows: list[dict[str, Any]]) -> float:
        summary = _summary(rows)
        speed_ratio = summary["mean_ball_speed_mps"] / parent["mean_ball_speed_mps"]
        distance_ratio = (
            summary["mean_ball_pelvis_distance_m"] / parent["mean_ball_pelvis_distance_m"]
        )
        value = (
            40 * summary["clean_foot_count"]
            + 3 * summary["safe_count"]
            - 40 * summary["nonfoot_count"]
            - 100 * (len(COURSES) - summary["safe_count"])
            - 40 * speed_ratio
            - 25 * distance_ratio
            - 30 * max(0.0, speed_ratio / 0.85 - 1)
            - 30 * max(0.0, distance_ratio / 0.90 - 1)
        )
        return float(value)

    rng = np.random.default_rng(seed)
    mean = original[BIAS_START:].astype(np.float32)
    sigma = 0.18
    best_full = original.copy()
    best_rows = warm_rows
    best_fitness = fitness(warm_rows)
    history = []
    print(
        json.dumps({"parent": parent, "warm_start": warm, "warm_fitness": best_fitness}), flush=True
    )
    for generation in range(generations):
        biases = np.clip(
            rng.normal(mean, sigma, size=(population, len(FULL_JOINTS))).astype(np.float32),
            -2.0,
            2.0,
        )
        ranked = []
        for index, bias in enumerate(biases):
            full = original.copy()
            full[BIAS_START:] = bias
            rows = evaluate(full)
            ranked.append((fitness(rows), index, rows))
        ranked.sort(key=lambda row: row[0], reverse=True)
        elite = [biases[index] for _, index, _ in ranked[: max(2, population // 4)]]
        mean = np.mean(elite, axis=0).astype(np.float32)
        sigma = max(0.025, sigma * 0.85)
        winner_fitness, winner_index, winner_rows = ranked[0]
        if winner_fitness > best_fitness:
            best_fitness = winner_fitness
            best_full = original.copy()
            best_full[BIAS_START:] = biases[winner_index]
            best_rows = winner_rows
        row = {
            "generation": generation,
            "winner_fitness": winner_fitness,
            "best_fitness": best_fitness,
            "winner": _summary(winner_rows),
            "best": _summary(best_rows),
            "sigma": sigma,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        (output_dir / "progress.json").write_text(
            json.dumps(history, sort_keys=True, indent=2, allow_nan=False) + "\n"
        )
    if (
        hash_bytes(source.read_bytes()) != source_hash
        or hash_bytes(helper.read_bytes()) != helper_hash
    ):
        raise RuntimeError("CPU recalibration source changed during physics")
    best32 = best_full.astype(np.float32)
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": protocol["protocol_hash"],
        "parent": parent,
        "warm_start": warm,
        "best": _summary(best_rows),
        "best_fitness": best_fitness,
        "parent_courses": parent_rows,
        "warm_start_courses": warm_rows,
        "best_courses": best_rows,
        "full_weights": best32.tolist(),
        "full_weights_hash": hash_bytes(best32.tobytes()),
        "fresh8_opened": False,
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    (output_dir / "candidate.json").write_text(
        json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--captured", required=True, type=Path)
    parser.add_argument("--fidelity", required=True, type=Path)
    parser.add_argument("--gpu-candidate", required=True, type=Path)
    parser.add_argument("--cpu-center", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--population", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    report = train(**vars(parser.parse_args()))
    print(json.dumps({key: report[key] for key in ("best", "report_hash")}, sort_keys=True))


if __name__ == "__main__":
    main()
