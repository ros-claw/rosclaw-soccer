"""SIM_ONLY MJX body-coordination ES atop frozen clean-touch leg feedback.

Only right support leg and waist residual weights are trained; the selected
left-leg clean-touch actor is frozen, not promoted. Fresh8 is not accessed.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from numpy.typing import NDArray
from rsi_mjx_clean_touch_control_es import (
    ACTION_LIMIT_RAD,
    COURSES,
    EXAM_FRAME,
    FEATURES,
    FRESH8,
    JOINTS,
    SNAPSHOT,
    _capture,
    _rollout,
)

from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_sensorless_model

SCHEMA = "rosclaw_soccer.rsi.mjx_clean_touch_body_coord_es.v1"
BODY_JOINTS = (6, 7, 9, 13, 14)
FULL_JOINTS = JOINTS + BODY_JOINTS
FULL_FEATURES = 5 + 2 * len(FULL_JOINTS)
BODY_WEIGHTS = len(BODY_JOINTS) * (FULL_FEATURES + 1)


def _load_contact_seed(path: Path) -> tuple[NDArray[np.float32], str]:
    payload: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    commitment = payload.pop("report_hash")
    weights = np.asarray(payload["weights"], dtype=np.float32)
    if (
        commitment != hash_json(payload)
        or payload["promotion_authorized"] is not False
        or payload["best_clean_foot_count"] != len(COURSES)
        or payload["best_nonfoot_count"] != 0
        or weights.shape != (len(JOINTS) * (FEATURES + 1),)
        or not np.isfinite(weights).all()
        or payload["weights_hash"] != hash_bytes(weights.tobytes())
    ):
        raise ValueError("sealed unpromoted clean-foot contact seed required")
    return weights, commitment


def _full_weights(contact: NDArray[np.float32], body: NDArray[np.float32]) -> NDArray[np.float32]:
    if body.shape != (BODY_WEIGHTS,) or not np.isfinite(body).all():
        raise ValueError("finite body-coordination weights required")
    matrix = np.zeros((len(FULL_JOINTS), FULL_FEATURES), dtype=np.float32)
    old = contact[: len(JOINTS) * FEATURES].reshape(len(JOINTS), FEATURES)
    matrix[: len(JOINTS), :10] = old[:, :10]
    matrix[: len(JOINTS), 5 + len(FULL_JOINTS) : 5 + len(FULL_JOINTS) + len(JOINTS)] = old[:, 10:]
    matrix[len(JOINTS) :] = body[: len(BODY_JOINTS) * FULL_FEATURES].reshape(
        len(BODY_JOINTS), FULL_FEATURES
    )
    bias = np.zeros(len(FULL_JOINTS), dtype=np.float32)
    bias[: len(JOINTS)] = contact[len(JOINTS) * FEATURES :]
    bias[len(JOINTS) :] = body[len(BODY_JOINTS) * FULL_FEATURES :]
    return np.concatenate((matrix.ravel(), bias))


def _validate_body_joints(model: mujoco.MjModel) -> None:
    expected = (
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_knee_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    actual = tuple(
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, int(model.actuator_trnid[index, 0]))
        for index in BODY_JOINTS
    )
    if actual != expected:
        raise ValueError(f"body-coordination joint map mismatch: {actual!r}")


def train(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    contact_seed: Path,
    output_dir: Path,
    generations: int,
    population: int,
    seed: int,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_mjx_clean_touch_control_es.py")
    source_hash, helper_hash = hash_bytes(source.read_bytes()), hash_bytes(helper.read_bytes())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        visible is None
        or not visible.isdecimal()
        or not 0 <= int(visible) <= 3
        or jax.devices()[0].platform != "gpu"
        or output_dir.exists()
        or output_dir.resolve().is_relative_to(source.resolve().parents[1])
        or not 1 <= generations <= 20
        or not 4 <= population <= 32
        or population % 2
        or not 0 <= seed < 2**31
    ):
        raise ValueError("bounded SIM_ONLY GPU whole-body training required")
    arrays = _capture(captured, fidelity)
    contact, seed_hash = _load_contact_seed(contact_seed)
    model = build_g1_stadium_sensorless_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    _validate_body_joints(model)
    model.opt.timestep = 0.002
    evaluate, batch = _rollout(model, arrays, FULL_JOINTS)
    output_dir.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "physical_gpu_index": int(visible),
        "compiled_model_hash": compiled_model_hash(model),
        "capture_hash": hash_bytes((captured / "motor-trace.npz").read_bytes()),
        "fidelity_hash": hash_bytes(fidelity.read_bytes()),
        "contact_seed_report_hash": seed_hash,
        "train_courses": [list(row) for row in COURSES],
        "reserved_fresh8": [list(row) for row in FRESH8],
        "snapshot_frame": SNAPSHOT,
        "exam_frame": EXAM_FRAME,
        "joint_indices": FULL_JOINTS,
        "action_limit_rad": ACTION_LIMIT_RAD,
        "generations": generations,
        "population": population,
        "seed": seed,
        "promotion_authorized": False,
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    rng = np.random.default_rng(seed)
    body_mean = np.zeros(BODY_WEIGHTS, dtype=np.float32)
    zero = np.zeros(len(FULL_JOINTS) * (FULL_FEATURES + 1), dtype=np.float32)
    parent_outcome = np.asarray(evaluate(batch, jnp.asarray(zero)))
    seed_outcome = np.asarray(evaluate(batch, jnp.asarray(_full_weights(contact, body_mean))))
    parent_speed = float(np.mean(parent_outcome[:, 5]))
    parent_distance = float(np.mean(parent_outcome[:, 6]))

    def fitness(result: NDArray[np.float32]) -> float:
        speed = float(np.mean(result[:, 5]))
        distance = float(np.mean(result[:, 6]))
        penalty = 12 * max(0.0, speed / (0.85 * parent_speed) - 1)
        penalty += 12 * max(0.0, distance / (0.90 * parent_distance) - 1)
        return float(np.mean(result[:, 0]) + 0.25 * np.min(result[:, 0]) - penalty)

    best_body = body_mean.copy()
    best_fitness = fitness(seed_outcome)
    sigma = 0.25
    history = []
    print(json.dumps({"seed_fitness": best_fitness}), flush=True)
    for generation in range(generations):
        candidates = rng.normal(body_mean, sigma, size=(population, BODY_WEIGHTS)).astype(
            np.float32
        )
        rows = []
        for index, candidate in enumerate(candidates):
            outcome = np.asarray(evaluate(batch, jnp.asarray(_full_weights(contact, candidate))))
            rows.append((fitness(outcome), index, outcome))
        rows.sort(key=lambda row: row[0], reverse=True)
        elites = [candidates[index] for _, index, _ in rows[: max(2, population // 4)]]
        body_mean = np.mean(elites, axis=0).astype(np.float32)
        sigma = max(0.06, sigma * 0.85)
        winner_fitness, winner_index, winner = rows[0]
        if winner_fitness > best_fitness:
            best_body, best_fitness = candidates[winner_index].copy(), winner_fitness
        row = {
            "generation": generation,
            "winner_fitness": winner_fitness,
            "best_fitness": best_fitness,
            "winner_clean_foot_count": int(
                np.count_nonzero((winner[:, 1] > 0.5) & (winner[:, 2] < 0.5))
            ),
            "winner_nonfoot_count": int(np.count_nonzero(winner[:, 2] > 0.5)),
            "winner_mean_half_second_ball_speed_mps": float(np.mean(winner[:, 5])),
            "winner_mean_half_second_distance_m": float(np.mean(winner[:, 6])),
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
        raise RuntimeError("whole-body trainer source changed during GPU physics")
    full = _full_weights(contact, best_body)
    best_outcome = np.asarray(evaluate(batch, jnp.asarray(full)))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": protocol["protocol_hash"],
        "parent_clean_foot_count": int(
            np.count_nonzero((parent_outcome[:, 1] > 0.5) & (parent_outcome[:, 2] < 0.5))
        ),
        "seed_clean_foot_count": int(
            np.count_nonzero((seed_outcome[:, 1] > 0.5) & (seed_outcome[:, 2] < 0.5))
        ),
        "best_clean_foot_count": int(
            np.count_nonzero((best_outcome[:, 1] > 0.5) & (best_outcome[:, 2] < 0.5))
        ),
        "best_nonfoot_count": int(np.count_nonzero(best_outcome[:, 2] > 0.5)),
        "parent_mean_half_second_ball_speed_mps": parent_speed,
        "seed_mean_half_second_ball_speed_mps": float(np.mean(seed_outcome[:, 5])),
        "best_mean_half_second_ball_speed_mps": float(np.mean(best_outcome[:, 5])),
        "parent_mean_half_second_distance_m": parent_distance,
        "seed_mean_half_second_distance_m": float(np.mean(seed_outcome[:, 6])),
        "best_mean_half_second_distance_m": float(np.mean(best_outcome[:, 6])),
        "parent_courses": parent_outcome.tolist(),
        "seed_courses": seed_outcome.tolist(),
        "best_courses": best_outcome.tolist(),
        "best_training_fitness": best_fitness,
        "full_weights": full.tolist(),
        "full_weights_hash": hash_bytes(full.tobytes()),
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
    parser.add_argument("--contact-seed", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--population", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    report = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: report[key] for key in ("best_training_fitness", "report_hash")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
