"""SIM_ONLY MJX evolution of bounded feedback for clean-foot receiving control.

Train only on a qualified exact SONIC trace and a disjoint declared course grid.
The candidate never receives hardware authority or automatic promotion.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from numpy.typing import NDArray
from rsi_mjx_contact_replay_smoke import _contact_masks

from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.sim.physical_checkpoint import compiled_model_hash
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_sensorless_model

SCHEMA = "rosclaw_soccer.rsi.mjx_clean_touch_control_es.v1"
COURSES = tuple(itertools.product((2.52, 2.55), (1.36, 1.40), (-0.82, -0.90)))
FRESH8 = tuple(itertools.product((2.525, 2.545), (1.37, 1.39), (-0.84, -0.88)))
JOINTS = (0, 1, 3, 4, 5)
FEATURES = 15
WEIGHTS = len(JOINTS) * (FEATURES + 1)
ACTION_LIMIT_RAD = 0.12
SNAPSHOT = 45
CONTROL_FRAMES = 55
EXAM_FRAME = 86


def _capture(captured: Path, fidelity: Path) -> dict[str, NDArray[np.float64]]:
    trace_path = captured / "motor-trace.npz"
    report: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    commitment = report.pop("report_hash")
    qualified: dict[str, Any] = json.loads(fidelity.read_text(encoding="utf-8"))
    qualified_hash = qualified.pop("report_hash")
    if (
        commitment != hash_json(report)
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or qualified_hash != hash_json(qualified)
        or not qualified["fidelity_gate_passed"]
        or qualified["capture_hash"] != commitment
        or qualified["snapshot_frame"] != SNAPSHOT
    ):
        raise ValueError("qualified exact clean-touch capture required")
    with np.load(trace_path, allow_pickle=False) as trace:
        arrays = {
            key: np.asarray(trace[key], dtype=np.float64)
            for key in (
                "sonic_recorded_qpos",
                "sonic_recorded_qvel",
                "sonic_recorded_target",
                "sonic_recorded_kp",
                "sonic_recorded_kd",
            )
        }
    if (
        arrays["sonic_recorded_qpos"].shape != (300, 43)
        or arrays["sonic_recorded_qvel"].shape != (300, 41)
        or any(arrays[key].shape != (300, 29) for key in tuple(arrays)[2:])
        or any(not np.isfinite(value).all() for value in arrays.values())
    ):
        raise ValueError("complete finite measured SONIC trace required")
    return arrays


def _rollout(model: mujoco.MjModel, arrays: dict[str, NDArray[np.float64]]) -> tuple[Any, Any]:
    physics = mjx.put_model(model)
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = arrays["sonic_recorded_qpos"][SNAPSHOT]
    cpu.qvel[:] = arrays["sonic_recorded_qvel"][SNAPSHOT]
    mujoco.mj_forward(model, cpu)
    initial = mjx.put_data(model, cpu)
    batch = jax.tree.map(
        lambda value: jnp.broadcast_to(value, (len(COURSES),) + value.shape), initial
    )
    course = np.asarray(COURSES, dtype=np.float64)
    base_vx = float(cpu.qvel[35])
    batch = batch.replace(
        qpos=batch.qpos.at[:, 36]
        .set(jnp.asarray(course[:, 0]))
        .at[:, 37]
        .set(jnp.asarray(course[:, 1])),
        qvel=batch.qvel.at[:, 35]
        .set(jnp.asarray(course[:, 2]))
        .at[:, 39]
        .set(cpu.qvel[39] + jnp.asarray(course[:, 2] - base_vx) / 0.115),
    )
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    robot_flags, foot_flags = jnp.asarray(robot_mask), jnp.asarray(foot_mask)
    index = jnp.asarray(JOINTS)
    joint_limits = jnp.asarray(model.jnt_range[model.actuator_trnid[list(JOINTS), 0]])
    targets = jnp.asarray(
        arrays["sonic_recorded_target"][SNAPSHOT + 1 : SNAPSHOT + 1 + CONTROL_FRAMES]
    )
    kp = jnp.asarray(arrays["sonic_recorded_kp"][SNAPSHOT + 1 : SNAPSHOT + 1 + CONTROL_FRAMES])
    kd = jnp.asarray(arrays["sonic_recorded_kd"][SNAPSHOT + 1 : SNAPSHOT + 1 + CONTROL_FRAMES])
    frames = jnp.arange(SNAPSHOT + 1, SNAPSHOT + 1 + CONTROL_FRAMES)
    torque_limit = jnp.asarray(G1_HARD_TORQUE_LIMITS, dtype=jnp.float32)

    def episode(state: mjx.Data, weights: jax.Array) -> jax.Array:
        matrix = weights[: len(JOINTS) * FEATURES].reshape(len(JOINTS), FEATURES)
        bias = weights[len(JOINTS) * FEATURES :]

        def control_step(
            carry: tuple[Any, ...], desired: tuple[jax.Array, ...]
        ) -> tuple[tuple[Any, ...], None]:
            (
                current,
                foot_seen,
                nonfoot_seen,
                minimum,
                maximum_tilt,
                effort,
                exam_speed,
                exam_distance,
            ) = carry
            frame, target, p_gain, d_gain = desired
            dx = current.qpos[36] - current.qpos[0]
            feature = jnp.concatenate(
                (
                    jnp.asarray(
                        (
                            dx,
                            current.qpos[37] - current.qpos[1],
                            current.qvel[35] - current.qvel[0],
                            current.qvel[36] - current.qvel[1],
                            current.qpos[2] - 0.75,
                        )
                    ),
                    current.qpos[7 + index],
                    current.qvel[6 + index] / 5.0,
                )
            )
            gate = jnp.where((dx > 0.12) & (dx < 0.85), 1.0, 0.0)
            residual = ACTION_LIMIT_RAD * gate * jnp.tanh(matrix @ feature + bias)
            proposal = target.at[index].set(
                jnp.clip(target[index] + residual, joint_limits[:, 0], joint_limits[:, 1])
            )

            def physics_step(data: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
                torque = jnp.clip(
                    p_gain * (proposal - data.qpos[7:36]) - d_gain * data.qvel[6:35],
                    -torque_limit,
                    torque_limit,
                )
                advanced = mjx.step(physics, data.replace(ctrl=torque))
                contact = advanced.contact
                a, b = contact.geom1, contact.geom2
                other = jnp.where(a == ball_geom, b, a)
                hit = (contact.dist < 0.0) & ((a == ball_geom) | (b == ball_geom))
                robot_hit = hit & robot_flags[jnp.clip(other, 0, model.ngeom - 1)]
                foot_hit = robot_hit & foot_flags[jnp.clip(other, 0, model.ngeom - 1)]
                return advanced, jnp.asarray((jnp.any(foot_hit), jnp.any(robot_hit & ~foot_hit)))

            advanced, hits = jax.lax.scan(physics_step, current, None, length=10)
            quat = advanced.qpos[3:7]
            tilt = jnp.arccos(jnp.clip(1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2), -1.0, 1.0))
            speed = jnp.linalg.norm(advanced.qvel[35:37])
            distance = jnp.linalg.norm(advanced.qpos[36:38] - advanced.qpos[:2])
            return (
                advanced,
                foot_seen | jnp.any(hits[:, 0]),
                nonfoot_seen | jnp.any(hits[:, 1]),
                jnp.minimum(minimum, advanced.qpos[2]),
                jnp.maximum(maximum_tilt, tilt),
                effort + jnp.mean(jnp.square(residual)),
                jnp.where(frame == EXAM_FRAME, speed, exam_speed),
                jnp.where(frame == EXAM_FRAME, distance, exam_distance),
            ), None

        initial_carry = (
            state,
            jnp.asarray(False),
            jnp.asarray(False),
            state.qpos[2],
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        (final, foot, nonfoot, minimum, tilt, effort, speed, distance), _ = jax.lax.scan(
            control_step, initial_carry, (frames, targets, kp, kd)
        )
        clean = foot & ~nonfoot
        safe = (minimum >= 0.65) & (tilt < 0.30)
        score = (
            25.0 * clean.astype(jnp.float32)
            + 3.0 * foot.astype(jnp.float32)
            - 25.0 * nonfoot.astype(jnp.float32)
            - 8.0 * (~foot & ~nonfoot).astype(jnp.float32)
            - 8.0 * speed
            - 5.0 * distance
            - 100.0 * (~safe).astype(jnp.float32)
            - 2.0 * tilt
            - 0.2 * effort
        )
        score = jnp.nan_to_num(score, nan=-100.0, posinf=-100.0, neginf=-100.0)
        return jnp.asarray(
            (
                score,
                foot.astype(jnp.float32),
                nonfoot.astype(jnp.float32),
                minimum,
                tilt,
                speed,
                distance,
            )
        )

    return jax.jit(jax.vmap(episode, in_axes=(0, None))), batch


def train(
    *,
    asset_root: Path,
    captured: Path,
    fidelity: Path,
    output_dir: Path,
    generations: int,
    population: int,
    seed: int,
) -> dict[str, Any]:
    source = Path(__file__)
    helper = source.with_name("rsi_mjx_contact_replay_smoke.py")
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
        raise ValueError("bounded SIM_ONLY GPU clean-touch training required")
    arrays = _capture(captured, fidelity)
    model = build_g1_stadium_sensorless_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.115, ball_mass_kg=0.41)
    )
    model.opt.timestep = 0.002
    evaluate, batch = _rollout(model, arrays)
    output_dir.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "physical_gpu_index": int(visible),
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "capture_hash": hash_bytes((captured / "motor-trace.npz").read_bytes()),
        "fidelity_hash": hash_bytes(fidelity.read_bytes()),
        "compiled_model_hash": compiled_model_hash(model),
        "train_courses": [list(row) for row in COURSES],
        "reserved_fresh8": [list(row) for row in FRESH8],
        "snapshot_frame": SNAPSHOT,
        "exam_frame": EXAM_FRAME,
        "control_frames": CONTROL_FRAMES,
        "generations": generations,
        "population": population,
        "seed": seed,
        "action_joint_indices": JOINTS,
        "action_limit_rad": ACTION_LIMIT_RAD,
        "promotion_authorized": False,
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    rng = np.random.default_rng(seed)
    mean = np.zeros(WEIGHTS, dtype=np.float32)
    sigma = 0.35
    baseline = np.asarray(evaluate(batch, jnp.asarray(mean)))

    def fitness(result: NDArray[np.float32]) -> float:
        return float(np.mean(result[:, 0]) + 0.25 * np.min(result[:, 0]))

    baseline_fitness = fitness(baseline)
    best, best_fitness = mean.copy(), baseline_fitness
    history = []
    print(json.dumps({"baseline_fitness": baseline_fitness}), flush=True)
    for generation in range(generations):
        candidates = rng.normal(mean, sigma, size=(population, WEIGHTS)).astype(np.float32)
        rows = []
        for index, candidate in enumerate(candidates):
            outcome = np.asarray(evaluate(batch, jnp.asarray(candidate)))
            rows.append((fitness(outcome), index, outcome))
        rows.sort(key=lambda row: row[0], reverse=True)
        elites = [candidates[index] for _, index, _ in rows[: max(2, population // 4)]]
        mean = np.mean(elites, axis=0).astype(np.float32)
        sigma = max(0.08, sigma * 0.85)
        winner_fitness, winner_index, winner = rows[0]
        if winner_fitness > best_fitness:
            best, best_fitness = candidates[winner_index].copy(), winner_fitness
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
        raise RuntimeError("clean-touch trainer source changed during GPU physics")
    outcome = np.asarray(evaluate(batch, jnp.asarray(best)))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": protocol["protocol_hash"],
        "baseline_fitness": baseline_fitness,
        "best_training_fitness": best_fitness,
        "baseline_clean_foot_count": int(
            np.count_nonzero((baseline[:, 1] > 0.5) & (baseline[:, 2] < 0.5))
        ),
        "best_clean_foot_count": int(
            np.count_nonzero((outcome[:, 1] > 0.5) & (outcome[:, 2] < 0.5))
        ),
        "baseline_nonfoot_count": int(np.count_nonzero(baseline[:, 2] > 0.5)),
        "best_nonfoot_count": int(np.count_nonzero(outcome[:, 2] > 0.5)),
        "baseline_mean_half_second_ball_speed_mps": float(np.mean(baseline[:, 5])),
        "best_mean_half_second_ball_speed_mps": float(np.mean(outcome[:, 5])),
        "baseline_mean_half_second_distance_m": float(np.mean(baseline[:, 6])),
        "best_mean_half_second_distance_m": float(np.mean(outcome[:, 6])),
        "baseline_courses": baseline.tolist(),
        "best_courses": outcome.tolist(),
        "weights": best.tolist(),
        "weights_hash": hash_bytes(best.tobytes()),
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
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--generations", required=True, type=int)
    parser.add_argument("--population", required=True, type=int)
    parser.add_argument("--seed", required=True, type=int)
    result = train(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: result[key] for key in ("best_training_fitness", "report_hash")}, sort_keys=True
        )
    )


if __name__ == "__main__":
    main()
