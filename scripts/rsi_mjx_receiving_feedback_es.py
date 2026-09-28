"""SIM_ONLY MJX evolution of a bounded receiver-specific feedback leg actor."""

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
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_sensorless_model

SCHEMA = "rosclaw_soccer.rsi.mjx_receiving_feedback_es.v1"
COURSES = tuple(itertools.product((2.96, 3.02), (1.34, 1.42), (-1.15, -0.90)))
JOINT_INDICES = (0, 3, 6, 9)
FEATURES = 15
ACTION_LIMIT_RAD = 0.12
SNAPSHOT_FRAME = 20
CONTROL_FRAMES = 50


def _load_capture(captured: Path, fidelity: Path) -> dict[str, NDArray[np.float64]]:
    trace_path = captured / "motor-trace.npz"
    report: dict[str, Any] = json.loads((captured / "report.json").read_text(encoding="utf-8"))
    report_hash = report.pop("report_hash")
    fidelity_report: dict[str, Any] = json.loads(fidelity.read_text(encoding="utf-8"))
    fidelity_hash = fidelity_report.pop("report_hash")
    if (
        report_hash != hash_json(report)
        or report["trace_hash"] != hash_bytes(trace_path.read_bytes())
        or fidelity_hash != hash_json(fidelity_report)
        or not fidelity_report["fidelity_gate_passed"]
        or fidelity_report["capture_hash"] != report_hash
        or fidelity_report["snapshot_frame"] != SNAPSHOT_FRAME
    ):
        raise ValueError("qualified exact receiving motor capture required")
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
        raise ValueError("finite complete 300-frame receiving target trace required")
    return arrays


def _rollout(model: mujoco.MjModel, arrays: dict[str, NDArray[np.float64]]) -> tuple[Any, Any]:
    physics = mjx.put_model(model)
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = arrays["sonic_recorded_qpos"][SNAPSHOT_FRAME]
    cpu.qvel[:] = arrays["sonic_recorded_qvel"][SNAPSHOT_FRAME]
    mujoco.mj_forward(model, cpu)
    initial = mjx.put_data(model, cpu)
    n = len(COURSES)
    batch = jax.tree.map(lambda value: jnp.broadcast_to(value, (n,) + value.shape), initial)
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
    index = jnp.asarray(JOINT_INDICES)
    limits = jnp.asarray(model.jnt_range[model.actuator_trnid[list(JOINT_INDICES), 0]])
    targets = jnp.asarray(
        arrays["sonic_recorded_target"][SNAPSHOT_FRAME + 1 : SNAPSHOT_FRAME + 1 + CONTROL_FRAMES]
    )
    kp = jnp.asarray(
        arrays["sonic_recorded_kp"][SNAPSHOT_FRAME + 1 : SNAPSHOT_FRAME + 1 + CONTROL_FRAMES]
    )
    kd = jnp.asarray(
        arrays["sonic_recorded_kd"][SNAPSHOT_FRAME + 1 : SNAPSHOT_FRAME + 1 + CONTROL_FRAMES]
    )
    torque_limit = jnp.asarray(G1_HARD_TORQUE_LIMITS, dtype=jnp.float32)

    def episode(state: mjx.Data, weights: jax.Array) -> jax.Array:
        matrix, bias = weights[: 4 * FEATURES].reshape(4, FEATURES), weights[4 * FEATURES :]

        def control_step(
            carry: tuple[Any, ...], desired: tuple[jax.Array, ...]
        ) -> tuple[Any, None]:
            current, seen_foot, seen_nonfoot, minimum, maximum_tilt, effort = carry
            target, p_gain, d_gain = desired
            ball_dx = current.qpos[36] - current.qpos[0]
            feature = jnp.concatenate(
                (
                    jnp.asarray(
                        (
                            ball_dx,
                            current.qpos[37] - current.qpos[1],
                            current.qvel[35] - current.qvel[0],
                            current.qvel[0],
                            current.qpos[2] - 0.75,
                        )
                    ),
                    current.qpos[7 + index],
                    current.qvel[6 + index] / 5.0,
                    jnp.asarray((seen_foot, seen_nonfoot), dtype=jnp.float32),
                )
            )
            gate = jnp.where(((ball_dx > 0.05) & (ball_dx < 1.2)) | seen_foot, 1.0, 0.0)
            residual = ACTION_LIMIT_RAD * gate * jnp.tanh(matrix @ feature + bias)
            modified = jnp.clip(target[index] + residual, limits[:, 0], limits[:, 1])
            proposal = target.at[index].set(modified)

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
            return (
                advanced,
                seen_foot | jnp.any(hits[:, 0]),
                seen_nonfoot | jnp.any(hits[:, 1]),
                jnp.minimum(minimum, advanced.qpos[2]),
                jnp.maximum(maximum_tilt, tilt),
                effort + jnp.mean(jnp.square(residual)),
            ), None

        initial_carry = (
            state,
            jnp.asarray(False),
            jnp.asarray(False),
            state.qpos[2],
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        (final, foot, nonfoot, minimum, tilt, effort), _ = jax.lax.scan(
            control_step, initial_carry, (targets, kp, kd)
        )
        clean = foot & ~nonfoot
        safe = (minimum >= 0.65) & (tilt < 0.30)
        retained_speed = jnp.abs(final.qvel[35])
        score = (
            22.0 * clean.astype(jnp.float32)
            + 3.0 * foot.astype(jnp.float32)
            - 22.0 * nonfoot.astype(jnp.float32)
            - 6.0 * (~foot & ~nonfoot).astype(jnp.float32)
            - 4.0 * foot.astype(jnp.float32) * jnp.clip(retained_speed, 0.0, 3.0)
            - 100.0 * (~safe).astype(jnp.float32)
            - 2.0 * tilt
            - 0.2 * effort
        )
        score = jnp.nan_to_num(score, nan=-100.0, posinf=-100.0, neginf=-100.0)
        return jnp.asarray(
            (score, foot.astype(jnp.float32), nonfoot.astype(jnp.float32), minimum, tilt)
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
        raise ValueError("bounded SIM_ONLY GPU receiving training required")
    arrays = _load_capture(captured, fidelity)
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
        "train_courses": [list(row) for row in COURSES],
        "snapshot_frame": SNAPSHOT_FRAME,
        "control_frames": CONTROL_FRAMES,
        "generations": generations,
        "population": population,
        "seed": seed,
        "action_joint_indices": JOINT_INDICES,
        "action_limit_rad": ACTION_LIMIT_RAD,
        "promotion_authorized": False,
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    rng = np.random.default_rng(seed)
    mean = np.zeros(64, dtype=np.float32)
    sigma = 0.35
    baseline = np.asarray(evaluate(batch, jnp.asarray(mean)))

    def fitness(result: NDArray[np.float32]) -> float:
        return float(np.mean(result[:, 0]) + 0.25 * np.min(result[:, 0]))

    baseline_fitness = fitness(baseline)
    best, best_fitness = mean.copy(), baseline_fitness
    history = []
    print(json.dumps({"baseline_fitness": baseline_fitness}), flush=True)
    for generation in range(generations):
        candidates = rng.normal(mean, sigma, size=(population, 64)).astype(np.float32)
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
        raise RuntimeError("receiving training source changed during GPU physics")
    best_outcome = np.asarray(evaluate(batch, jnp.asarray(best)))
    result: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "protocol_hash": protocol["protocol_hash"],
        "baseline_fitness": baseline_fitness,
        "best_training_fitness": best_fitness,
        "baseline_clean_foot_count": int(
            np.count_nonzero((baseline[:, 1] > 0.5) & (baseline[:, 2] < 0.5))
        ),
        "best_training_clean_foot_count": int(
            np.count_nonzero((best_outcome[:, 1] > 0.5) & (best_outcome[:, 2] < 0.5))
        ),
        "baseline_nonfoot_count": int(np.count_nonzero(baseline[:, 2] > 0.5)),
        "best_training_nonfoot_count": int(np.count_nonzero(best_outcome[:, 2] > 0.5)),
        "parameters": best.tolist(),
        "progress": history,
        "promotion_authorized": False,
        "cpu_holdout_evaluated": False,
    }
    result["result_hash"] = hash_json(result)
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
    parser.add_argument("--generations", type=int, default=8)
    parser.add_argument("--population", type=int, default=16)
    parser.add_argument("--seed", type=int, default=92811)
    result = train(**vars(parser.parse_args()))
    print(json.dumps({key: value for key, value in result.items() if key != "parameters"}))


if __name__ == "__main__":
    main()
