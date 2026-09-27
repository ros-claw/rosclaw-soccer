"""SIM_ONLY state-feedback leg-residual evolution on physical MJX rollouts.

The frozen SONIC target trace supplies full-body motion; this learner owns
only bounded left/right hip-pitch and knee target residuals. A GPU screening
score never authorizes promotion without disjoint CPU MuJoCo examination.
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

from rosclaw_soccer.providers.g1.sonic_runup import _sonic_control_parameters
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_sensorless_model

SCHEMA = "rosclaw_soccer.rsi.mjx_contact_residual_es.v1"
JOINT_INDICES = (0, 3, 6, 9)
FEATURES = 15
ACTION_LIMIT_RAD = 0.12
TRAIN_COURSES = tuple(itertools.product((2.08, 2.16, 2.24, 2.32), (0.06, 0.14), (-0.5, 0.5)))


def _load_frozen_trace(
    path: Path, *, snapshot_frame: int, frames: int
) -> tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
    with np.load(path, allow_pickle=False) as trace:
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        qvel = np.asarray(trace["qvel"], dtype=np.float64)
        target = np.asarray(trace["target"], dtype=np.float64)
    if (
        qpos.shape != (300, 43)
        or qvel.shape != (300, 41)
        or target.shape != (300, 29)
        or not all(np.isfinite(array).all() for array in (qpos, qvel, target))
        or not 0 <= snapshot_frame < 300 - frames - 1
    ):
        raise ValueError("complete finite qualified 300-frame SONIC trace required")
    return (
        qpos[snapshot_frame],
        qvel[snapshot_frame],
        target[snapshot_frame + 1 : snapshot_frame + frames + 1],
    )


def _make_rollout(
    *,
    model: mujoco.MjModel,
    qpos: NDArray[np.float64],
    qvel: NDArray[np.float64],
    targets: NDArray[np.float64],
) -> Any:
    physics_model = mjx.put_model(model)
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos
    cpu.qvel[:] = qvel
    mujoco.mj_forward(model, cpu)
    initial = mjx.put_data(model, cpu)
    count = len(TRAIN_COURSES)
    batched = jax.tree.map(lambda value: jnp.broadcast_to(value, (count,) + value.shape), initial)
    positions = np.asarray(TRAIN_COURSES, dtype=np.float64)
    ball_x, ball_y, ball_vx = (jnp.asarray(positions[:, column]) for column in range(3))
    batched = batched.replace(
        qpos=batched.qpos.at[:, 36].set(ball_x).at[:, 37].set(ball_y),
        qvel=batched.qvel.at[:, 35].set(ball_vx).at[:, 39].set(ball_vx / 0.11),
    )
    ball_geom, robot_mask, foot_mask = _contact_masks(model)
    robot_flags, foot_flags = jnp.asarray(robot_mask), jnp.asarray(foot_mask)
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    kp_j, kd_j = jnp.asarray(kp), jnp.asarray(kd)
    torque_limit = jnp.asarray(G1_HARD_TORQUE_LIMITS, dtype=jnp.float32)
    target_j = jnp.asarray(targets)
    joint_indices = jnp.asarray(JOINT_INDICES)
    limits = jnp.asarray(model.jnt_range[model.actuator_trnid[list(JOINT_INDICES), 0]])

    def one_episode(state: mjx.Data, weights: jax.Array) -> jax.Array:
        matrix = weights[: 4 * FEATURES].reshape(4, FEATURES)
        bias = weights[4 * FEATURES :]

        def control_step(
            carry: tuple[Any, ...], desired: jax.Array
        ) -> tuple[tuple[Any, ...], None]:
            current, seen_foot, seen_nonfoot, min_height, effort = carry
            ball_dx = current.qpos[36] - current.qpos[0]
            joint_pos = current.qpos[7 + joint_indices]
            joint_vel = current.qvel[6 + joint_indices]
            features = jnp.concatenate(
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
                    joint_pos,
                    joint_vel / 5.0,
                    jnp.asarray((seen_foot, seen_nonfoot), dtype=jnp.float32),
                )
            )
            gate = jnp.where(((ball_dx > 0.05) & (ball_dx < 1.2)) | seen_foot, 1.0, 0.0)
            residual = ACTION_LIMIT_RAD * gate * jnp.tanh(matrix @ features + bias)
            modified = jnp.clip(desired[joint_indices] + residual, limits[:, 0], limits[:, 1])
            proposal = desired.at[joint_indices].set(modified)

            def physics_step(physical: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
                torque = jnp.clip(
                    kp_j * (proposal - physical.qpos[7:36]) - kd_j * physical.qvel[6:35],
                    -torque_limit,
                    torque_limit,
                )
                advanced = mjx.step(physics_model, physical.replace(ctrl=torque))
                contact = advanced.contact
                a, b = contact.geom1, contact.geom2
                other = jnp.where(a == ball_geom, b, a)
                hit = (contact.dist < 0.0) & ((a == ball_geom) | (b == ball_geom))
                robot_hit = hit & robot_flags[jnp.clip(other, 0, model.ngeom - 1)]
                foot_hit = robot_hit & foot_flags[jnp.clip(other, 0, model.ngeom - 1)]
                return advanced, jnp.asarray((jnp.any(foot_hit), jnp.any(robot_hit & ~foot_hit)))

            advanced, hits = jax.lax.scan(physics_step, current, None, length=10)
            ever_foot = seen_foot | jnp.any(hits[:, 0])
            ever_nonfoot = seen_nonfoot | jnp.any(hits[:, 1])
            return (
                advanced,
                ever_foot,
                ever_nonfoot,
                jnp.minimum(min_height, advanced.qpos[2]),
                effort + jnp.mean(jnp.square(residual)),
            ), None

        initial_carry = (
            state,
            jnp.asarray(False),
            jnp.asarray(False),
            state.qpos[2],
            jnp.asarray(0.0, dtype=jnp.float32),
        )
        (final, foot, nonfoot, minimum, effort), _ = jax.lax.scan(
            control_step, initial_carry, target_j
        )
        clean = foot & ~nonfoot
        safe = minimum >= 0.65
        ball_vx_gain = jnp.clip(final.qvel[35] - state.qvel[35], 0.0, 5.0)
        forward_gain = jnp.clip(final.qpos[36] - state.qpos[36], 0.0, 3.0)
        score = (
            jnp.where(clean, 20.0 + 2.0 * ball_vx_gain + forward_gain, 0.0)
            - 20.0 * nonfoot.astype(jnp.float32)
            - 2.0 * (~foot & ~nonfoot).astype(jnp.float32)
            - 100.0 * (~safe).astype(jnp.float32)
            - 0.2 * effort
        )
        score = jnp.nan_to_num(score, nan=-100.0, posinf=-100.0, neginf=-100.0)
        return jnp.asarray(
            (score, foot.astype(jnp.float32), nonfoot.astype(jnp.float32), minimum, final.qvel[35])
        )

    return (
        jax.jit(jax.vmap(one_episode, in_axes=(0, None)))(batched, jnp.zeros(64)).shape,
        jax.jit(jax.vmap(one_episode, in_axes=(0, None))),
        batched,
    )


def train(
    *,
    asset_root: Path,
    parent_trace: Path,
    output_dir: Path,
    snapshot_frame: int,
    control_frames: int,
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
        or population % 2 != 0
        or not 20 <= control_frames <= 50
        or not 0 <= seed < 2**31
    ):
        raise ValueError("bounded SIM_ONLY GPU training inputs and new external output required")
    qpos, qvel, targets = _load_frozen_trace(
        parent_trace, snapshot_frame=snapshot_frame, frames=control_frames
    )
    goal = G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    model = build_g1_stadium_sensorless_model(asset_root, goal)
    model.opt.timestep = 0.002
    shape, evaluate, batch = _make_rollout(model=model, qpos=qpos, qvel=qvel, targets=targets)
    if shape != (16, 5):
        raise AssertionError("training rollout did not advance all physical courses")
    output_dir.mkdir(parents=True)
    protocol: dict[str, Any] = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "physical_gpu_index": int(visible),
        "source_hash": source_hash,
        "helper_hash": helper_hash,
        "parent_trace_hash": hash_bytes(parent_trace.read_bytes()),
        "scene_hash": hash_bytes(
            (asset_root / "g1_description" / "scene_with_ball.xml").read_bytes()
        ),
        "train_courses": [list(course) for course in TRAIN_COURSES],
        "snapshot_frame": snapshot_frame,
        "control_frames": control_frames,
        "generations": generations,
        "population": population,
        "seed": seed,
        "action_joint_indices": JOINT_INDICES,
        "action_limit_rad": ACTION_LIMIT_RAD,
        "activation_authorized": False,
    }
    protocol["protocol_hash"] = hash_json(protocol)
    (output_dir / "protocol.json").write_text(
        json.dumps(protocol, sort_keys=True, indent=2, allow_nan=False) + "\n"
    )
    rng = np.random.default_rng(seed)
    mean = np.zeros(64, dtype=np.float32)
    sigma = 0.35
    baseline = np.asarray(evaluate(batch, jnp.asarray(mean)))
    baseline_fitness = float(np.mean(baseline[:, 0]) + 0.25 * np.min(baseline[:, 0]))
    best_params, best_fitness = mean.copy(), baseline_fitness
    history: list[dict[str, Any]] = []
    print(json.dumps({"baseline_fitness": baseline_fitness}), flush=True)
    for generation in range(generations):
        candidates = rng.normal(mean, sigma, size=(population, 64)).astype(np.float32)
        rows = []
        for index, candidate in enumerate(candidates):
            outcomes = np.asarray(evaluate(batch, jnp.asarray(candidate)))
            fitness = float(np.mean(outcomes[:, 0]) + 0.25 * np.min(outcomes[:, 0]))
            rows.append((fitness, index, outcomes))
        rows.sort(key=lambda row: row[0], reverse=True)
        elites = [candidates[index] for _, index, _ in rows[: max(2, population // 4)]]
        mean = np.mean(elites, axis=0).astype(np.float32)
        sigma = max(0.08, sigma * 0.85)
        winner_fitness, winner_index, winner_outcomes = rows[0]
        if winner_fitness > best_fitness:
            best_fitness = winner_fitness
            best_params = candidates[winner_index].copy()
        row = {
            "generation": generation,
            "winner_fitness": winner_fitness,
            "best_fitness": best_fitness,
            "winner_clean_foot_count": int(
                np.count_nonzero((winner_outcomes[:, 1] > 0.5) & (winner_outcomes[:, 2] < 0.5))
            ),
            "winner_nonfoot_count": int(np.count_nonzero(winner_outcomes[:, 2] > 0.5)),
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
        raise RuntimeError("training source changed during physical rollouts")
    best_outcomes = np.asarray(evaluate(batch, jnp.asarray(best_params)))
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
            np.count_nonzero((best_outcomes[:, 1] > 0.5) & (best_outcomes[:, 2] < 0.5))
        ),
        "baseline_nonfoot_count": int(np.count_nonzero(baseline[:, 2] > 0.5)),
        "best_training_nonfoot_count": int(np.count_nonzero(best_outcomes[:, 2] > 0.5)),
        "parameters": best_params.tolist(),
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
    parser.add_argument("--parent-trace", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--snapshot-frame", type=int, default=80)
    parser.add_argument("--control-frames", type=int, default=30)
    parser.add_argument("--generations", type=int, default=4)
    parser.add_argument("--population", type=int, default=8)
    parser.add_argument("--seed", type=int, default=92801)
    args = parser.parse_args()
    result = train(**vars(args))
    print(json.dumps({key: value for key, value in result.items() if key != "parameters"}))


if __name__ == "__main__":
    main()
