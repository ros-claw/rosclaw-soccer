"""SIM_ONLY batched MJX replay of frozen SONIC targets through ball contact.

This is a CPU-vs-MJX label agreement diagnostic, not actor training or a
promotion exam. It deliberately retains failures and raw differences.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx
from numpy.typing import NDArray

from rosclaw_soccer.providers.g1.sonic_runup import _sonic_control_parameters
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_sensorless_model


def _contact_masks(model: mujoco.MjModel) -> tuple[int, NDArray[np.bool_], NDArray[np.bool_]]:
    ball = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "ball_geom")
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if ball < 0 or pelvis < 0:
        raise ValueError("qualified ball and pelvis geoms required")
    robot = np.zeros(model.ngeom, dtype=bool)
    foot = np.zeros(model.ngeom, dtype=bool)
    for geom in range(model.ngeom):
        body = int(model.geom_bodyid[geom])
        while body > 0:
            if body == pelvis:
                robot[geom] = True
                break
            body = int(model.body_parentid[body])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom) or ""
        foot[geom] = robot[geom] and ("foot" in name or "ankle" in name)
    if not np.any(foot):
        raise ValueError("qualified foot contact geometries required")
    return ball, robot, foot


def run(
    *,
    asset_root: Path,
    parent_trace: Path,
    output: Path,
    snapshot_frame: int,
    control_frames: int,
    envs: int,
    course_grid: bool = False,
) -> dict[str, object]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if (
        visible is None
        or not visible.isdecimal()
        or not 0 <= int(visible) <= 3
        or not 1 <= envs <= 16
        or (course_grid and envs != 16)
        or not 1 <= control_frames <= 50
        or not 0 <= snapshot_frame <= 200
        or output.exists()
        or not output.parent.is_dir()
        or output.resolve().is_relative_to(source.resolve().parents[1])
        or not parent_trace.is_file()
    ):
        raise ValueError("bounded GPU replay with external new output and frozen trace required")
    if jax.devices()[0].platform != "gpu":
        raise RuntimeError("GPU MJX physics required")
    with np.load(parent_trace, allow_pickle=False) as trace:
        qpos = np.asarray(trace["qpos"], dtype=np.float64)
        qvel = np.asarray(trace["qvel"], dtype=np.float64)
        target = np.asarray(trace["target"], dtype=np.float64)
    if (
        qpos.shape != (300, 43)
        or qvel.shape != (300, 41)
        or target.shape != (300, 29)
        or not all(np.isfinite(array).all() for array in (qpos, qvel, target))
        or snapshot_frame + control_frames >= 300
    ):
        raise ValueError("complete finite 300-frame frozen SONIC trajectory required")
    model = build_g1_stadium_sensorless_model(
        asset_root, G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    )
    model.opt.timestep = 0.002
    ball_id, robot_mask, foot_mask = _contact_masks(model)
    kp, kd, _ = _sonic_control_parameters(1.0, (1.0,) * 29)
    limit = np.asarray(G1_HARD_TORQUE_LIMITS, dtype=np.float64)
    physics_model = mjx.put_model(model)
    cpu = mujoco.MjData(model)
    cpu.qpos[:] = qpos[snapshot_frame]
    cpu.qvel[:] = qvel[snapshot_frame]
    mujoco.mj_forward(model, cpu)
    first_x = float(cpu.qpos[36])
    courses = (
        tuple(itertools.product((2.08, 2.16, 2.24, 2.32), (0.06, 0.14), (-0.5, 0.5)))
        if course_grid
        else tuple(
            (x, float(cpu.qpos[37]), float(cpu.qvel[35]))
            for x in np.linspace(first_x - 0.04, first_x + 0.04, envs)
        )
    )
    ball_x, ball_y, ball_vx = (np.asarray(column) for column in zip(*courses, strict=True))
    initial = mjx.put_data(model, cpu)
    batched = jax.tree.map(lambda value: jnp.broadcast_to(value, (envs,) + value.shape), initial)
    batched = batched.replace(
        qpos=batched.qpos.at[:, 36].set(jnp.asarray(ball_x)).at[:, 37].set(jnp.asarray(ball_y)),
        qvel=batched.qvel.at[:, 35]
        .set(jnp.asarray(ball_vx))
        .at[:, 39]
        .set(jnp.asarray(ball_vx / 0.11)),
    )
    robot_flags = jnp.asarray(robot_mask)
    foot_flags = jnp.asarray(foot_mask)
    kp_j, kd_j, limit_j = (jnp.asarray(value) for value in (kp, kd, limit))
    target_j = jnp.asarray(target[snapshot_frame + 1 : snapshot_frame + control_frames + 1])

    def one_lane(state: mjx.Data) -> tuple[jax.Array, jax.Array, jax.Array]:
        def control_step(
            current: mjx.Data, desired: jax.Array
        ) -> tuple[mjx.Data, tuple[jax.Array, jax.Array]]:
            def physics_step(physical: mjx.Data, _: None) -> tuple[mjx.Data, jax.Array]:
                torque = jnp.clip(
                    kp_j * (desired - physical.qpos[7:36]) - kd_j * physical.qvel[6:35],
                    -limit_j,
                    limit_j,
                )
                advanced = mjx.step(physics_model, physical.replace(ctrl=torque))
                contact = advanced.contact
                a, b = contact.geom1, contact.geom2
                other = jnp.where(a == ball_id, b, a)
                hit = (contact.dist < 0.0) & ((a == ball_id) | (b == ball_id))
                robot_hit = hit & robot_flags[jnp.clip(other, 0, model.ngeom - 1)]
                foot_hit = robot_hit & foot_flags[jnp.clip(other, 0, model.ngeom - 1)]
                return advanced, jnp.asarray((jnp.any(foot_hit), jnp.any(robot_hit & ~foot_hit)))

            updated, contacts = jax.lax.scan(physics_step, current, None, length=10)
            return updated, (updated.qpos, jnp.any(contacts, axis=0))

        _, (poses, impacts) = jax.lax.scan(control_step, state, target_j)
        return poses, impacts, jnp.asarray(state.time)

    poses, impacts, initial_time = jax.jit(jax.vmap(one_lane))(batched)
    gpu_qpos = np.asarray(poses)
    gpu_contact = np.asarray(impacts)
    if (
        gpu_qpos.shape != (envs, control_frames, 43)
        or gpu_contact.shape != (envs, control_frames, 2)
        or not np.isfinite(gpu_qpos).all()
        or not np.isfinite(np.asarray(initial_time)).all()
    ):
        raise ValueError("invalid vectorized MJX contact trajectory")
    cpu_poses_rows = []
    cpu_impacts_rows = []
    for initial_ball_x, initial_ball_y, initial_ball_vx in courses:
        lane = mujoco.MjData(model)
        lane.qpos[:] = qpos[snapshot_frame]
        lane.qvel[:] = qvel[snapshot_frame]
        lane.qpos[36] = initial_ball_x
        lane.qpos[37] = initial_ball_y
        lane.qvel[35] = initial_ball_vx
        lane.qvel[39] = initial_ball_vx / 0.11
        mujoco.mj_forward(model, lane)
        lane_poses = []
        lane_impacts = []
        for desired in target[snapshot_frame + 1 : snapshot_frame + control_frames + 1]:
            episode_contact = np.zeros(2, dtype=bool)
            for _ in range(10):
                lane.ctrl[:] = np.clip(
                    kp * (desired - lane.qpos[7:36]) - kd * lane.qvel[6:35], -limit, limit
                )
                mujoco.mj_step(model, lane)
                for index in range(lane.ncon):
                    contact = lane.contact[index]
                    a, b = int(contact.geom1), int(contact.geom2)
                    if a != ball_id and b != ball_id:
                        continue
                    other = b if a == ball_id else a
                    if robot_mask[other]:
                        episode_contact[0 if foot_mask[other] else 1] = True
            lane_poses.append(lane.qpos.copy())
            lane_impacts.append(episode_contact)
        cpu_poses_rows.append(lane_poses)
        cpu_impacts_rows.append(lane_impacts)
    cpu_poses = np.asarray(cpu_poses_rows)
    cpu_impacts = np.asarray(cpu_impacts_rows)
    true_positive = np.count_nonzero(cpu_impacts & gpu_contact, axis=1)
    cpu_positive = np.count_nonzero(cpu_impacts, axis=1)
    gpu_positive = np.count_nonzero(gpu_contact, axis=1)
    recall = np.where(cpu_positive > 0, true_positive / np.maximum(cpu_positive, 1), 1.0)
    precision = np.where(gpu_positive > 0, true_positive / np.maximum(gpu_positive, 1), 1.0)
    label_agreement = bool(
        np.all(recall >= 0.9)
        and np.all(precision >= 0.9)
        and np.array_equal(cpu_positive > 0, gpu_positive > 0)
    )
    episode_class_agreement = np.all((cpu_positive > 0) == (gpu_positive > 0), axis=1)
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("contact replay source changed during physics")
    report: dict[str, object] = {
        "schema": "rosclaw_soccer.rsi.mjx_contact_replay_smoke.v3",
        "activation_ceiling": "SIM_ONLY",
        "physical_gpu_index": int(visible),
        "envs": envs,
        "snapshot_frame": snapshot_frame,
        "control_frames": control_frames,
        "ball_initial_x_m": ball_x.tolist(),
        "ball_initial_y_m": ball_y.tolist(),
        "ball_initial_vx_m_s": ball_vx.tolist(),
        "course_grid": course_grid,
        "cpu_foot_contact_frames": cpu_positive[:, 0].tolist(),
        "cpu_nonfoot_contact_frames": cpu_positive[:, 1].tolist(),
        "mjx_foot_contact_frames": np.count_nonzero(gpu_contact[:, :, 0], axis=1).tolist(),
        "mjx_nonfoot_contact_frames": np.count_nonzero(gpu_contact[:, :, 1], axis=1).tolist(),
        "cpu_contact_frames": [
            {
                "foot": np.flatnonzero(cpu_impacts[index, :, 0]).tolist(),
                "nonfoot": np.flatnonzero(cpu_impacts[index, :, 1]).tolist(),
            }
            for index in range(envs)
        ],
        "mjx_contact_frames": [
            {
                "foot": np.flatnonzero(gpu_contact[index, :, 0]).tolist(),
                "nonfoot": np.flatnonzero(gpu_contact[index, :, 1]).tolist(),
            }
            for index in range(envs)
        ],
        "contact_label_recall": recall.tolist(),
        "contact_label_precision": precision.tolist(),
        "contact_label_agreement_passed": label_agreement,
        "episode_contact_class_agreement": episode_class_agreement.tolist(),
        "episode_contact_class_agreement_count": int(np.count_nonzero(episode_class_agreement)),
        "maximum_cpu_mjx_qpos_error": np.max(np.abs(cpu_poses - gpu_qpos), axis=(1, 2)).tolist(),
        "gpu_qpos_hash": hash_bytes(gpu_qpos.tobytes()),
        "gpu_contact_hash": hash_bytes(gpu_contact.tobytes()),
        "pelvis_minimum_m": np.min(gpu_qpos[:, :, 2], axis=1).tolist(),
        "source_hash": source_hash,
        "parent_trace_hash": hash_bytes(parent_trace.read_bytes()),
        "promotion_authorized": False,
    }
    report["report_hash"] = hash_json(report)
    output.write_text(json.dumps(report, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--parent-trace", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--snapshot-frame", type=int, default=80)
    parser.add_argument("--control-frames", type=int, default=30)
    parser.add_argument("--envs", type=int, default=4)
    parser.add_argument("--course-grid", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(**vars(args)), sort_keys=True))


if __name__ == "__main__":
    main()
