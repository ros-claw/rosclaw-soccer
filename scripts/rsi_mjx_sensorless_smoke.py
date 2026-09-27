"""SIM_ONLY GPU MJX football-scene physics and batching smoke test.

This is a training-readiness check, not a learned policy, soccer skill, or
promotion exam. Raw evidence must be written outside the source checkout.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from rosclaw_soccer.providers.g1.sonic_runup import G1SonicRunupController
from rosclaw_soccer.sim.contracts import hash_bytes, hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec, build_g1_stadium_sensorless_model


def run(*, asset_root: Path, output: Path, envs: int) -> dict[str, object]:
    source = Path(__file__)
    source_hash = hash_bytes(source.read_bytes())
    scene = asset_root / "g1_description" / "scene_with_ball.xml"
    if (
        not scene.is_file()
        or not 1 <= envs <= 64
        or output.exists()
        or not output.parent.is_dir()
        or output.resolve().is_relative_to(Path(__file__).resolve().parents[1])
    ):
        raise ValueError("qualified assets, 1..64 environments, and external new output required")
    devices = jax.devices()
    if not devices or devices[0].platform != "gpu":
        raise RuntimeError("CUDA MJX smoke requires a real GPU JAX device")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible is None or not visible.isdecimal() or not 0 <= int(visible) <= 3:
        raise RuntimeError("one explicit physical GPU index 0..3 is required")
    goal = G1TrainingGoalSpec(ball_radius_m=0.11, ball_mass_kg=0.43)
    model = build_g1_stadium_sensorless_model(asset_root, goal)
    if (model.nq, model.nv, model.nu, model.nsensor) != (43, 41, 29, 0):
        raise ValueError("qualified sensorless G1 + football dynamics required")
    model.opt.timestep = 0.002
    cpu = mujoco.MjData(model)
    cpu.qpos[:7] = (0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0)
    cpu.qpos[7:36] = G1SonicRunupController.default_angles
    cpu.qpos[36:43] = (2.5, 0.1, 0.11, 1.0, 0.0, 0.0, 0.0)
    cpu.qvel[35] = -0.5
    cpu.qvel[39] = -0.5 / 0.11
    mujoco.mj_forward(model, cpu)
    physics_model = mjx.put_model(model)
    initial = mjx.put_data(model, cpu)
    batched = jax.tree.map(lambda value: jnp.broadcast_to(value, (envs,) + value.shape), initial)
    # Distinct ball positions prove there are separate physical environments.
    ball_x = jnp.linspace(2.45, 2.55, envs)
    batched = batched.replace(qpos=batched.qpos.at[:, 36].set(ball_x))

    def one_frame(state: mjx.Data) -> mjx.Data:
        return jax.lax.fori_loop(0, 10, lambda _, value: mjx.step(physics_model, value), state)

    advanced = jax.jit(jax.vmap(one_frame))(batched)
    qpos = np.asarray(advanced.qpos)
    qvel = np.asarray(advanced.qvel)
    if qpos.shape != (envs, 43) or qvel.shape != (envs, 41):
        raise AssertionError("MJX did not advance the requested environment batch")
    if not (np.isfinite(qpos).all() and np.isfinite(qvel).all()):
        raise ValueError("nonfinite vectorized MJX football physics")
    if not math.isclose(float(np.asarray(advanced.time)[0]), 0.02, abs_tol=1e-6):
        raise ValueError("MJX did not advance ten 2 ms physical steps")
    if envs > 1 and len(np.unique(qpos[:, 36])) != envs:
        raise ValueError("vectorized ball states collapsed onto one shared world")
    # Compare the first lane to CPU MuJoCo with its own initial ball position.
    cpu.qpos[36] = float(ball_x[0])
    mujoco.mj_forward(model, cpu)
    for _ in range(10):
        mujoco.mj_step(model, cpu)
    max_qpos_error = float(np.max(np.abs(qpos[0] - cpu.qpos)))
    max_qvel_error = float(np.max(np.abs(qvel[0] - cpu.qvel)))
    if max_qpos_error > 1e-4 or max_qvel_error > 1e-3:
        raise ValueError("one-frame GPU MJX and CPU MuJoCo physics diverged")
    if hash_bytes(source.read_bytes()) != source_hash:
        raise RuntimeError("smoke source changed during physics execution")
    result: dict[str, object] = {
        "schema": "rosclaw_soccer.rsi.mjx_sensorless_smoke.v1",
        "activation_ceiling": "SIM_ONLY",
        "envs": envs,
        "device": str(devices[0]),
        "physical_gpu_index": int(visible),
        "gpu_physics_advanced": True,
        "control_frame_s": 0.02,
        "physical_steps": 10,
        "sensor_count": model.nsensor,
        "model_shape": [model.nq, model.nv, model.nu, model.ngeom],
        "ball_x_after_m": qpos[:, 36].tolist(),
        "pelvis_height_after_m": qpos[:, 2].tolist(),
        "maximum_cpu_mjx_qpos_error": max_qpos_error,
        "maximum_cpu_mjx_qvel_error": max_qvel_error,
        "source_hash": source_hash,
        "scene_hash": hash_bytes(scene.read_bytes()),
        "promotion_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    output.write_text(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--envs", type=int, default=4)
    args = parser.parse_args()
    print(json.dumps(run(asset_root=args.asset_root, output=args.output, envs=args.envs)))


if __name__ == "__main__":
    main()
