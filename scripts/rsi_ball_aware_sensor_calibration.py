"""Bounded SIM sensor calibration fixtures; NOT policy performance rollouts."""

import mujoco
import numpy as np
import torch

from scripts.rsi_mjlab_ball_aware_backend import actual_foot_force_history


def calibrate(env, layout, output, *, trace=False):
    """Six capsule-end contacts (alternating feet), two negatives, ten substeps.

    Fixture state is set BEFORE integration, with the robot raised 0.5m so the
    probes do not intersect the ground. No state overwrite occurs in the run.
    Targets hold the fixture's initial joint angles through original native PD.
    """
    if env.num_envs != 8:
        raise ValueError("exact eight-world calibration fixture required")
    model = env.sim.mj_model
    env._ball_fixture_counts = dict(
        gpu_attempted_world_steps=0, gpu_world_steps=0, native_attempted_steps=0, native_steps=0
    )
    feet = [model.body("robot/" + side + "_ankle_roll_link").id for side in ("left", "right")]
    q = env.sim.data.qpos.cpu().numpy().copy()
    v = np.zeros((8, model.nv), dtype=np.float32)
    for world in range(8):
        q[world, layout.rq + 2] += 0.5
        data = mujoco.MjData(model)
        data.qpos[:] = q[world]
        data.qvel[:] = v[world]
        mujoco.mj_forward(model, data)
        if world < 6:
            side = world % 2
            geom = model.geom("robot/" + ("left" if side == 0 else "right") + "_foot4_collision").id
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError("declared capsule-end calibration geometry required")
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            center = data.geom_xpos[geom] + axis * (model.geom_size[geom, :2].sum() + 0.11 - 0.02)
        else:
            center = q[world, layout.rq : layout.rq + 3] + np.array([2.0, 2.0, 1.0])
        q[world, layout.bq : layout.bq + 3] = center
        q[world, layout.bq + 3 : layout.bq + 7] = [1, 0, 0, 0]
    # Calibration-only initial conditions; this is not a football success demo.
    env.sim.data.qpos[:] = torch.as_tensor(q, device=env.device)
    env.sim.data.qvel[:] = torch.as_tensor(v, device=env.device)
    env.sim.data.qacc_warmstart.zero_()
    env.sim.forward()
    env.sim.sense()
    target = q[:, layout.qi].copy()
    cpu = []
    for world in range(8):
        data = mujoco.MjData(model)
        data.qpos[:] = q[world]
        data.qvel[:] = v[world]
        data.mocap_pos[:] = env.sim.data.mocap_pos[world].cpu().numpy()
        data.mocap_quat[:] = env.sim.data.mocap_quat[world].cpu().numpy()
        data.ctrl[layout.ai] = target[world]
        cpu.append(data)
    gpu_q, gpu_v, native_q, native_v, native_force = [], [], [], [], []
    env._ball_fixture_native_data = cpu
    trace_buffers = {}
    trace_sources = {}
    if trace:
        import warp as wp

        data = env.sim.wp_data
        trace_sources = {
            "nacon": data.nacon,
            "nefc": data.nefc,
            "sensordata": data.sensordata,
            "efc_force": data.efc.force,
            **{
                "contact_" + name: getattr(data.contact, name)
                for name in ("geom", "worldid", "dim", "efc_address", "friction", "type")
            },
        }
        trace_sources = {key: wp.to_torch(value) for key, value in trace_sources.items()}
        if (
            sum(value.numel() * value.element_size() * 10 for value in trace_sources.values())
            > 64 * 1024 * 1024
        ):
            raise ValueError("bounded 64MiB contact diagnostic buffers required")
        trace_buffers = {
            key: torch.empty((10, *value.shape), dtype=value.dtype, device=value.device)
            for key, value in trace_sources.items()
        }
    native_sensor_error = 0.0
    for substep in range(10):
        env.sim.data.ctrl[:, layout.ai] = torch.as_tensor(target, device=env.device)
        env._ball_fixture_counts["gpu_attempted_world_steps"] += 8
        env.sim.step()
        env._ball_fixture_counts["gpu_world_steps"] += 8
        for key, value in trace_sources.items():
            trace_buffers[key][substep].copy_(value)
        env.scene.update(dt=0.002)
        gpu_q.append(env.sim.data.qpos.cpu().numpy().copy())
        gpu_v.append(env.sim.data.qvel.cpu().numpy().copy())
        nq, nv, nf = [], [], []
        for data in cpu:
            env._ball_fixture_counts["native_attempted_steps"] += 1
            mujoco.mj_step(model, data)
            env._ball_fixture_counts["native_steps"] += 1
            forces = np.zeros(2)
            for index in range(data.ncon):
                contact = data.contact[index]
                a, b = (int(model.geom_bodyid[g]) for g in (contact.geom1, contact.geom2))
                other = b if a == layout.ball else a if b == layout.ball else -1
                if other in feet:
                    wrench = np.zeros(6)
                    mujoco.mj_contactForce(model, data, index, wrench)
                    side = feet.index(other)
                    forces[side] = max(forces[side], np.linalg.norm(wrench[:3]))
            for side, name in enumerate(("left", "right")):
                sensor = model.sensor("ball_" + name + "_foot_ball_force")
                sensed = np.linalg.norm(data.sensordata[sensor.adr[0] : sensor.adr[0] + 3])
                native_sensor_error = max(native_sensor_error, abs(sensed - forces[side]))
            nq.append(data.qpos.copy())
            nv.append(data.qvel.copy())
            nf.append(forces)
        native_q.append(nq)
        native_v.append(nv)
        native_force.append(nf)
    history = actual_foot_force_history(env).cpu().numpy()
    native = np.asarray(native_force).transpose(1, 0, 2)
    q_error = float(np.abs(np.asarray(gpu_q) - native_q).max())
    v_error = float(np.abs(np.asarray(gpu_v) - native_v).max())
    force_error = float(np.abs(history - native).max())
    expected = np.zeros((8, 2), dtype=bool)
    for world in range(6):
        expected[world, world % 2] = True
    native_label = native.max(axis=1) > 10
    gpu_label = history.max(axis=1) > 10
    np.savez_compressed(
        output / "sensor-calibration.npz",
        initial_qpos=q,
        initial_qvel=v,
        target=target,
        gpu_qpos=gpu_q,
        gpu_qvel=gpu_v,
        native_qpos=native_q,
        native_qvel=native_v,
        native_force=native,
        gpu_force_history=history,
        **{"trace_" + key: value.cpu().numpy() for key, value in trace_buffers.items()},
    )
    result = dict(
        gpu_world_steps=80,
        native_steps=80,
        fixture_only_not_skill=True,
        native_sensor_vs_collision_max_error_n=native_sensor_error,
        max_qpos_error_mixed_units=q_error,
        max_qvel_error_mixed_units=v_error,
        max_force_error_n=force_error,
        gpu_labels=gpu_label.tolist(),
        native_labels=native_label.tolist(),
        expected_labels=expected.tolist(),
    )
    # Tolerances declared before execution; no widening after an observed failure.
    passed = (
        np.array_equal(gpu_label, expected)
        and np.array_equal(native_label, expected)
        and native_sensor_error <= 1e-5
        and q_error <= 1e-3
        and v_error <= 0.02
        and np.allclose(history, native, atol=2.0, rtol=0.05)
    )
    return result, bool(passed)
