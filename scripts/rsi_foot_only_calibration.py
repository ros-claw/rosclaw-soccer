"""SIM-only paired native/GPU foot and knee probes; never skill evidence."""

import mujoco
import numpy as np
import torch

from scripts.rsi_mjlab_ball_aware_backend import actual_foot_force_history
from scripts.rsi_mjlab_foot_only_backend import actual_nonfoot_force_history


def body_bindings(model, layout):
    feet = {model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")}
    expected = {
        i for i in range(model.nbody) if (model.body(i).name or "").startswith("robot/")
    } - feet
    found = []
    for i in range(model.nsensor):
        sensor = model.sensor(i)
        if not sensor.name.startswith("ball_nonfoot") or not sensor.name.endswith("_force"):
            continue
        if (
            sensor.objtype[0] != mujoco.mjtObj.mjOBJ_BODY
            or sensor.reftype[0] != mujoco.mjtObj.mjOBJ_BODY
            or sensor.refid[0] != layout.ball
        ):
            raise ValueError("nonfoot sensor must bind robot BODY against ball BODY")
        found.append(int(sensor.objid[0]))
    if set(found) != expected or len(found) != len(expected):
        raise ValueError(f"nonfoot sensor coverage differs: {found}, expected {sorted(expected)}")
    return dict(nonfoot_body_ids=sorted(expected), foot_body_ids=sorted(feet))


def calibrate(env, layout, output, *, profile):
    if env.num_envs != 8 or profile not in ("foot", "knee"):
        raise ValueError("eight-world explicit foot/knee fixture required")
    model = env.sim.mj_model
    bindings = body_bindings(model, layout)
    feet = [model.body("robot/" + s + "_ankle_roll_link").id for s in ("left", "right")]
    nonfeet = set(bindings["nonfoot_body_ids"])
    counts = dict(gpu_world_steps=0, native_steps=0)
    env._foot_only_fixture_counts = counts
    q = env.sim.data.qpos.cpu().numpy().copy()
    v = np.zeros((8, model.nv), dtype=np.float32)
    geometry = []
    for world in range(8):
        q[world, layout.rq + 2] += 0.5
        data = mujoco.MjData(model)
        data.qpos[:] = q[world]
        mujoco.mj_forward(model, data)
        if world < 6:
            side = ("left", "right")[world % 2]
            if profile == "foot":
                geom = model.geom("robot/" + side + "_foot4_collision").id
            else:
                knee = model.body("robot/" + side + "_knee_link").id
                eligible = [
                    g
                    for g in range(model.ngeom)
                    if model.geom_bodyid[g] == knee
                    and model.geom_type[g] == mujoco.mjtGeom.mjGEOM_CAPSULE
                    and (model.geom_contype[g] or model.geom_conaffinity[g])
                ]
                if len(eligible) != 1:
                    raise ValueError("unique compiled knee collision capsule required")
                geom = eligible[0]
            if model.geom_type[geom] != mujoco.mjtGeom.mjGEOM_CAPSULE:
                raise ValueError("capsule fixture required")
            axis = data.geom_xmat[geom].reshape(3, 3)[:, 2]
            radius = model.geom_size[geom, :2].sum() + 0.11 - 0.02
            sign = 1
            if profile == "knee":
                # Select BEFORE integration; farther from both feet, no outcome search.
                sign = max(
                    (-1, 1),
                    key=lambda s: min(
                        np.linalg.norm(data.geom_xpos[geom] + s * axis * radius - data.xpos[f])
                        for f in feet
                    ),
                )
            center = data.geom_xpos[geom] + sign * axis * radius
            geometry.append(dict(world=world, geom=model.geom(geom).name, endpoint_sign=sign))
        else:
            center = q[world, layout.rq : layout.rq + 3] + [2, 2, 1]
        q[world, layout.bq : layout.bq + 3] = center
        q[world, layout.bq + 3 : layout.bq + 7] = [1, 0, 0, 0]
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
        data.ctrl[layout.ai] = target[world]
        data.mocap_pos[:] = env.sim.data.mocap_pos[world].cpu().numpy()
        data.mocap_quat[:] = env.sim.data.mocap_quat[world].cpu().numpy()
        cpu.append(data)
    gq, gv, nq, nv, forces = [], [], [], [], []
    for _ in range(10):
        env.sim.data.ctrl[:, layout.ai] = torch.as_tensor(target, device=env.device)
        env.sim.step()
        counts["gpu_world_steps"] += 8
        env.scene.update(dt=0.002)
        gq.append(env.sim.data.qpos.cpu().numpy().copy())
        gv.append(env.sim.data.qvel.cpu().numpy().copy())
        frame = []
        for data in cpu:
            mujoco.mj_step(model, data)
            counts["native_steps"] += 1
            row = np.zeros(3)
            for i in range(data.ncon):
                c = data.contact[i]
                a, b = (int(model.geom_bodyid[g]) for g in (c.geom1, c.geom2))
                other = b if a == layout.ball else a if b == layout.ball else -1
                if other not in nonfeet and other not in feet:
                    continue
                wrench = np.zeros(6)
                mujoco.mj_contactForce(model, data, i, wrench)
                index = feet.index(other) if other in feet else 2
                row[index] = max(row[index], np.linalg.norm(wrench[:3]))
            frame.append(row)
        forces.append(frame)
        nq.append([d.qpos.copy() for d in cpu])
        nv.append([d.qvel.copy() for d in cpu])
    native = np.asarray(forces).transpose(1, 0, 2)
    gpu = np.concatenate(
        (
            actual_foot_force_history(env).cpu().numpy(),
            actual_nonfoot_force_history(env).cpu().numpy()[:, :, None],
        ),
        axis=2,
    )
    expected = np.zeros((8, 3), dtype=bool)
    for i in range(6):
        expected[i, i % 2 if profile == "foot" else 2] = True
    thresholds = np.array([10.0, 10.0, 1.0])
    gl, nl = (a.max(axis=1) > thresholds for a in (gpu, native))
    qe = float(np.max(np.abs(np.asarray(gq) - nq)))
    ve = float(np.max(np.abs(np.asarray(gv) - nv)))
    passed = (
        np.array_equal(gl, expected)
        and np.array_equal(nl, expected)
        and qe <= 0.001
        and ve <= 0.02
        and np.allclose(gpu, native, atol=2.0, rtol=0.05)
    )
    np.savez_compressed(
        output / "fixture.npz",
        initial_qpos=q,
        initial_qvel=v,
        target=target,
        gpu_qpos=gq,
        gpu_qvel=gv,
        native_qpos=nq,
        native_qvel=nv,
        gpu_force=gpu,
        native_force=native,
    )
    return dict(
        profile=profile,
        passed=bool(passed),
        **counts,
        geometry=geometry,
        gpu_labels=gl.tolist(),
        native_labels=nl.tolist(),
        expected=expected.tolist(),
        max_qpos_error=qe,
        max_qvel_error=ve,
        max_force_error_n=float(np.max(np.abs(gpu - native))),
    )
