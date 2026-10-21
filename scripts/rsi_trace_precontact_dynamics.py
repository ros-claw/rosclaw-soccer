"""Trace the rejected fixed-control CPU/GPU probe; no learning or qualification.

Record every native and GPU substep, including the contact manifold before
integration. Fixed diagnostic worlds include the observed outlier and controls.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("probe-root", "role-root", "bank-root", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("probe-hash", "role-hash", "bank-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen source required")
    pins, reports = {}, []
    for key in ("probe", "role", "bank"):
        f = getattr(a, key + "_root") / "complete.json"
        h = getattr(a, key + "_hash")
        if digest(f) != h:
            raise ValueError("original receipt required")
        r = json.loads(f.read_text())
        reports.append(r)
        pins.update(r["input_file_hashes"])
        pins[str(f)] = h
    probe, role, bank = reports
    if probe.get("status") != "GPU_PRECONTACT_RESET_CONTRACT_REJECTED":
        raise ValueError("trace the preserved rejected probe only")
    rawpath = a.probe_root / "first-control.npz"
    modelpath = a.role_root / "compiled_model.mjb"
    bankpath = a.bank_root / "precontact-resets.npz"
    pins[str(rawpath)] = probe["first_raw_hash"]
    pins[str(modelpath)] = role["model_hash"]
    pins[str(bankpath)] = bank["bank_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    worlds = (0, 15, 49, 168)
    fields = (
        "qpos",
        "qvel",
        "qacc",
        "qfrc_actuator",
        "qfrc_bias",
        "qfrc_passive",
        "qfrc_constraint",
    )
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            worlds=256,
            substeps=10,
            detailed_contact_worlds=list(worlds),
            maximum_gpu_world_steps=2560,
            maximum_native_steps=2560,
            neural_calls=0,
            optimizer_updates=0,
            recorded_control_not_neural_reexecution=True,
            no_physics_configuration_changes=True,
            no_threshold_widening=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    gpu_steps = native_steps = 0
    try:
        import mujoco
        import numpy as np
        import torch
        import warp as wp

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from scripts.rsi_mjlab_precontact_backend import (
            PrecontactMotionEnv,
            make_precontact_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        with np.load(rawpath, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        with np.load(bankpath, allow_pickle=False) as z:
            b = {k: z[k].copy() for k in z.files}
        cfgpath = next(
            Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml")
        )
        params = VelocityPolicyParameters.load(cfgpath, expected_hash=pins[str(cfgpath)])
        m = mujoco.MjModel.from_binary_path(str(modelpath))
        layout = NativeVelocityLayout(m, robot_prefix="robot/", ball_body_name="ball/ball")
        if (
            m.opt.iterations != 10
            or m.opt.timestep != 0.002
            or mujoco.get_mjcb_control() is not None
        ):
            raise ValueError("unchanged original model required")
        cpu = {key: [] for key in fields}
        contacts = {"cpu": [], "gpu": []}
        for w in range(256):
            d = mujoco.MjData(m)
            d.qpos[:] = raw["initial_qpos"][w]
            d.qvel[:] = raw["initial_qvel"][w]
            d.time = int(b["reference_frame"][w]) * 0.02
            d.ctrl[layout.ai] = target29(b["previous_action"][w], params)
            d.qacc_warmstart[:] = raw["initial_warmstart"][w]
            mujoco.mj_forward(m, d)
            d.qacc_warmstart[:] = raw["initial_warmstart"][w]
            d.ctrl[layout.ai] = raw["processed_joint_target"][w]
            series = {key: [] for key in fields}
            for sub in range(10):
                mujoco.mj_step(m, d)
                native_steps += 1
                for key in fields:
                    series[key].append(getattr(d, key).copy())
                if w in worlds:
                    contacts["cpu"].append(
                        dict(
                            world=w,
                            substep=sub,
                            contacts=[
                                dict(
                                    geom=c.geom.tolist(),
                                    dist=float(c.dist),
                                    pos=c.pos.tolist(),
                                    frame=c.frame.reshape(3, 3).tolist(),
                                )
                                for c in d.contact
                            ],
                        )
                    )
            for key in fields:
                cpu[key].append(series[key])
        arrays = {"cpu_" + k: np.array(v).transpose(1, 0, 2) for k, v in cpu.items()}
        cfg = make_precontact_training_cfg(
            parameters=params,
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=256,
            seed=6601,
            balanced_initial_sources=True,
            ball_placement_protocol="body-bound-v1",
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        env = PrecontactMotionEnv(
            cfg=cfg,
            device="cuda:0",
            render_mode=None,
            reset_bank=bankpath,
            reset_bank_hash=bank["bank_hash"],
        )
        env.reset(seed=6601)
        if not np.array_equal(
            env.sim.data.qpos.cpu().numpy(), raw["initial_qpos"]
        ) or not np.array_equal(env.sim.data.qvel.cpu().numpy(), raw["initial_qvel"]):
            raise ValueError("same original initial physical state required")
        env.sim.data.qacc_warmstart[:] = torch.as_tensor(
            raw["initial_warmstart"], device=env.device
        )
        targets = torch.as_tensor(raw["processed_joint_target"], device=env.device)
        gpu = {key: [] for key in fields}
        for sub in range(10):
            env.sim.data.ctrl[:, layout.ai] = targets
            env.sim.step()
            gpu_steps += 256
            env.scene.update(dt=0.002)
            for key in fields:
                gpu[key].append(wp.to_torch(getattr(env.sim.wp_data, key)).cpu().numpy().copy())
            count = int(wp.to_torch(env.sim.wp_data.nacon).cpu()[0])
            contact = env.sim.wp_data.contact
            view = {
                key: wp.to_torch(getattr(contact, key)).cpu().numpy()[:count].copy()
                for key in ("worldid", "geom", "dist", "pos", "frame")
            }
            for w in worlds:
                indices = np.flatnonzero(view["worldid"] == w)
                contacts["gpu"].append(
                    dict(
                        world=w,
                        substep=sub,
                        contacts=[
                            dict(
                                geom=view["geom"][i].tolist(),
                                dist=float(view["dist"][i]),
                                pos=view["pos"][i].tolist(),
                                frame=view["frame"][i].tolist(),
                            )
                            for i in indices
                        ],
                    )
                )
        arrays.update({"gpu_" + k: np.array(v) for k, v in gpu.items()})
        path = a.output / "dynamics.npz"
        np.savez_compressed(path, **arrays)
        write_once(a.output / "contacts.json", contacts)
        rows = []
        for sub in range(10):
            row = dict(substep=sub)
            for key in fields:
                e = abs(arrays["gpu_" + key][sub] - arrays["cpu_" + key][sub])
                where = np.unravel_index(e.argmax(), e.shape)
                row[key] = dict(max=float(e.max()), world=int(where[0]), dof=int(where[1]))
            row["world15_ankle_qvel_error"] = float(
                abs(arrays["gpu_qvel"][sub, 15, 23] - arrays["cpu_qvel"][sub, 15, 23])
            )
            rows.append(row)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PRECONTACT_DYNAMICS_TRACE_NOT_QUALIFICATION",
                input_file_hashes=pins,
                raw_hash=digest(path),
                contacts_hash=digest(a.output / "contacts.json"),
                results=rows,
                gpu_world_steps=gpu_steps,
                native_steps=native_steps,
                neural_calls=0,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=native_steps,
            gpu_world_steps=gpu_steps,
            neural_calls=0,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
