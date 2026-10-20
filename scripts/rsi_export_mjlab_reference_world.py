"""Export the qualified reference-PD world and audit one CPU/GPU control.

This is a bounded simulation diagnostic, not football or long-run agreement.
Native CPU physics uses the actual compiled MJLab model and actuator mapping.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("teacher-root", "reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("teacher-unit", "teacher-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen diagnostic required")
    wait_for_terminal_collection(a.teacher_unit)
    receipt_path = a.teacher_root / "complete.json"
    if digest(receipt_path) != a.teacher_hash:
        raise ValueError("actual immutable reference-control evidence required")
    receipt = json.loads(receipt_path.read_text())
    if (
        receipt["status"] != "MJLAB_REFERENCE_PD_AUTHORITY_NOT_NEURAL_LEARNING"
        or receipt["actual_neural_control_samples"] != 0
        or receipt["new_optimizer_steps"] != 0
    ):
        raise ValueError("completed non-neural authority diagnostic required")
    pins = dict(receipt["input_file_hashes"])
    pins[str(receipt_path)] = a.teacher_hash
    snapshot_path = a.teacher_root / "0-default-initial.npz"
    pins[str(snapshot_path)] = digest(snapshot_path)
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        for source in directory.rglob("*.py"):
            pins[str(source)] = digest(source)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=64,
            maximum_cpu_native_steps=640,
            maximum_gpu_world_substeps=640,
            control_steps=1,
            physics_dt=0.002,
            decimation=10,
            hardware_authorized=False,
            new_optimizer_steps=0,
            new_neural_calls=0,
        ),
    )
    env = None
    native_steps = 0
    gpu_calls = 0
    try:
        import mujoco
        import numpy as np
        import torch

        if torch.cuda.device_count() != 1:
            raise ValueError("one visible GPU required")
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from scripts.rsi_mjlab_motion_backend import (
            ExplicitBoundaryMotionEnv,
            make_motion_training_cfg,
        )

        cfg = make_motion_training_cfg(
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=64,
            seed=1601,
            balanced_initial_sources=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = ExplicitBoundaryMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=1601)
        motion = env.command_manager.get_term("motion")
        model = env.sim.mj_model
        state = dict(
            qpos=env.sim.data.qpos.cpu().numpy().copy(),
            qvel=env.sim.data.qvel.cpu().numpy().copy(),
            observation=obs["actor"].cpu().numpy().copy(),
            phase=motion.time_steps.cpu().numpy().copy(),
            motion_id=motion.motion_idx.cpu().numpy().copy(),
        )
        with np.load(snapshot_path, allow_pickle=False) as original:
            if any(not np.array_equal(state[k], original[k]) for k in state):
                raise ValueError("reconstructed body snapshot differs from original diagnostic")
        if env._sim_step_counter != 0 or model.nu != 29 or model.opt.timestep != 0.002:
            raise ValueError("original zero-step model and 2ms physical contract required")
        term = env.action_manager.get_term("joint_pos")
        if tuple(term.target_names) != G1_DDS_JOINT_NAMES or torch.any(
            motion.robot.data.encoder_bias != 0
        ):
            raise ValueError("explicit HG29 codec and zero encoder bias required")
        actuator_ids = []
        for name in G1_DDS_JOINT_NAMES:
            jid = model.joint("robot/" + name).id
            found = np.flatnonzero(
                (model.actuator_trntype == int(mujoco.mjtTrn.mjTRN_JOINT))
                & (model.actuator_trnid[:, 0] == jid)
            )
            if found.shape != (1,):
                raise ValueError("one compiled actuator per qualified robot joint required")
            actuator_ids.append(int(found[0]))
        defaults = torch.as_tensor(term.offset).cpu().numpy().copy()
        warmstart = env.sim.data.qacc_warmstart.cpu().numpy().copy()
        gpu_initial_xpos = env.sim.data.xpos.cpu().numpy().copy()
        native_q = []
        native_v = []
        initial_fk_errors = []
        for world in range(64):
            data = mujoco.MjData(model)
            data.qpos[:] = state["qpos"][world]
            data.qvel[:] = state["qvel"][world]
            data.qacc_warmstart[:] = warmstart[world]
            data.ctrl[actuator_ids] = defaults[world]
            mujoco.mj_forward(model, data)
            initial_fk_errors.append(float(np.max(np.abs(data.xpos - gpu_initial_xpos[world]))))
            for _ in range(10):
                mujoco.mj_step(model, data)
                native_steps += 1
            native_q.append(data.qpos.copy())
            native_v.append(data.qvel.copy())
        counter = env._sim_step_counter
        env.step(torch.zeros((64, 29), device="cuda:0"))
        torch.cuda.synchronize()
        gpu_calls = env._sim_step_counter - counter
        if gpu_calls != 10:
            raise ValueError("exactly one actual ten-substep GPU control required")
        gpu_q = env.sim.data.qpos.cpu().numpy().copy()
        gpu_v = env.sim.data.qvel.cpu().numpy().copy()
        native_q = np.asarray(native_q)
        native_v = np.asarray(native_v)
        if not all(np.isfinite(v).all() for v in (native_q, native_v, gpu_q, gpu_v)):
            raise ValueError("finite raw CPU and GPU states required")
        np.savez_compressed(
            a.output / "native-gpu-control1.npz",
            **state,
            cpu_qpos=native_q,
            cpu_qvel=native_v,
            gpu_qpos=gpu_q,
            gpu_qvel=gpu_v,
            qacc_warmstart=warmstart,
            initial_default_target=defaults,
        )
        model_path = a.output / "compiled_model.mjb"
        mujoco.mj_saveModel(model, str(model_path))
        qerr = np.max(np.abs(native_q - gpu_q), axis=1)
        verr = np.max(np.abs(native_v - gpu_v), axis=1)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="MJLAB_NATIVE_EXPORT_ONE_CONTROL_DIAGNOSTIC_NOT_LONG_RUN_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                model_hash=digest(model_path),
                state_artifact_hash=digest(a.output / "native-gpu-control1.npz"),
                nq=model.nq,
                nv=model.nv,
                nu=model.nu,
                timestep=model.opt.timestep,
                cpu_native_steps=native_steps,
                gpu_step_calls=gpu_calls,
                gpu_world_substeps=gpu_calls * 64,
                max_initial_fk_position_error_m=max(initial_fk_errors),
                max_abs_qpos_error=qerr.tolist(),
                max_abs_qvel_error=verr.tolist(),
                max_abs_qpos_error_mixed_units=float(qerr.max()),
                max_abs_qvel_error_mixed_units=float(verr.max()),
                actuator_ids_hg=actuator_ids,
                actuator_gainprm=model.actuator_gainprm.tolist(),
                actuator_biasprm=model.actuator_biasprm.tolist(),
                actuator_forcelimited=model.actuator_forcelimited.tolist(),
                actuator_forcerange=model.actuator_forcerange.tolist(),
                new_optimizer_steps=0,
                new_neural_calls=0,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                cpu_native_steps=native_steps,
                gpu_step_calls=gpu_calls,
                active_env_gpu_counter=env._sim_step_counter if env is not None else None,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
