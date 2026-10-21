"""Recorded-target GPU Newton convergence diagnostic; not a policy certificate."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("probe-root", "native-diagnostic-root", "bank-root", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("probe-hash", "native-diagnostic-hash", "bank-hash", "expected-head"):
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
        raise ValueError("new output and frozen diagnostic required")
    pins = {}
    reports = []
    for k in ("probe", "native_diagnostic", "bank"):
        root = getattr(a, k + "_root")
        h = getattr(a, k + "_hash")
        f = root / "complete.json"
        if digest(f) != h:
            raise ValueError("immutable diagnostic prerequisites required")
        d = json.loads(f.read_text())
        reports.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(f)] = h
    probe, native, bank = reports
    if (
        native.get("status") != "PRECONTACT_NUMERICAL_DIAGNOSTIC_NOT_QUALIFICATION"
        or native.get("original_cpu_replay_bitexact") is not True
    ):
        raise ValueError("actual native same-target replay required first")
    raw_path = a.probe_root / "first-control.npz"
    pins[str(raw_path)] = probe["first_raw_hash"]
    np_path = a.native_diagnostic_root / "native-numerical-diagnostic.npz"
    pins[str(np_path)] = native["raw_hash"]
    bp = a.bank_root / "precontact-resets.npz"
    pins[str(bp)] = bank["bank_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            newton_iterations=[10, 20, 40],
            full_independent_repeats=3,
            worlds=256,
            substeps=10,
            maximum_gpu_world_steps=23040,
            neural_calls=0,
            optimizer_updates=0,
            unchanged_bodies_colliders_motors_targets_dt_and_initial_state=True,
            numerical_setting_diagnostic_not_changed_football_task=True,
            no_threshold_widening=True,
            no_automatic_qualification=True,
            hardware_authorized=False,
        ),
    )
    env = None
    steps = 0
    results = []
    try:
        import numpy as np
        import torch
        import warp as wp

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from scripts.rsi_mjlab_precontact_backend import (
            PrecontactMotionEnv,
            make_precontact_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        with np.load(raw_path, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        with np.load(np_path, allow_pickle=False) as z:
            oracle_q = z["iter40_gpu_target_qpos"].copy()
            oracle_v = z["iter40_gpu_target_qvel"].copy()
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        for iterations in (10, 20, 40):
            for repeat in range(3):
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
                cfg.sim.mujoco.iterations = iterations
                cfg.commands["motion"].sampling_mode = "uniform"
                env = PrecontactMotionEnv(
                    cfg=cfg,
                    device="cuda:0",
                    render_mode=None,
                    reset_bank=bp,
                    reset_bank_hash=bank["bank_hash"],
                )
                env.reset(seed=6601)
                if env.sim.mj_model.opt.iterations != iterations:
                    raise ValueError("actual compiled Newton setting differs")
                if not np.array_equal(
                    env.sim.data.qpos.cpu().numpy(), raw["initial_qpos"]
                ) or not np.array_equal(env.sim.data.qvel.cpu().numpy(), raw["initial_qvel"]):
                    raise ValueError(
                        "same initial physical state required for numerical comparison"
                    )
                layout = env._course_layout
                env.sim.data.qacc_warmstart[:] = torch.as_tensor(
                    raw["initial_warmstart"], device=env.device
                )
                target = torch.as_tensor(raw["processed_joint_target"], device=env.device)
                for _ in range(10):
                    env.sim.data.ctrl[:, layout.ai] = target
                    env.sim.step()
                    steps += 256
                    env.scene.update(dt=0.002)
                q = env.sim.data.qpos.cpu().numpy().copy()
                v = env.sim.data.qvel.cpu().numpy().copy()
                error = abs(v - oracle_v)
                worst = np.unravel_index(error.argmax(), error.shape)
                solver = getattr(env.sim.wp_data, "solver_niter", None)
                counts = (
                    wp.to_torch(solver).cpu().numpy().copy()
                    if solver is not None
                    else np.zeros(0, dtype=np.int32)
                )
                folder = a.output / f"iter{iterations}-repeat{repeat}"
                folder.mkdir()
                path = folder / "gpu.npz"
                np.savez_compressed(path, qpos=q, qvel=v, solver_niter=counts)
                row = dict(
                    iterations=iterations,
                    repeat=repeat,
                    max_qpos=float(abs(q - oracle_q).max()),
                    max_qvel=float(error.max()),
                    worst_world=int(worst[0]),
                    worst_dof=int(worst[1]),
                    max_solver_iterations=int(counts.max()) if counts.size else None,
                    saturated_worlds=int((counts >= iterations).sum()) if counts.size else None,
                    raw_hash=digest(path),
                )
                write_once(folder / "result.json", row)
                results.append(row)
                env.close()
                env = None
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="GPU_PRECONTACT_SOLVER_DIAGNOSTIC_NOT_QUALIFICATION",
                input_file_hashes=pins,
                results=results,
                gpu_world_steps=steps,
                neural_calls=0,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc), gpu_world_steps=steps, neural_calls=0, hardware_authorized=False
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
