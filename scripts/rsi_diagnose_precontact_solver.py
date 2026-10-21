"""Recorded-control numerical diagnostic, NOT neural replay or qualification.

Keep rejected GPU256 evidence immutable. Separate independent NN roundoff
from native/GPU integration, and measure native Newton iteration convergence.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("probe-root", "role-root", "bank-root", "output"):
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
        raise ValueError("new output and frozen diagnostic required")
    pins = {}
    reports = []
    for k in ("probe", "role", "bank"):
        root = getattr(a, k + "_root")
        h = getattr(a, k + "_hash")
        f = root / "complete.json"
        if digest(f) != h:
            raise ValueError("exact original receipt required")
        d = json.loads(f.read_text())
        reports.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(f)] = h
    probe, role, bank = reports
    if (
        probe.get("status") != "GPU_PRECONTACT_RESET_CONTRACT_REJECTED"
        or probe.get("first_failed_worlds") != 0
        or probe.get("first_timeout_worlds") != 0
    ):
        raise ValueError("original complete rejected probe without reset discontinuities required")
    rawpath = a.probe_root / "first-control.npz"
    pins[str(rawpath)] = probe["first_raw_hash"]
    modelpath = a.role_root / "compiled_model.mjb"
    pins[str(modelpath)] = role["model_hash"]
    bankpath = a.bank_root / "precontact-resets.npz"
    pins[str(bankpath)] = bank["bank_hash"]
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            native_iterations=[10, 20, 40],
            targets=["independent_cpu_nn_recorded", "actual_gpu_target_recorded"],
            worlds=256,
            substeps=10,
            maximum_native_steps=15360,
            neural_calls=0,
            optimizer_updates=0,
            no_threshold_widening=True,
            recorded_control_diagnostic_not_neural_reexecution=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    steps = 0
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        with np.load(rawpath, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        with np.load(bankpath, allow_pickle=False) as z:
            b = {k: z[k].copy() for k in z.files}
        if not np.array_equal(raw["bank_index"], np.arange(256)):
            raise ValueError("complete original bank order required")
        cfg = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        m = mujoco.MjModel.from_binary_path(str(modelpath))
        layout = NativeVelocityLayout(m, robot_prefix="robot/", ball_body_name="ball/ball")
        if (
            m.opt.iterations != 10
            or m.opt.timestep != 0.002
            or mujoco.get_mjcb_control() is not None
        ):
            raise ValueError("original numerical model required")
        rows = []
        arrays = {}
        for iterations in (10, 20, 40):
            m.opt.iterations = iterations
            for arm, key in (
                ("cpu-target", "cpu_target"),
                ("gpu-target", "processed_joint_target"),
            ):
                q = []
                v = []
                niter = []
                for w in range(256):
                    d = mujoco.MjData(m)
                    d.qpos[:] = raw["initial_qpos"][w]
                    d.qvel[:] = raw["initial_qvel"][w]
                    d.time = int(b["reference_frame"][w]) * 0.02
                    d.ctrl[layout.ai] = target29(b["previous_action"][w], params)
                    d.qacc_warmstart[:] = raw["initial_warmstart"][w]
                    mujoco.mj_forward(m, d)
                    d.qacc_warmstart[:] = raw["initial_warmstart"][w]
                    d.ctrl[layout.ai] = raw[key][w]
                    counts = []
                    warnings = tuple(int(x.number) for x in d.warning)
                    for _ in range(10):
                        mujoco.mj_step(m, d)
                        steps += 1
                        counts.append(int(d.solver_niter.max()))
                        if tuple(int(x.number) for x in d.warning) != warnings:
                            raise ValueError("native warning during numerical diagnostic")
                    q.append(d.qpos.copy())
                    v.append(d.qvel.copy())
                    niter.append(counts)
                q = np.array(q)
                v = np.array(v)
                niter = np.array(niter)
                label = f"iter{iterations}_{arm.replace('-', '_')}"
                arrays[label + "_qpos"] = q
                arrays[label + "_qvel"] = v
                arrays[label + "_solver_niter"] = niter
                error = abs(v - raw["qvel_after"])
                where = np.unravel_index(error.argmax(), error.shape)
                rows.append(
                    dict(
                        iterations=iterations,
                        arm=arm,
                        max_qpos_vs_gpu10=float(abs(q - raw["qpos_after"]).max()),
                        max_qvel_vs_gpu10=float(error.max()),
                        worst_world=int(where[0]),
                        worst_dof=int(where[1]),
                        saturated_substeps=int((niter >= iterations).sum()),
                        max_solver_iterations=int(niter.max()),
                    )
                )
                if (
                    iterations == 10
                    and arm == "cpu-target"
                    and (
                        not np.array_equal(q, raw["cpu_qpos"])
                        or not np.array_equal(v, raw["cpu_qvel"])
                    )
                ):
                    np.savez_compressed(a.output / "failed-original-replay.npz", qpos=q, qvel=v)
                    raise ValueError(
                        "same model and recorded CPU target must exactly reproduce original CPU arm"
                    )
        if steps != 15360:
            raise ValueError("complete declared numerical diagnostic required")
        path = a.output / "native-numerical-diagnostic.npz"
        np.savez_compressed(path, **arrays)
        convergence = []
        for lo, hi in ((10, 20), (20, 40)):

            def key(n):
                return f"iter{n}_gpu_target"

            convergence.append(
                dict(
                    lower=lo,
                    higher=hi,
                    max_qpos=float(
                        abs(arrays[key(lo) + "_qpos"] - arrays[key(hi) + "_qpos"]).max()
                    ),
                    max_qvel=float(
                        abs(arrays[key(lo) + "_qvel"] - arrays[key(hi) + "_qvel"]).max()
                    ),
                )
            )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PRECONTACT_NUMERICAL_DIAGNOSTIC_NOT_QUALIFICATION",
                input_file_hashes=pins,
                source_head=a.expected_head,
                results=rows,
                native_convergence=convergence,
                original_cpu_replay_bitexact=True,
                raw_hash=digest(path),
                native_steps=steps,
                neural_calls=0,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc), native_steps=steps, neural_calls=0, hardware_authorized=False
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise


if __name__ == "__main__":
    main()
