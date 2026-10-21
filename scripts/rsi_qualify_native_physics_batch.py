"""Qualify threaded native vector physics against every recorded scalar substep."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("trace-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("trace-hash", "expected-head"):
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
        raise ValueError("frozen source and unused output required")
    f = a.trace_root / "complete.json"
    if digest(f) != a.trace_hash:
        raise ValueError("exact complete trace required")
    trace = json.loads(f.read_text())
    if trace.get("status") != "PRECONTACT_DYNAMICS_TRACE_NOT_QUALIFICATION":
        raise ValueError("full scalar dynamics trace required")
    pins = {**trace["input_file_hashes"], str(f): a.trace_hash}
    pins[str(a.trace_root / "dynamics.npz")] = trace["raw_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
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
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=256,
            substeps=10,
            independent_thread_counts=[4, 8],
            maximum_native_steps=5120,
            neural_calls=0,
            optimizer_updates=0,
            comparison="all seven fields all substeps bitexact",
            unchanged_physics_and_recorded_targets=True,
            hardware_authorized=False,
        ),
    )
    steps = 0
    batch = None
    try:
        import mujoco
        import numpy as np

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
        from rosclaw_soccer.rsi.native_physics_batch import NativePhysicsBatch
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError(f"one original input required: {suffix}")
            return matches[0]

        rawpath = pinned("/rsi-gpu-precontact-reset-qualification-v2-20261021/first-control.npz")
        modelpath = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        bankpath = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        with np.load(rawpath, allow_pickle=False) as z:
            raw = {k: z[k].copy() for k in z.files}
        with np.load(bankpath, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        with np.load(a.trace_root / "dynamics.npz", allow_pickle=False) as z:
            oracle = {k: z["cpu_" + k].transpose(1, 0, 2).copy() for k in fields}
        m = mujoco.MjModel.from_binary_path(str(modelpath))
        layout = NativeVelocityLayout(m, robot_prefix="robot/", ball_body_name="ball/ball")
        old = np.zeros((256, m.nu))
        targets = old.copy()
        old[:, layout.ai] = np.stack([target29(v, params) for v in bank["previous_action"]])
        targets[:, layout.ai] = raw["processed_joint_target"]
        results = []
        for threads in (4, 8):
            with NativePhysicsBatch(m, 256, threads=threads) as batch:
                batch.restore(
                    qpos=raw["initial_qpos"],
                    qvel=raw["initial_qvel"],
                    previous_ctrl=old,
                    warmstart=raw["initial_warmstart"],
                    time=bank["reference_frame"] * 0.02,
                )
                actual = batch.advance(targets, substeps=10, fields=fields)
                path = a.output / f"threads{threads}.npz"
                np.savez_compressed(path, **actual)
                errors = {k: float(abs(actual[k] - oracle[k]).max()) for k in fields}
                passed = all(np.array_equal(actual[k], oracle[k]) for k in fields)
                results.append(
                    dict(threads=threads, passed=passed, max_errors=errors, raw_hash=digest(path))
                )
            completed_steps = batch.native_steps
            batch = None
            steps += completed_steps
        check_pins(pins)
        passed = all(r["passed"] for r in results)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_VECTOR_PHYSICS_BITEXACT_NOT_LEARNER"
                if passed
                else "NATIVE_VECTOR_PHYSICS_REJECTED",
                input_file_hashes=pins,
                results=results,
                native_steps=steps,
                neural_calls=0,
                optimizer_updates=0,
                training_env_qualified=False,
                skill_improvement_proven=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        # A failed active batch may already have completed physical work.
        actual_steps = steps + (batch.native_steps if batch is not None else 0)
        failure = dict(
            error=repr(exc), native_steps=actual_steps, neural_calls=0, hardware_authorized=False
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        raise


if __name__ == "__main__":
    main()
