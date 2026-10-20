"""Frozen SIM foot-only sensor qualification; zero optimizer updates."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("learner-root", "export-root", "reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("learner-hash", "export-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("new output and exact clean frozen source required")
    pins = {}
    receipts = []
    for root, expected in ((a.learner_root, a.learner_hash), (a.export_root, a.export_hash)):
        f = root / "complete.json"
        if digest(f) != expected:
            raise ValueError("prerequisite receipt bytes differ")
        d = json.loads(f.read_text())
        pins.update(d["input_file_hashes"])
        pins[str(f)] = expected
        receipts.append(d)
    learner, export = receipts
    checkpoint = a.learner_root / "ball-aware-update200.pt"
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    model_path = a.export_root / "compiled_model.mjb"
    pins[str(model_path)] = export["model_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            fixture_only_not_skill=True,
            profiles=["foot", "knee"],
            repeats=3,
            worlds=8,
            substeps=10,
            qpos_atol=0.001,
            qvel_atol=0.02,
            force_atol_n=2.0,
            force_rtol=0.05,
            contact_thresholds_n=[10.0, 10.0, 1.0],
            robot_raise_m=0.5,
            penetration_m=0.02,
            maximum_fixture_gpu_world_steps=480,
            maximum_native_steps=480,
            maximum_neural_probe_gpu_world_steps=80,
            maximum_nn_rows=8,
            optimizer_updates=0,
            force_decoder="addressed",
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    results = []
    neural_rows = probe_steps = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()  # Must precede simulator construction/kernel capture.
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_foot_only_calibration import body_bindings, calibrate
        from scripts.rsi_mjlab_foot_only_backend import (
            FootOnlyMotionEnv,
            make_foot_only_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        onnx = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/exported/policy.onnx"))
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        )
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        state = loaded["algorithm"]["actor_state_dict"]
        actor.load_state_dict({k: state["mlp." + k] for k in actor.state_dict()}, strict=True)
        actor.cuda().eval()
        cfg = make_foot_only_training_cfg(
            parameters=params,
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=8,
            seed=1601,
            balanced_initial_sources=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = FootOnlyMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=1601)
        if obs["actor"].shape != (8, 177) or obs["critic"].shape != (8, 304):
            raise ValueError("actor/critic ABI changed")
        model = env.sim.mj_model
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        old = mujoco.MjModel.from_binary_path(str(model_path))
        if (
            layout.profile()
            != NativeVelocityLayout(
                old, robot_prefix="robot/", ball_body_name="ball/ball"
            ).profile()
        ):
            raise ValueError("body/motors/collider changed")
        bindings = body_bindings(model, layout)
        write_once(a.output / "bindings.json", bindings)
        mujoco.mj_saveModel(model, str(a.output / "compiled_model.mjb"), None)
        for profile in ("foot", "knee"):
            for repeat in range(3):
                env.reset(seed=1601)
                folder = a.output / f"{profile}-{repeat}"
                folder.mkdir()
                result = calibrate(env, layout, folder, profile=profile)
                result.update(repeat=repeat, raw_hash=digest(folder / "fixture.npz"))
                results.append(result)
                write_once(folder / "result.json", result)
                if not result["passed"]:
                    raise ValueError(f"unchanged-threshold fixture failed: {profile}/{repeat}")
        # New, independently initialized NN probe. No state overwrite after stepping.
        with np.load(a.output / "knee-2/fixture.npz") as raw:
            q, v = raw["initial_qpos"].copy(), raw["initial_qvel"].copy()
        env.reset(seed=1601)
        env.sim.data.qpos[:] = torch.as_tensor(q, device=env.device)
        env.sim.data.qvel[:] = torch.as_tensor(v, device=env.device)
        env.sim.data.qacc_warmstart.zero_()
        env.sim.forward()
        env.sim.sense()
        env.scene.update(dt=0.002)
        obs = env.observation_manager.compute(update_history=True)
        with torch.no_grad():
            actions = actor(obs["actor"])
        neural_rows += 8
        _, _, terminated, truncated, _ = env.step(actions)
        probe_steps += 80
        snap = env.physical_step_data()
        cached = env._identity_kick_credit.cached
        forbidden = cached["forbidden_contact"].cpu().numpy()
        expected = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0])
        probe_ok = bool(
            np.array_equal(forbidden, expected)
            and all(
                torch.count_nonzero(cached[k][:6]).item() == 0
                for k in ("contact", "alignment", "speed")
            )
        )
        np.savez_compressed(
            a.output / "neural-probe.npz",
            initial_qpos=q,
            initial_qvel=v,
            observation=obs["actor"].cpu().numpy(),
            action=actions.cpu().numpy(),
            terminated=terminated.cpu().numpy(),
            truncated=truncated.cpu().numpy(),
            **{k: t.cpu().numpy() for k, t in snap.items()},
            **{"credit_" + k: t.cpu().numpy() for k, t in cached.items()},
        )
        if not probe_ok:
            raise ValueError("actual NN env.step forbidden-contact reward path failed")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FOOT_ONLY_SENSOR_REWARD_QUALIFIED_NOT_SKILL",
                input_file_hashes=pins,
                results=results,
                bindings=bindings,
                gpu_world_steps=480 + probe_steps,
                native_steps=480,
                nn_rows=neural_rows,
                optimizer_updates=0,
                model_hash=digest(a.output / "compiled_model.mjb"),
                probe_hash=digest(a.output / "neural-probe.npz"),
                probe_passed=probe_ok,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as exc:
        write_once(
            a.output / "failed.json",
            dict(
                status="FOOT_ONLY_QUALIFICATION_FAILED",
                error=f"{type(exc).__name__}: {exc}",
                results=results,
                fixture_counts=getattr(env, "_foot_only_fixture_counts", {}),
                nn_rows=neural_rows,
                neural_probe_gpu_world_steps=probe_steps,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
