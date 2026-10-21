"""Export TRAIN8-only reset states and exogenous plans; zero physics or NN calls.

These are ordinary randomized episode STARTS, not contact snapshots, experts,
successful demonstrations or new training. Contact experience must be collected
by a later uninterrupted native neural rollout, with all failures retained.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--ball-placement",
        choices=("legacy-subset-full-v0", "body-bound-v1"),
        default="legacy-subset-full-v0",
    )
    for k in ("reference-root", "qualification-root", "export-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("qualification-hash", "export-hash", "expected-head"):
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
        raise ValueError("new output and exact frozen source required")
    pins = {}
    for root, expected in (
        (a.qualification_root, a.qualification_hash),
        (a.export_root, a.export_hash),
    ):
        f = root / "complete.json"
        if digest(f) != expected:
            raise ValueError("actual immutable prerequisite receipt required")
        report = json.loads(f.read_text())
        pins.update(report["input_file_hashes"])
        pins[str(f)] = expected
        if root == a.qualification_root and (
            report.get("status") != "FOOT_ONLY_SENSOR_REWARD_QUALIFIED_NOT_SKILL"
            or report.get("probe_passed") is not True
        ):
            raise ValueError("actual foot-only sensor/reward qualification required")
        if root == a.export_root:
            pins[str(root / "compiled_model.mjb")] = report["model_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            source_split="TRAIN8_ONLY",
            ball_placement_protocol=a.ball_placement,
            worlds=64,
            seed=5601,
            maximum_physics_steps=0,
            maximum_neural_calls=0,
            optimizer_updates=0,
            frame_zero_starts_only=True,
            not_contact_bank_yet=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    try:
        import mujoco
        import numpy as np
        import torch

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from scripts.rsi_foot_only_calibration import body_bindings
        from scripts.rsi_mjlab_foot_only_backend import (
            FootOnlyMotionEnv,
            make_foot_only_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        cfg = make_foot_only_training_cfg(
            parameters=VelocityPolicyParameters.load(config, expected_hash=pins[str(config)]),
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=64,
            seed=5601,
            balanced_initial_sources=True,
            ball_placement_protocol=a.ball_placement,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        env = FootOnlyMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=5601)
        motion = env.command_manager.get_term("motion")
        if tuple(n + ".npz" for n in motion.motion.motion_names) != TRAIN_MOTION_NAMES or torch.any(
            motion.time_steps != 0
        ):
            raise ValueError("balanced TRAIN8 frame-zero starts only")
        if not np.array_equal(motion.motion_idx.cpu().numpy(), np.arange(64) % 8):
            raise ValueError("eight randomized resets per training source required")
        model = env.sim.mj_model
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        old = mujoco.MjModel.from_binary_path(str(a.export_root / "compiled_model.mjb"))
        if (
            layout.profile()
            != NativeVelocityLayout(
                old, robot_prefix="robot/", ball_body_name="ball/ball"
            ).profile()
        ):
            raise ValueError("source body/motor/collider profile changed")
        bindings = body_bindings(model, layout)
        ids = [motion.robot.joint_names.index(n) for n in G1_DDS_JOINT_NAMES]
        root_index = motion.cfg.body_names.index("pelvis")
        torso_index = motion.cfg.body_names.index("torso_link")
        if torso_index != motion.motion_anchor_body_index:
            raise ValueError("reference anchor must be actual torso")
        plan_hashes = {}
        for i, name in enumerate(TRAIN_MOTION_NAMES):
            length = int(motion.motion.file_lengths[i])
            if not 1 <= length <= 300:
                raise ValueError("bounded original full reference required")
            loader = motion.motion
            arrays = {
                "joint": loader.joint_pos[i, :length, ids],
                "joint_velocity": loader.joint_vel[i, :length, ids],
            }
            for label, index in (("pelvis", root_index), ("torso", torso_index)):
                for field, attribute in (
                    ("position", "body_pos_w"),
                    ("quaternion", "body_quat_w"),
                    ("linear", "body_lin_vel_w"),
                    ("angular", "body_ang_vel_w"),
                ):
                    arrays[label + "_" + field] = getattr(loader, attribute)[i, :length, index]
            values = {k: v.cpu().numpy().copy() for k, v in arrays.items()}
            # Validate exported HG order directly against actual actor reference channels.
            if not np.array_equal(values["joint"][0], obs["actor"][i, 98:127].cpu().numpy()):
                raise ValueError("exported reference joint order differs from measured actor ABI")
            path = a.output / (name.removesuffix(".npz") + "-plan.npz")
            np.savez_compressed(path, **values)
            plan_hashes[name] = digest(path)
        starts = a.output / "train-starts.npz"
        np.savez_compressed(
            starts,
            qpos=env.sim.data.qpos.cpu().numpy(),
            qvel=env.sim.data.qvel.cpu().numpy(),
            observation=obs["actor"].cpu().numpy(),
            motion_id=motion.motion_idx.cpu().numpy(),
            goal_world=(motion.target_destination_pos + env.scene.env_origins).cpu().numpy(),
            origins=env.scene.env_origins.cpu().numpy(),
            expected_foot=motion.kick_leg.cpu().numpy(),
        )
        if env._sim_step_counter != 0:
            raise ValueError("reset exporter must not advance physics")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="TRAIN8_RESET_PLANS_NOT_CONTACT_EXPERIENCE",
                input_file_hashes=pins,
                source_head=a.expected_head,
                starts_hash=digest(starts),
                plan_hashes=plan_hashes,
                worlds=64,
                seed=5601,
                source_split="train",
                ball_placement_protocol=a.ball_placement,
                bindings=bindings,
                physics_steps=0,
                neural_calls=0,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(exc),
                physics_steps=getattr(env, "_sim_step_counter", 0),
                neural_calls=0,
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
