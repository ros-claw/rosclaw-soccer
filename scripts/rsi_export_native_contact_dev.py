"""Export disjoint public DEV007/010 body-bound starts; zero physics/NN/gradients.

Data are evaluation-only. Never mix these files into TRAIN reset pools or PPO.
No candidate checkpoint participates in constructing these exogenous tasks.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("course-proof-root", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("course-proof-hash", "expected-head"):
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
        raise ValueError("frozen exporter and unused output required")
    f = a.course_proof_root / "complete.json"
    if digest(f) != a.course_proof_hash:
        raise ValueError("immutable foundational course proof required")
    proof = json.loads(f.read_text())
    if proof.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL":
        raise ValueError("complete original body/course proof required")
    pins = {**proof["input_file_hashes"], str(f): a.course_proof_hash}
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    for f in (a.reference_root / "src/assets/soccer/motions/shooter").glob("*.npz"):
        pins[str(f)] = digest(f)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            sources=["soccer-standard-007_left.npz", "soccer-standard-010_right.npz"],
            worlds_per_source=32,
            seed_by_source=[7603, 7604],
            ball_placement_protocol="body-bound-v1",
            source_split="DEV evaluation only",
            candidate_checkpoint_used=False,
            maximum_physics_steps=0,
            maximum_policy_rows=0,
            optimizer_updates=0,
            no_body_or_motor_or_ball_dimension_changes=True,
            promotion_authorized=False,
            hardware_authorized=False,
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
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import DEVELOPMENT_MOTION_NAMES
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from scripts.rsi_mjlab_foot_only_backend import (
            FootOnlyMotionEnv,
            make_foot_only_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required for reference loading/reset only")

        def pinned(suffix):
            paths = [Path(k) for k in pins if k.endswith(suffix)]
            if len(paths) != 1:
                raise ValueError("one immutable input required: " + suffix)
            return paths[0]

        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        old_model_path = pinned(
            "/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb"
        )
        old_model = mujoco.MjModel.from_binary_path(str(old_model_path))
        old_profile = NativeVelocityLayout(
            old_model, robot_prefix="robot/", ball_body_name="ball/ball"
        ).profile()
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        results = []
        for i, name in enumerate(DEVELOPMENT_MOTION_NAMES):
            seed = 7603 + i
            cfg = make_foot_only_training_cfg(
                parameters=params,
                command_mode="reference_position_velocity",
                command_point="pelvis",
                motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
                num_envs=32,
                seed=seed,
                balanced_initial_sources=False,
                motion_glob=name,
                ball_placement_protocol="body-bound-v1",
            )
            cfg.commands["motion"].sampling_mode = "uniform"
            env = FootOnlyMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
            observation, _ = env.reset(seed=seed)
            motion = env.command_manager.get_term("motion")
            if (
                tuple(n + ".npz" for n in motion.motion.motion_names) != (name,)
                or torch.any(motion.time_steps != 0)
                or torch.any(motion.motion_idx != 0)
            ):
                raise ValueError("all and only declared DEV source frame zero required")
            if not torch.all(motion.kick_leg.long() == (0 if i == 0 else 1)):
                raise ValueError("source foot intent changed")
            layout = NativeVelocityLayout(
                env.sim.mj_model, robot_prefix="robot/", ball_body_name="ball/ball"
            )
            if layout.profile() != old_profile:
                raise ValueError("DEV cannot change qualified physical body/motor/collider profile")
            folder = a.output / name.removesuffix(".npz")
            folder.mkdir()
            loader = motion.motion
            length = int(loader.file_lengths[0])
            joint_ids = [motion.robot.joint_names.index(n) for n in G1_DDS_JOINT_NAMES]
            plan = dict(
                joint=loader.joint_pos[0, :length, joint_ids],
                joint_velocity=loader.joint_vel[0, :length, joint_ids],
            )
            for label, body in (("pelvis", "pelvis"), ("torso", "torso_link")):
                index = motion.cfg.body_names.index(body)
                for field, attr in (
                    ("position", "body_pos_w"),
                    ("quaternion", "body_quat_w"),
                    ("linear", "body_lin_vel_w"),
                    ("angular", "body_ang_vel_w"),
                ):
                    plan[label + "_" + field] = getattr(loader, attr)[0, :length, index]
            plan = {k: v.cpu().numpy().copy() for k, v in plan.items()}
            if not np.array_equal(plan["joint"][0], observation["actor"][0, 98:127].cpu().numpy()):
                raise ValueError("full DEV reference HG order differs from actual177 observation")
            np.savez_compressed(folder / "plan.npz", **plan)
            np.savez_compressed(
                folder / "starts.npz",
                qpos=env.sim.data.qpos.cpu().numpy(),
                qvel=env.sim.data.qvel.cpu().numpy(),
                observation=observation["actor"].cpu().numpy(),
                goal_world=(motion.target_destination_pos + env.scene.env_origins).cpu().numpy(),
                origins=env.scene.env_origins.cpu().numpy(),
                expected_foot=motion.kick_leg.cpu().numpy(),
            )
            if env._sim_step_counter != 0:
                raise ValueError("DEV task exporter unexpectedly advanced physics")
            results.append(
                dict(
                    source=name,
                    seed=seed,
                    worlds=32,
                    controls=length,
                    plan_hash=digest(folder / "plan.npz"),
                    starts_hash=digest(folder / "starts.npz"),
                )
            )
            env.close()
            env = None
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_BODY_BOUND_DEV_TASKS_NOT_POLICY_EVIDENCE",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                source_split="DEV evaluation only",
                ball_placement_protocol="body-bound-v1",
                physics_steps=0,
                policy_rows=0,
                optimizer_updates=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            physics_steps=0,
            policy_rows=0,
            optimizer_updates=0,
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
