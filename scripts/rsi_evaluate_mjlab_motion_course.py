"""Paired initial/final neural motion tracking on two public DEV sources.

No learning, physical-body edits during steps, football promotion or video.
Every role/state/physics comparison after this remains an independent exam.
"""

import argparse
import copy
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("course-root", "reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("course-unit", "course-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--control", choices=("neural", "reference-pd"), default="neural")
    a = p.parse_args()
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import motion_checkpoint_filename
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_mjlab_motion_learning_smoke import changed_weight_names, cloned_tensors
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen evaluator and unused external output required")
    wait_for_terminal_collection(a.course_unit)
    course_path = a.course_root / "complete.json"
    if digest(course_path) != a.course_hash:
        raise ValueError("declared actual course receipt required")
    course = json.loads(course_path.read_text())
    ball_aware = course["status"] in (
        "BALL_AWARE_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED",
        "BALL_AWARE_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED",
    )
    stable = course["status"] in (
        "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED",
        "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_COURSE_NOT_SKILL_QUALIFIED",
        "MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED",
        "BALL_AWARE_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED",
        "BALL_AWARE_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED",
    )
    position_gain = course.get("velocity_command_position_feedback_gain", 0.0)
    if position_gain not in (0.0, 1.0):
        raise ValueError("declared position-command protocol required")
    command_point = course.get("velocity_command_anatomical_point", "torso")
    if command_point not in ("torso", "pelvis"):
        raise ValueError("declared anatomical command point required")
    if (
        course["status"]
        not in (
            "MJLAB_MOTION_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED",
            "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED",
            "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_COURSE_NOT_SKILL_QUALIFIED",
            "MJLAB_BOUNDED_REFERENCE_IMITATION_NOT_SKILL_QUALIFIED",
            "BALL_AWARE_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED",
            "BALL_AWARE_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED",
        )
        or course["new_optimizer_steps"] <= 0
        or not course["weights_only_checkpoint_reload"]
        or course["football_skill_proven"]
        or course["hardware_authorized"]
    ):
        raise ValueError("actual completed non-promoted motion course required")
    if stable and a.control != "neural":
        raise ValueError("stable-prior evaluation is an initial/final neural comparison")
    if ball_aware:
        from rosclaw_soccer.rsi.ball_aware_inputs import require_ball_aware_learning

        require_ball_aware_learning(course)
    pins = dict(course["input_file_hashes"])
    pins[str(course_path)] = a.course_hash
    checkpoint_path = a.course_root / motion_checkpoint_filename(course)
    pins[str(checkpoint_path)] = course["checkpoint_hash"]
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        for source in directory.rglob("*.py"):
            pins[str(source)] = digest(source)
    check_pins(pins)
    force_decoder = None
    if ball_aware:
        import torch

        from rosclaw_soccer.rsi.velocity_command_protocol import (
            ball_aware_checkpoint_protocol,
            ball_aware_force_protocol,
        )

        metadata = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        protocol = ball_aware_checkpoint_protocol(course, metadata)
        force_decoder = ball_aware_force_protocol(course, metadata)
        if force_decoder == "addressed":
            relative = "scripts/rsi_mjwarp_addressed_contact_force.py"
            qualified_hashes = {
                v for k, v in course["input_file_hashes"].items() if k.endswith("/" + relative)
            }
            if qualified_hashes != {digest(Path.cwd() / relative)}:
                raise ValueError("GPU evaluation must use the actual qualified force decoder")
            from scripts.rsi_mjwarp_addressed_contact_force import install

            install()
        command_point = protocol["velocity_command_anatomical_point"]
        position_gain = protocol["velocity_command_position_feedback_gain"]
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            num_envs=32 if a.control == "neural" else 64,
            control=a.control,
            horizon=300,
            reset_seeds=[1501, 1502] if a.control == "neural" else [1601],
            first_episode_only=True,
            stable_prior_rehearsal=stable,
            ball_aware=ball_aware,
            force_decoder=force_decoder,
            gpu_contact_labels_qualified=force_decoder == "addressed",
            actor_observation_dim=177 if ball_aware else 168 if stable else 166,
            command_protocol_read_from_hash_bound_checkpoint=ball_aware,
            velocity_command_position_feedback_gain=position_gain,
            velocity_command_anatomical_point=command_point,
            mode="SIM_ONLY",
            max_gpu_world_substeps=384000,
            new_optimizer_steps=0,
            public_source_development_not_fresh_world=True,
            football_skill_proven=False,
            hardware_authorized=False,
        ),
    )
    env = None
    steps_total = 0
    neural_samples = 0
    rows = []
    try:
        import numpy as np
        import torch
        from tensordict import TensorDict

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        sys.path.insert(0, str(a.reference_root))
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg

        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
            DEVELOPMENT_MOTION_NAMES,
            TRAIN_MOTION_GLOB,
            TRAIN_MOTION_NAMES,
        )
        from rosclaw_soccer.rsi.mjlab_motion_evaluation import MotionFirstEpisode
        from scripts.rsi_mjlab_motion_backend import (
            ExplicitBoundaryMotionEnv,
            make_motion_training_cfg,
        )

        initial, final = None, None
        if a.control == "neural":
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
            initial = checkpoint["initial_actor_state_dict"]
            final = checkpoint["algorithm"]["actor_state_dict"]
            if not changed_weight_names(initial, cloned_tensors(final)):
                raise ValueError("actual initial and final neural networks required")
        summaries = []
        source_patterns = (
            DEVELOPMENT_MOTION_NAMES if a.control == "neural" else (TRAIN_MOTION_GLOB,)
        )
        n = 32 if a.control == "neural" else 64
        arms = (
            (("initial", initial), ("final", final))
            if a.control == "neural"
            else (("default", None), ("reference", None))
        )
        for source_index, source in enumerate(source_patterns):
            seed = 1501 + source_index if a.control == "neural" else 1601
            initial_state = None
            for arm, state in arms:
                rows = []
                cfg_factory = make_motion_training_cfg
                env_class = ExplicitBoundaryMotionEnv
                cfg_extra = {}
                if stable:
                    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
                    from scripts.rsi_mjlab_velocity_prior_backend import (
                        make_velocity_prior_training_cfg,
                    )

                    # The source checkpoint commits its actual initialization
                    # paths and hashes. No template/random network substitution.
                    actor_config = copy.deepcopy(checkpoint["prior_configuration"]["actor"])
                    config_path = next(
                        Path(k)
                        for k in pins
                        if k.endswith("/policy/velocity/v0/params/deploy.yaml")
                    )
                    params = VelocityPolicyParameters.load(
                        config_path, expected_hash=pins[str(config_path)]
                    )
                    cfg_factory = make_velocity_prior_training_cfg
                    cfg_extra = dict(
                        parameters=params,
                        command_mode="reference_position_velocity"
                        if position_gain
                        else "reference_velocity",
                        command_point=command_point,
                    )
                    if ball_aware:
                        from scripts.rsi_mjlab_ball_aware_backend import (
                            BallAwareMotionEnv,
                            make_ball_aware_training_cfg,
                        )

                        cfg_factory = make_ball_aware_training_cfg
                        env_class = BallAwareMotionEnv
                        cfg_extra["football_rewards"] = True
                cfg = cfg_factory(
                    motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
                    num_envs=n,
                    seed=seed,
                    motion_glob=source,
                    balanced_initial_sources=a.control == "reference-pd",
                    **cfg_extra,
                )
                # uniform loads the actual source frame0 body with reset perturbations;
                # the template's "start" mode only resets time and is NOT appropriate.
                cfg.commands["motion"].sampling_mode = "uniform"
                cfg.sim.nan_guard.enabled = True
                env = env_class(cfg=cfg, device="cuda:0", render_mode=None)
                agent_cfg = unitree_g1_soccer_ppo_runner_cfg()
                policy, loaded_state = None, None
                if a.control == "neural":
                    if stable:
                        from scripts.rsi_mjlab_prior_rl import VelocityPriorMLPModel

                        actor_config.pop("class_name", None)
                        # RSL5 pops the distribution class from the dict in
                        # construction. This versioned smoke used Gaussian29
                        # with scalar std; reconstruct that declared class only.
                        actor_config["distribution_cfg"]["class_name"] = "GaussianDistribution"
                        before_obs, _ = env.reset(seed=seed)
                        model_class = VelocityPriorMLPModel
                        if ball_aware:
                            from scripts.rsi_mjlab_ball_aware_rl import BallAwarePriorMLPModel

                            model_class = BallAwarePriorMLPModel
                        policy = model_class(
                            TensorDict(before_obs, batch_size=[n]),
                            {"actor": ["actor"]},
                            "actor",
                            29,
                            **actor_config,
                        ).to("cuda:0")
                        policy.eval()
                    else:
                        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                        runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), None, "cuda:0")
                        policy = runner.get_inference_policy()
                    policy.load_state_dict(state, strict=True)
                    loaded_state = cloned_tensors(policy.state_dict())
                obs, _ = env.reset(seed=seed)
                motion = env.command_manager.get_term("motion")
                expected_names = (
                    [source.removesuffix(".npz")]
                    if a.control == "neural"
                    else [name.removesuffix(".npz") for name in TRAIN_MOTION_NAMES]
                )
                if motion.motion.motion_names != expected_names or torch.any(
                    motion.time_steps != 0
                ):
                    raise ValueError("single declared DEV source from actual frame0 required")
                snapshot = dict(
                    qpos=env.sim.data.qpos.cpu().numpy().copy(),
                    qvel=env.sim.data.qvel.cpu().numpy().copy(),
                    observation=obs["actor"].cpu().numpy().copy(),
                    phase=motion.time_steps.cpu().numpy().copy(),
                    motion_id=motion.motion_idx.cpu().numpy().copy(),
                )
                if initial_state is None:
                    initial_state = snapshot
                elif any(not np.array_equal(snapshot[k], initial_state[k]) for k in snapshot):
                    raise ValueError("initial/final arms must start from exactly identical states")
                np.savez_compressed(a.output / f"{source_index}-{arm}-initial.npz", **snapshot)
                action_term = env.action_manager.get_term("joint_pos")
                if tuple(action_term.target_names) != G1_DDS_JOINT_NAMES or not torch.equal(
                    action_term.target_ids, torch.arange(29, device="cuda:0")
                ):
                    raise ValueError("actual action targets must match qualified HG joint order")
                if a.control == "reference-pd":
                    if not np.array_equal(snapshot["motion_id"], np.arange(n) % 8):
                        raise ValueError("balanced eight-source initial resets required")
                    write_once(
                        a.output / f"{source_index}-{arm}-codec.json",
                        dict(
                            target_names=action_term.target_names,
                            target_ids=action_term.target_ids.cpu().tolist(),
                            scale=torch.as_tensor(action_term.scale).cpu().tolist(),
                            offset=torch.as_tensor(action_term.offset).cpu().tolist(),
                            defaults_match=bool(
                                torch.equal(action_term.offset, motion.robot.data.default_joint_pos)
                            ),
                            zero_neural_or_optimizer_calls=True,
                        ),
                    )
                score = MotionFirstEpisode(n)
                counter = env._sim_step_counter
                with torch.inference_mode():
                    for _frame in range(300):
                        raw = dict(
                            qpos_before=env.sim.data.qpos.cpu().numpy().copy(),
                            qvel_before=env.sim.data.qvel.cpu().numpy().copy(),
                            observation=obs["actor"].cpu().numpy().copy(),
                            source_phase=motion.time_steps.cpu().numpy().copy(),
                            active_before=score.active.copy(),
                            motion_id=motion.motion_idx.cpu().numpy().copy(),
                            reference_joint=motion.joint_pos.cpu().numpy().copy(),
                            measured_joint=motion.robot.data.joint_pos.cpu().numpy().copy(),
                            encoder_bias=motion.robot.data.encoder_bias.cpu().numpy().copy(),
                            reference_anchor=motion.anchor_pos_w.cpu().numpy().copy(),
                            measured_anchor=motion.robot_anchor_pos_w.cpu().numpy().copy(),
                            reference_anchor_quat=motion.anchor_quat_w.cpu().numpy().copy(),
                            reference_anchor_linear_velocity=motion.anchor_lin_vel_w.cpu()
                            .numpy()
                            .copy(),
                            reference_anchor_angular_velocity=motion.anchor_ang_vel_w.cpu()
                            .numpy()
                            .copy(),
                            reference_joint_velocity=motion.joint_vel.cpu().numpy().copy(),
                        )
                        if ball_aware:
                            raw["goal_target_world"] = (
                                (motion.target_destination_pos + env.scene.env_origins)
                                .cpu()
                                .numpy()
                                .copy()
                            )
                            raw["expected_kicking_foot"] = motion.kick_leg.cpu().numpy().copy()
                        if a.control == "neural":
                            actions = policy(
                                TensorDict(obs, batch_size=[n]), stochastic_output=False
                            )
                            neural_samples += n
                        elif arm == "default":
                            actions = torch.zeros((n, 29), device="cuda:0")
                        else:
                            actions = (motion.joint_pos - action_term.offset) / action_term.scale
                        if not torch.isfinite(actions).all():
                            raise ValueError("finite deterministic neural actions required")
                        raw["action"] = actions.cpu().numpy().copy()
                        expected_target = actions * action_term.scale + action_term.offset
                        raw["requested_joint_target"] = expected_target.cpu().numpy().copy()
                        applied = (
                            actions
                            if agent_cfg.clip_actions is None
                            else torch.clamp(
                                actions, -agent_cfg.clip_actions, agent_cfg.clip_actions
                            )
                        )
                        obs, reward, term, trunc, _ = env.step(applied)
                        torch.cuda.synchronize()
                        steps_total += 10
                        raw["processed_joint_target"] = (
                            action_term._processed_actions.cpu().numpy().copy()
                        )
                        raw["position_target_after_step"] = (
                            motion.robot.data.joint_pos_target.cpu().numpy().copy()
                        )
                        ongoing = ~(term | trunc)
                        if not torch.allclose(
                            motion.robot.data.joint_pos_target[ongoing],
                            (action_term._processed_actions - motion.robot.data.encoder_bias)[
                                ongoing
                            ],
                            atol=1e-6,
                            rtol=1e-6,
                        ):
                            raise ValueError(
                                "non-reset bodies must receive the actual decoded PD target"
                            )
                        if a.control == "reference-pd" and (
                            agent_cfg.clip_actions is not None
                            or not torch.allclose(
                                action_term._processed_actions,
                                expected_target,
                                atol=1e-6,
                                rtol=1e-6,
                            )
                        ):
                            raise ValueError(
                                "reference decoding must match the actual processed PD target"
                            )
                        if (
                            not torch.isfinite(env.sim.data.qpos).all()
                            or not torch.isfinite(env.sim.data.qvel).all()
                        ):
                            raise ValueError("finite actual body state required")
                        raw.update(
                            qpos_after=env.sim.data.qpos.cpu().numpy().copy(),
                            qvel_after=env.sim.data.qvel.cpu().numpy().copy(),
                            reward=reward.cpu().numpy().copy(),
                            terminated=term.cpu().numpy().copy(),
                            truncated=trunc.cpu().numpy().copy(),
                            reference_exhausted=env.termination_manager.get_term(
                                "reference_exhausted"
                            )
                            .cpu()
                            .numpy()
                            .copy(),
                        )
                        if ball_aware:
                            physical = env.physical_step_data()
                            if not np.array_equal(
                                physical["terminal_capture_mask"].cpu().numpy(),
                                (term | trunc).cpu().numpy(),
                            ):
                                raise ValueError(
                                    "terminal actual states/contact histories must precede reset"
                                )
                            raw.update(
                                {key: value.cpu().numpy().copy() for key, value in physical.items()}
                            )
                        score.observe(
                            raw["reward"],
                            raw["terminated"],
                            raw["truncated"],
                            raw["reference_exhausted"],
                        )
                        rows.append(raw)
                        if not score.active.any():
                            break
                if env._sim_step_counter - counter != len(rows) * 10:
                    raise ValueError("actual GPU physics counter differs")
                if a.control == "neural" and any(
                    not torch.equal(v, loaded_state[k])
                    for k, v in cloned_tensors(policy.state_dict()).items()
                ):
                    raise ValueError("inference must not update model or normalization state")
                path = a.output / f"{source_index}-{arm}-raw.npz"
                np.savez_compressed(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
                summary = dict(
                    source=source,
                    arm=arm,
                    seed=seed,
                    raw_hash=digest(path),
                    batched_control_steps=len(rows),
                    gpu_world_substeps=len(rows) * 10 * n,
                    initial_phase_zero=True,
                    **score.summary(),
                )
                active = np.asarray([r["active_before"] for r in rows])
                reference = np.asarray([r["reference_joint"] for r in rows])
                measured = np.asarray([r["measured_joint"] for r in rows])
                anchor_delta = np.asarray(
                    [r["reference_anchor"] - r["measured_anchor"] for r in rows]
                )
                summary["mean_active_joint_abs_error_rad"] = float(
                    np.abs(reference - measured)[active].mean()
                )
                summary["mean_active_anchor_xy_error_m"] = float(
                    np.linalg.norm(anchor_delta[:, :, :2], axis=2)[active].mean()
                )
                # This compiled layout is ball-first. Inspect it explicitly
                # before reporting displacement; no reset jump is counted.
                from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

                layout = NativeVelocityLayout(
                    env.sim.mj_model, robot_prefix="robot/", ball_body_name="ball/ball"
                )
                ball_motion = []
                for world, count in enumerate(score.steps):
                    ball_q = np.asarray(
                        [
                            r["qpos_before"][world, layout.bq : layout.bq + 3]
                            for r in rows[: int(count)]
                        ]
                    )
                    if ball_aware:
                        # Include the actual final physical position, never the
                        # post-reset position that happens to share its tick.
                        ball_q = np.concatenate(
                            (
                                ball_q[:1],
                                np.asarray(
                                    [
                                        r["qpos_after"][world, layout.bq : layout.bq + 3]
                                        for r in rows[: int(count)]
                                    ]
                                ),
                            ),
                            axis=0,
                        )
                    ball_motion.append(
                        float(np.linalg.norm(ball_q[:, :2] - ball_q[0, :2], axis=1).max())
                    )
                summary["first_episode_ball_max_xy_displacement_m"] = ball_motion
                if ball_aware:
                    intended_contact, intended_first, sampled_speed = [], [], []
                    for world, count in enumerate(score.steps):
                        force = np.asarray(
                            [r["foot_force_history"][world] for r in rows[: int(count)]]
                        ).reshape(-1, 2)
                        expected_side = int(rows[0]["expected_kicking_foot"][world])
                        hit = force > 10.0
                        intended = np.flatnonzero(hit[:, expected_side])
                        any_hit = np.flatnonzero(hit.any(axis=1))
                        intended_contact.append(bool(len(intended)))
                        intended_first.append(
                            bool(
                                len(any_hit)
                                and hit[any_hit[0]].sum() == 1
                                and hit[any_hit[0], expected_side]
                            )
                        )
                        speed = np.linalg.norm(
                            np.asarray(
                                [
                                    r["qvel_after"][world, layout.bv : layout.bv + 3]
                                    for r in rows[: int(count)]
                                ]
                            ),
                            axis=1,
                        )
                        sampled_speed.append(
                            float(speed[intended[0] // 10 :].max()) if len(intended) else None
                        )
                    summary.update(
                        intended_foot_contact_detected=intended_contact,
                        intended_foot_first=intended_first,
                        sampled_max_ball_speed_after_intended_contact_mps=sampled_speed,
                        force_threshold_n=10.0,
                        terminal_physical_state_preserved=True,
                        later_speed_is_not_causal_kick_attribution=True,
                    )
                write_once(a.output / f"{source_index}-{arm}.json", summary)
                summaries.append(summary)
                print(json.dumps(summary, allow_nan=False), flush=True)
                env.close()
                env = None
                rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "BALL_AWARE_PAIRED_DEV_NOT_FOOTBALL_QUALIFIED"
                    if ball_aware
                    else "MJLAB_PAIRED_DEV_MOTION_TRACKING_NOT_FOOTBALL_QUALIFIED"
                    if a.control == "neural"
                    else "MJLAB_REFERENCE_PD_AUTHORITY_NOT_NEURAL_LEARNING"
                ),
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=summaries,
                gpu_step_calls=steps_total,
                gpu_world_substeps=steps_total * n,
                actual_neural_control_samples=neural_samples,
                new_optimizer_steps=0,
                all_initial_states_paired_exact=True,
                first_episode_only=True,
                public_source_development_not_fresh_world=True,
                evaluated_source_split="dev" if a.control == "neural" else "train",
                stable_prior_rehearsal=stable,
                ball_aware=ball_aware,
                force_decoder=force_decoder,
                gpu_contact_labels_qualified=force_decoder == "addressed",
                actor_observation_dim=177 if ball_aware else 168 if stable else 166,
                velocity_command_position_feedback_gain=position_gain,
                velocity_command_anatomical_point=command_point,
                football_skill_proven=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "partial-failed-arm.npz",
                **{k: np.array([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                gpu_step_calls=steps_total,
                active_env_gpu_step_counter=env._sim_step_counter if env is not None else None,
                actual_neural_control_samples=neural_samples,
                new_optimizer_steps=0,
                hardware_authorized=False,
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
