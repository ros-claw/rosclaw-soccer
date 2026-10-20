"""Bounded GPU motion learning; a longer course requires a completed smoke."""

import argparse
import json
import random
import subprocess
import sys
from dataclasses import asdict
from importlib.metadata import distribution, version
from pathlib import Path


def cloned_tensors(state):
    return {k: v.detach().cpu().clone() for k, v in state.items() if hasattr(v, "detach")}


def changed_weight_names(before, after):
    import torch

    if set(before) != set(after) or any(
        not torch.isfinite(v).all() or v.shape != before[k].shape or v.dtype != before[k].dtype
        for k, v in after.items()
    ):
        raise ValueError("same finite learner state schema required")
    return [
        k
        for k in before
        if (k.endswith(".weight") or k.endswith(".bias")) and not torch.equal(before[k], after[k])
    ]


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("reference-root", "dataset-review", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-dataset-hash",
        "dataset-unit",
        "install-unit",
        "expected-sensor-hash",
    ):
        p.add_argument("--" + k, required=True)
    p.add_argument("--mode", choices=("smoke", "course"), default="smoke")
    p.add_argument("--num-envs", type=int, default=32)
    p.add_argument("--iterations", type=int, default=2)
    p.add_argument("--seed", type=int, default=501)
    p.add_argument("--prerequisite-smoke", type=Path)
    p.add_argument("--prerequisite-unit")
    p.add_argument("--prerequisite-hash")
    for name in (
        "stable-prior-root",
        "stable-pilot-root",
        "native-rehearsal-root",
        "policy-model",
        "policy-config",
    ):
        p.add_argument("--" + name, type=Path)
    for name in (
        "stable-prior-unit",
        "stable-prior-hash",
        "stable-pilot-unit",
        "stable-pilot-hash",
    ):
        p.add_argument("--" + name)
    a = p.parse_args()
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
        MotionLearningBudget,
        require_completed_motion_smoke,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    budget = MotionLearningBudget(a.mode, a.num_envs, a.iterations, a.seed)
    stable = a.stable_prior_root is not None
    prior = pilot = native_bank = None
    if stable:
        if not all(
            (
                a.stable_pilot_root,
                a.native_rehearsal_root,
                a.policy_model,
                a.policy_config,
                a.stable_prior_unit,
                a.stable_prior_hash,
                a.stable_pilot_unit,
                a.stable_pilot_hash,
            )
        ):
            raise ValueError(
                "complete immutable stable prior and physical pilot prerequisites required"
            )
        if a.mode != "smoke":
            raise ValueError("stable-prior first learner is a two-iteration smoke only")
        for root, unit, expected in (
            (a.stable_prior_root, a.stable_prior_unit, a.stable_prior_hash),
            (a.stable_pilot_root, a.stable_pilot_unit, a.stable_pilot_hash),
        ):
            wait_for_terminal_collection(unit)
            if digest(root / "complete.json") != expected:
                raise ValueError("stable prior prerequisite receipt changed")
        prior = json.loads((a.stable_prior_root / "complete.json").read_text())
        pilot = json.loads((a.stable_pilot_root / "complete.json").read_text())
        native_path = a.native_rehearsal_root / "complete.json"
        if digest(native_path) != prior["input_file_hashes"][str(native_path)]:
            raise ValueError("actual prior-qualified native rehearsal source required")
        native_bank = json.loads(native_path.read_text())
        if (
            prior["status"] != "TORCH_VELOCITY_PRIOR_INITIALIZATION_QUALIFIED_NOT_LEARNED_SKILL"
            or pilot["status"] != "MJLAB_VELOCITY_PRIOR_REFERENCE_PILOT_NOT_LEARNING_OR_FOOTBALL"
            or pilot["first_episode_summary"]["completed_count"] != 8
            or not native_bank["all_four_independent_full_repeats_exact"]
        ):
            raise ValueError(
                "qualified initialization, complete physical prior pilot and native bank required"
            )
    prerequisite = None
    if a.mode == "course":
        if not all((a.prerequisite_smoke, a.prerequisite_unit, a.prerequisite_hash)):
            raise ValueError("longer course requires hash-bound actual terminal smoke")
        wait_for_terminal_collection(a.prerequisite_unit)
        if digest(a.prerequisite_smoke / "complete.json") != a.prerequisite_hash:
            raise ValueError("prerequisite receipt hash differs")
        prerequisite = json.loads((a.prerequisite_smoke / "complete.json").read_text())
        require_completed_motion_smoke(prerequisite)

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and exact frozen learning smoke source required")
    # This job is launched only AFTER dependency installation is terminal; do
    # not run with an interpreter whose environment is being mutated by pip.
    wait_for_terminal_collection(a.install_unit)
    wait_for_terminal_collection(a.dataset_unit)
    if digest(a.dataset_review / "complete.json") != a.expected_dataset_hash:
        raise ValueError("actual full ten-source FK review required")
    review = json.loads((a.dataset_review / "complete.json").read_text())
    if (
        review["status"] != "MJLAB_STANDARD_MOTION_NUMERIC_FK_QUALIFIED"
        or review["source_motion_files"] != 10
        or not all(r["numeric_fk_qualified"] for r in review["results"])
        or subprocess.check_output(["git", "-C", str(a.reference_root), "status", "--porcelain"])
        or subprocess.check_output(
            ["git", "-C", str(a.reference_root), "rev-parse", "HEAD"], text=True
        ).strip()
        != review["reference_head"]
    ):
        raise ValueError("complete qualified data and pristine source reference required")
    pins = dict(review["input_file_hashes"])
    if stable:
        for receipt in (prior, pilot, native_bank):
            pins.update(receipt["input_file_hashes"])
        pins[str(a.stable_prior_root / "complete.json")] = a.stable_prior_hash
        pins[str(a.stable_pilot_root / "complete.json")] = a.stable_pilot_hash
        pins[str(a.stable_prior_root / "prior-actor-initialization.pt")] = prior["checkpoint_hash"]
        for case in range(4):
            folder = a.native_rehearsal_root / f"case{case}-repeat0"
            case_receipt = json.loads((folder / "complete.json").read_text())
            pins[str(folder / "native300.npz")] = case_receipt["raw_hash"]
    if prerequisite is not None:
        pins[str(a.prerequisite_smoke / "complete.json")] = a.prerequisite_hash
    pins[str(a.dataset_review / "complete.json")] = a.expected_dataset_hash
    pins.update(
        {
            str(f): digest(f)
            for dr in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src")
            for f in dr.rglob("*.py")
        }
    )
    check_pins(pins)
    # Bind the installed dependency source before any simulator import. A
    # locally repaired wheel must not masquerade as an unmodified version.
    sensor_path = Path(distribution("mujoco-warp").locate_file("mujoco_warp/_src/sensor.py"))
    if digest(sensor_path) != a.expected_sensor_hash:
        raise ValueError("exact declared MuJoCo-Warp sensor source required")
    for package in ("mjlab", "mujoco-warp", "rsl-rl-lib"):
        dist = distribution(package)
        for member in dist.files or ():
            if str(member).endswith(".py"):
                installed = Path(dist.locate_file(member)).resolve()
                pins[str(installed)] = digest(installed)
    check_pins(pins)
    packages = {
        name: version(name)
        for name in ("mjlab", "mujoco", "mujoco-warp", "warp-lang", "torch", "rsl-rl-lib", "numpy")
    }
    if prerequisite is not None and (
        packages != prerequisite["packages"]
        or any(
            prerequisite["input_file_hashes"].get(k) != v
            for k, v in pins.items()
            if k in review["input_file_hashes"] or "site-packages" in k
        )
    ):
        raise ValueError("course must use smoke-qualified data and dependency source bytes")
    if any(
        packages[k].split("+")[0] != v
        for k, v in {
            "mjlab": "1.2.0",
            "mujoco": "3.5.0",
            "mujoco-warp": "3.5.0",
            "torch": "2.13.0",
        }.items()
    ):
        raise ValueError(
            "separate explicitly pinned training stack required, not old native environment"
        )
    a.output.mkdir(parents=True)
    if stable:
        import numpy as np

        rehearsal_observations = []
        rehearsal_actions = []
        for case in range(4):
            with np.load(
                a.native_rehearsal_root / f"case{case}-repeat0/native300.npz", allow_pickle=False
            ) as raw:
                rehearsal_observations.append(raw["observation"].copy())
                rehearsal_actions.append(raw["action"].copy())
        np.savez_compressed(
            a.output / "native-rehearsal1200.npz",
            observation=np.concatenate(rehearsal_observations),
            action=np.concatenate(rehearsal_actions),
        )
        pins[str(a.output / "native-rehearsal1200.npz")] = digest(
            a.output / "native-rehearsal1200.npz"
        )
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            packages=packages,
            mode=budget.mode,
            num_envs=budget.num_envs,
            seed=budget.seed,
            physics_dt=0.002,
            decimation=10,
            screen_control_steps=8,
            ppo_iterations=budget.iterations,
            ppo_control_steps_per_iteration=24,
            maximum_gpu_world_substeps=budget.gpu_world_substeps,
            new_cpu_native_steps=0,
            new_policy_is_joint_position_with_pd_not_direct_torque=True,
            learner_from_scratch=not stable,
            old_weights_loaded=stable,
            stable_prior_rehearsal=stable,
            rehearsal_updates_per_ppo_iteration=1 if stable else 0,
            rehearsal_batch_size=256 if stable else 0,
            learning_rate=1e-4 if stable else 1e-3,
            initial_action_std=0.1 if stable else 1.0,
            publication_license_cleared=False,
            cpu_agreement_and_skill_validation_not_yet_proven=True,
            hardware_authorized=False,
        ),
    )
    env = None
    attempted_optimization = 0
    optimizer_counts = []
    try:
        import numpy as np
        import torch

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one explicitly isolated visible GPU required for first smoke")
        random.seed(budget.seed)
        np.random.seed(budget.seed)
        torch.manual_seed(budget.seed)
        torch.cuda.manual_seed_all(budget.seed)
        sys.path.insert(0, str(a.reference_root))
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg

        from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
            DEVELOPMENT_MOTION_NAMES,
            TRAIN_MOTION_NAMES,
        )
        from scripts.rsi_mjlab_motion_backend import (
            ExplicitBoundaryMotionEnv,
            make_motion_training_cfg,
        )

        cfg_factory = make_motion_training_cfg
        cfg_extra = {}
        if stable:
            from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
            from scripts.rsi_mjlab_velocity_prior_backend import make_velocity_prior_training_cfg

            params = VelocityPolicyParameters.load(
                a.policy_config, expected_hash=pins[str(a.policy_config)]
            )
            cfg_factory = make_velocity_prior_training_cfg
            cfg_extra = dict(
                parameters=params, command_mode="reference_velocity", balanced_initial_sources=True
            )
        cfg = cfg_factory(
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=budget.num_envs,
            seed=budget.seed,
            **cfg_extra,
        )
        if stable:
            cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = ExplicitBoundaryMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=budget.seed)
        motion = env.command_manager.get_term("motion")
        if tuple(name + ".npz" for name in motion.motion.motion_names) != TRAIN_MOTION_NAMES:
            raise ValueError("exact eight training sources required; development sources forbidden")
        m = env.sim.mj_model
        ball = m.geom("ball/ball_geom")
        mass = float(m.body_mass[m.geom_bodyid[ball.id]])
        if not np.isclose(ball.size[0], 0.11, atol=1e-12) or not np.isclose(mass, 0.43, atol=1e-12):
            raise ValueError("compiled ball must actually match declared dimensions")
        actor_shape = tuple(obs["actor"].shape)
        critic_shape = tuple(obs["critic"].shape)
        if actor_shape != (budget.num_envs, 168 if stable else 166) or critic_shape != (
            budget.num_envs,
            295,
        ):
            raise ValueError(f"explicit new observation ABI differs: {actor_shape}/{critic_shape}")
        rows = []
        initial_steps = env._sim_step_counter
        for _frame in range(8):
            before_phase = motion.time_steps.clone()
            before_id = motion.motion_idx.clone()
            before_length = motion.motion_length.clone()
            before_q = env.sim.data.qpos.detach().cpu().numpy().copy()
            obs, reward, terminated, truncated, _ = env.step(
                torch.zeros((budget.num_envs, 29), device="cuda:0")
            )
            torch.cuda.synchronize()
            done = (terminated | truncated).clone()
            expected_phase = torch.minimum(before_phase + 1, before_length - 1)
            changed = (motion.time_steps != expected_phase) | (motion.motion_idx != before_id)
            if torch.any(changed & ~done):
                raise ValueError("source change or resampling without explicit terminal boundary")
            if (
                any(not torch.isfinite(v).all() for v in obs.values())
                or not torch.isfinite(reward).all()
                or not torch.isfinite(env.sim.data.qpos).all()
                or not torch.isfinite(env.sim.data.qvel).all()
            ):
                raise ValueError("finite actual GPU observations and rewards required")
            rows.append(
                dict(
                    qpos_before=before_q,
                    qpos_after=env.sim.data.qpos.detach().cpu().numpy().copy(),
                    qvel_after=env.sim.data.qvel.detach().cpu().numpy().copy(),
                    terminated=terminated.detach().cpu().numpy().copy(),
                    truncated=truncated.detach().cpu().numpy().copy(),
                    source_phase=motion.time_steps.detach().cpu().numpy().copy(),
                )
            )
        if env._sim_step_counter - initial_steps != 80:
            raise ValueError("actual eight controls must execute eighty GPU simulation steps")
        np.savez_compressed(
            a.output / "screen8.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]}
        )
        write_once(
            a.output / "physics-screen.json",
            dict(
                control_steps=8,
                gpu_step_calls=80,
                gpu_world_substeps=80 * budget.num_envs,
                actor_shape=actor_shape,
                critic_shape=critic_shape,
                reset_boundaries_checked=True,
                actual_ball_radius_m=float(ball.size[0]),
                actual_ball_mass_kg=mass,
                new_cpu_native_steps=0,
                not_skill_verification=True,
            ),
        )
        print("MOTION_GPU_PHYSICS_SCREEN_COMPLETE", flush=True)
        agent_cfg = unitree_g1_soccer_ppo_runner_cfg()
        agent_cfg.logger = "tensorboard"
        agent_cfg.upload_model = False
        agent_cfg.save_interval = min(50, budget.iterations)
        agent_cfg.max_iterations = budget.iterations
        agent_cfg.num_steps_per_env = 24
        agent_cfg.seed = budget.seed
        if stable:
            agent_cfg.actor.class_name = "scripts.rsi_mjlab_prior_rl:VelocityPriorMLPModel"
            agent_cfg.actor.obs_normalization = False
            agent_cfg.actor.distribution_cfg["init_std"] = 0.1
            agent_cfg.algorithm.class_name = "scripts.rsi_mjlab_prior_rl:RehearsalPPO"
            agent_cfg.algorithm.learning_rate = 1e-4
        # This is a plumbing smoke, not a selected learning-rate/std sweep.
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
        runner_cfg = asdict(agent_cfg)
        if stable:
            runner_cfg["actor"].update(
                prior_model=str(a.policy_model),
                prior_model_hash=pins[str(a.policy_model)],
                prior_checkpoint=str(a.stable_prior_root / "prior-actor-initialization.pt"),
                prior_checkpoint_hash=prior["checkpoint_hash"],
            )
            runner_cfg["algorithm"].update(
                rehearsal_path=str(a.output / "native-rehearsal1200.npz"),
                rehearsal_hash=pins[str(a.output / "native-rehearsal1200.npz")],
                rehearsal_weight=1.0,
                rehearsal_batch_size=256,
            )
        runner = MjlabOnPolicyRunner(wrapped, runner_cfg, str(a.output / "rsl"), "cuda:0")
        if stable:
            from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor

            original_prior = VelocityPriorActor.from_onnx(
                a.policy_model, expected_hash=pins[str(a.policy_model)]
            ).to("cuda:0")
            with torch.no_grad():
                actual = runner.alg.actor.mlp(obs["actor"])
                expected = original_prior(obs["actor"])
                if not torch.equal(actual, expected):
                    raise ValueError(
                        "actual RSL learner must start from qualified prior, not random weights"
                    )
            optimizer_ids = {id(p) for g in runner.alg.optimizer.param_groups for p in g["params"]}
            if not all(id(p) in optimizer_ids for p in runner.alg.actor.mlp.parameters()):
                raise ValueError("optimizer must own actual initialized actor parameters")
        start_state = runner.alg.save()
        actor0 = cloned_tensors(start_state["actor_state_dict"])
        critic0 = cloned_tensors(start_state["critic_state_dict"])
        optimizers = [o for o in vars(runner.alg).values() if isinstance(o, torch.optim.Optimizer)]
        optimizers = list({id(o): o for o in optimizers}.values())
        if not optimizers:
            raise ValueError("actual actor/critic optimizers required")
        for opt in optimizers:
            counts = dict(attempted=0, completed=0)
            original = opt.step

            def counted_step(*args, _step=original, _counts=counts, **kwargs):
                _counts["attempted"] += 1
                result = _step(*args, **kwargs)
                _counts["completed"] += 1
                return result

            opt.step = counted_step
            optimizer_counts.append(counts)
        before = env._sim_step_counter
        attempted_optimization = budget.iterations
        runner.learn(num_learning_iterations=budget.iterations, init_at_random_ep_len=False)
        torch.cuda.synchronize()
        final_state = runner.alg.save()
        if stable and (
            runner.alg.rehearsal_updates != budget.iterations
            or runner.alg.rehearsal_attempted_updates != budget.iterations
            or runner.alg.rehearsal_rows != budget.iterations * 256
            or torch.count_nonzero(runner.alg.actor.mlp.mlp[0].weight[:, 98:]) == 0
            or not torch.equal(
                actor0["mlp.prior_mean"],
                final_state["actor_state_dict"]["mlp.prior_mean"].cpu(),
            )
            or not torch.equal(
                actor0["mlp.prior_denominator"],
                final_state["actor_state_dict"]["mlp.prior_denominator"].cpu(),
            )
        ):
            raise ValueError(
                "real reference learning, counted rehearsal and fixed normalization required"
            )
        actor_changes = changed_weight_names(
            actor0, cloned_tensors(final_state["actor_state_dict"])
        )
        critic_changes = changed_weight_names(
            critic0, cloned_tensors(final_state["critic_state_dict"])
        )
        if (
            not actor_changes
            or not critic_changes
            or not all(
                c["completed"] > 0 and c["completed"] == c["attempted"] for c in optimizer_counts
            )
            or env._sim_step_counter - before != budget.training_control_steps * 10
        ):
            raise ValueError(
                "real actor AND critic weight changes and declared GPU step budget required"
            )
        checkpoint = dict(
            algorithm=final_state,
            initial_actor_state_dict=actor0,
            initial_critic_state_dict=critic0,
            iteration=budget.iterations,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
            python_rng=random.getstate(),
            numpy_rng=dict(
                generator=np.random.get_state()[0],
                keys=np.random.get_state()[1].tolist(),
                position=np.random.get_state()[2],
                has_gaussian=np.random.get_state()[3],
                gaussian=np.random.get_state()[4],
            ),
            source_sampler=motion.bin_failed_count.detach().cpu(),
            common_step_counter=env.common_step_counter,
            source_head=a.expected_head,
            training_source_names=list(TRAIN_MOTION_NAMES),
            development_source_names=list(DEVELOPMENT_MOTION_NAMES),
            stable_prior_rehearsal=stable,
            actor_observation_dim=168 if stable else 166,
            prior_configuration=runner_cfg if stable else None,
        )
        path = a.output / f"learner-update{budget.iterations}.pt"
        torch.save(checkpoint, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        for part in ("actor_state_dict", "critic_state_dict"):
            if any(
                not torch.equal(value, loaded["algorithm"][part][key])
                for key, value in cloned_tensors(final_state[part]).items()
            ):
                raise ValueError("safe actor/critic checkpoint reload differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED"
                    if stable
                    else (
                        "MJLAB_MOTION_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED"
                        if budget.mode == "smoke"
                        else "MJLAB_MOTION_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED"
                    )
                ),
                source_head=a.expected_head,
                mode=budget.mode,
                num_envs=budget.num_envs,
                iterations=budget.iterations,
                seed=budget.seed,
                input_file_hashes=pins,
                packages=packages,
                gpu_step_calls=env._sim_step_counter - initial_steps,
                gpu_world_substeps=(env._sim_step_counter - initial_steps) * budget.num_envs,
                policy_control_samples=budget.training_control_steps * budget.num_envs,
                new_optimizer_steps=sum(c["completed"] for c in optimizer_counts),
                optimizer_counts=optimizer_counts,
                stable_prior_rehearsal=stable,
                native_rehearsal=final_state.get("native_rehearsal"),
                changed_actor_weights=actor_changes,
                changed_critic_weights=critic_changes,
                checkpoint_hash=digest(path),
                weights_only_checkpoint_reload=True,
                full_training_resume_and_body_restoration_proven=False,
                new_cpu_native_steps=0,
                new_policy_is_joint_position_with_pd_not_direct_torque=True,
                football_skill_proven=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                optimizer_counts=optimizer_counts,
                attempted_learning_iterations=attempted_optimization,
                attempted_gpu_step_calls=env._sim_step_counter if env is not None else 0,
                new_cpu_native_steps=0,
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
