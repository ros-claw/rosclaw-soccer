"""Qualified ball-aware PPO smoke or fixed full-horizon same-ABI continuation."""

import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("resume-smoke", "dev", "retention"):
        p.add_argument("--" + key + "-root", type=Path)
        p.add_argument("--" + key + "-unit")
        p.add_argument("--" + key + "-hash")
    for key in ("qualification-root", "rehearsal-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in (
        "qualification-unit",
        "qualification-hash",
        "rehearsal-unit",
        "rehearsal-hash",
        "expected-head",
    ):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.ball_aware_learning_budget import BallAwareLearningBudget
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or (
            subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        )
    ):
        raise ValueError("unused output and frozen learner source required")
    pins = {}
    receipts = []
    for root, unit, expected in (
        (a.qualification_root, a.qualification_unit, a.qualification_hash),
        (a.rehearsal_root, a.rehearsal_unit, a.rehearsal_hash),
    ):
        wait_for_terminal_collection(unit)
        path = root / "complete.json"
        if digest(path) != expected:
            raise ValueError("actual terminal learning prerequisites required")
        d = json.loads(path.read_text())
        receipts.append(d)
        pins.update(d["input_file_hashes"])
        pins[str(path)] = expected
    qual, replay = receipts
    budget = BallAwareLearningBudget(a.resume_smoke_root is not None)
    resume = None
    if budget.course:
        for key in ("resume_smoke", "dev", "retention"):
            root, unit, expected = (getattr(a, key + "_" + s) for s in ("root", "unit", "hash"))
            if not all((root, unit, expected)):
                raise ValueError("course requires actual smoke, paired DEV and native retention")
            wait_for_terminal_collection(unit)
            path = root / "complete.json"
            if digest(path) != expected:
                raise ValueError("course physical prerequisite receipt changed")
            report = json.loads(path.read_text())
            pins.update(report["input_file_hashes"])
            pins[str(path)] = expected
            if key == "resume_smoke":
                from rosclaw_soccer.rsi.ball_aware_inputs import require_ball_aware_learning_smoke

                require_ball_aware_learning_smoke(report)
                if (
                    report.get("force_decoder") != "addressed"
                    or report["input_file_hashes"].get(str(a.qualification_root / "complete.json"))
                    != a.qualification_hash
                ):
                    raise ValueError(
                        "same addressed qualification and actual learned smoke required"
                    )
                resume = report
                resume_path = root / report["checkpoint_file"]
                pins[str(resume_path)] = report["checkpoint_hash"]
            elif (
                report["input_file_hashes"].get(str(a.resume_smoke_root / "complete.json"))
                != a.resume_smoke_hash
            ):
                raise ValueError("physical exam must bind the exact smoke candidate")
            elif key == "dev":
                if (
                    report.get("status") != "BALL_AWARE_PAIRED_DEV_NOT_FOOTBALL_QUALIFIED"
                    or report.get("all_initial_states_paired_exact") is not True
                    or report.get("force_decoder") != "addressed"
                    or report.get("gpu_contact_labels_qualified") is not True
                    or len(report.get("results", [])) != 4
                    or any(r.get("completed_count") != 32 for r in report["results"])
                ):
                    raise ValueError("all fixed paired DEV episodes required before course")
            elif (
                report.get("status") != "BALL_AWARE_NATIVE_RETENTION_NOT_FOOTBALL_QUALIFIED"
                or report.get("all_four_independent_full_repeats_exact") is not True
                or len(report.get("results", [])) != 4
                or any(
                    r.get("command_response_research_gate") is not True for r in report["results"]
                )
            ):
                raise ValueError("four native retention gates and exact full repeats required")
    from rosclaw_soccer.rsi.ball_aware_inputs import (
        BALL_AWARE_ABI,
        require_ball_aware_qualification,
    )
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import require_completed_stable_motion_smoke

    require_completed_stable_motion_smoke(replay)
    require_ball_aware_qualification(qual)
    if qual.get("force_decoder") != "addressed" or not qual.get("decoder_proof_hash"):
        raise ValueError("repeat-qualified addressed contact force required before new learning")
    if (
        qual["status"] != "BALL_AWARE_ABI_SENSOR_QUALIFICATION_NOT_LEARNED_SKILL"
        or qual["ball_aware_abi"] != BALL_AWARE_ABI
        or qual["new_optimizer_steps"] != 0
        or qual["actor_observation_dim"] != 177
        or qual["critic_observation_dim"] != 304
        or qual["optimizer_state_migrated"]
        or not qual["weights_only_checkpoint_reload"]
    ):
        raise ValueError("qualified new177/304 ABI, not an old motion certificate, required")
    initialization = a.qualification_root / qual["checkpoint_file"]
    qualified_runner = next(
        Path(k)
        for k in qual["input_file_hashes"]
        if k.endswith("/scripts/rsi_qualify_ball_aware_inputs.py")
    )
    qualified_source = qualified_runner.parent.parent
    for relative in (
        "scripts/rsi_mjlab_ball_aware_backend.py",
        "scripts/rsi_mjlab_ball_aware_rl.py",
        "scripts/rsi_mjwarp_addressed_contact_force.py",
        "scripts/rsi_mjlab_prior_rl.py",
        "src/rosclaw_soccer/rsi/ball_aware_migration.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
    ):
        if digest(Path.cwd() / relative) != qual["input_file_hashes"].get(
            str(qualified_source / relative)
        ):
            raise ValueError(
                "training must use qualified backend, reward and migration source bytes"
            )
    bank = a.rehearsal_root / "native-rehearsal1200.npz"
    pins[str(initialization)] = qual["checkpoint_hash"]
    if qual["input_file_hashes"].get(str(bank)) != replay["native_rehearsal"]["bank_hash"]:
        raise ValueError("same original actually recorded native rehearsal bank required")
    pins[str(bank)] = replay["native_rehearsal"]["bank_hash"]
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            force_decoder="addressed",
            input_file_hashes=pins,
            ball_aware_abi=BALL_AWARE_ABI,
            num_envs=budget.worlds,
            seed=budget.seed,
            ppo_iterations=budget.iterations,
            control_steps_per_iteration=24,
            maximum_gpu_world_substeps=budget.gpu_world_steps,
            optimizer_updates=budget.optimizer_updates,
            actor_observation_dim=177,
            critic_observation_dim=304,
            fresh_optimizer=not budget.course,
            parent_optimizer_state_loaded=budget.course,
            fresh_simulation_not_full_body_or_rng_resume=True,
            previous_iterations=2 if budget.course else 0,
            event_rows=budget.controls,
            ball_and_goal_are_current_sim_ground_truth_not_visual_perception=True,
            actor_position_targets_and_native_pd_not_direct_torque=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = None
    counts = []
    steps_before = 0
    try:
        import numpy as np
        import torch

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from scripts.rsi_mjlab_motion_learning_smoke import changed_weight_names, cloned_tensors

        sys.path.insert(0, str(a.reference_root))
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

        from scripts.rsi_mjlab_ball_aware_backend import (
            BallAwareMotionEnv,
            make_ball_aware_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated qualified GPU required")
        torch.manual_seed(budget.seed)
        np.random.seed(budget.seed)
        initial = torch.load(initialization, map_location="cpu", weights_only=True)
        if initial["ball_aware_abi"] != BALL_AWARE_ABI or not initial["fresh_optimizer_required"]:
            raise ValueError("explicit new architecture migration required")
        if initial.get("force_decoder") != "addressed":
            raise ValueError("initialization must bind the qualified force decoder")
        policy_config = next(
            Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml")
        )
        params = VelocityPolicyParameters.load(
            policy_config, expected_hash=pins[str(policy_config)]
        )
        cfg = make_ball_aware_training_cfg(
            parameters=params,
            command_mode="reference_position_velocity",
            command_point="pelvis",
            motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
            num_envs=budget.worlds,
            seed=budget.seed,
            balanced_initial_sources=True,
            football_rewards=True,
        )
        cfg.commands["motion"].sampling_mode = "uniform"
        cfg.sim.nan_guard.enabled = True
        env = BallAwareMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
        obs, _ = env.reset(seed=budget.seed)
        if (
            tuple(
                name + ".npz" for name in env.command_manager.get_term("motion").motion.motion_names
            )
            != TRAIN_MOTION_NAMES
        ):
            raise ValueError("exact TRAIN source split required")
        if obs["actor"].shape != (budget.worlds, 177) or obs["critic"].shape != (
            budget.worlds,
            304,
        ):
            raise ValueError("qualified new ABI dimensions required")
        settings = copy.deepcopy(initial["prior_configuration"])
        settings.update(
            logger="tensorboard",
            upload_model=False,
            save_interval=min(50, budget.iterations),
            max_iterations=budget.iterations,
            num_steps_per_env=24,
            seed=budget.seed,
        )
        settings["algorithm"].update(
            rehearsal_path=str(bank),
            rehearsal_hash=pins[str(bank)],
            rehearsal_reference_features=79,
        )
        runner = MjlabOnPolicyRunner(
            RslRlVecEnvWrapper(env, clip_actions=settings.get("clip_actions")),
            settings,
            str(a.output / "rsl"),
            "cuda:0",
        )
        start = runner.alg.save()
        for part in ("actor_state_dict", "critic_state_dict"):
            if any(
                not torch.equal(v.cpu(), initial["algorithm"][part][k])
                for k, v in start[part].items()
            ):
                raise ValueError("actual fresh RSL models differ from qualified migrated networks")
        if budget.course:
            previous = torch.load(resume_path, map_location="cuda:0", weights_only=True)
            if previous.get("force_decoder") != "addressed" or previous.get("iteration") != 2:
                raise ValueError("same-architecture addressed smoke continuation required")
            runner.alg.load(previous["algorithm"], None, True)
            runner.current_learning_iteration = 2
            start = runner.alg.save()
            for part in ("actor_state_dict", "critic_state_dict"):
                if any(
                    not torch.equal(v.cpu(), previous["algorithm"][part][k].cpu())
                    for k, v in start[part].items()
                ):
                    raise ValueError("resumed actual actor and critic differ from smoke")
        prior_rehearsal = 2 if budget.course else 0
        actor0, critic0 = (
            cloned_tensors(start[k]) for k in ("actor_state_dict", "critic_state_dict")
        )
        optimizers = list(
            {
                id(o): o for o in vars(runner.alg).values() if isinstance(o, torch.optim.Optimizer)
            }.values()
        )
        if not optimizers or any(bool(o.state) != budget.course for o in optimizers):
            raise ValueError("fresh smoke Adam or nonempty same-architecture resumed Adam required")
        owned = {id(p) for o in optimizers for group in o.param_groups for p in group["params"]}
        if not all(
            id(p) in owned
            for model in (runner.alg.actor, runner.alg.critic)
            for p in model.parameters()
        ):
            raise ValueError("new optimizer must own actual migrated actor and critic parameters")
        for optimizer in optimizers:
            counter = dict(attempted=0, completed=0)
            old_step = optimizer.step

            def counted(*args, _step=old_step, _counter=counter, **kwargs):
                _counter["attempted"] += 1
                result = _step(*args, **kwargs)
                _counter["completed"] += 1
                return result

            optimizer.step = counted
            counts.append(counter)
        steps_before = env._sim_step_counter
        event_rows = []
        original_env_step = env.step

        def observed_step(action):
            source_ids = env.command_manager.get_term("motion").motion_idx.clone()
            result = original_env_step(action)
            credit = env._identity_kick_credit
            if credit.last_step != env.common_step_counter or credit.cached is None:
                raise ValueError("current actual contact reward observation required")
            event_rows.append(
                dict(
                    source=source_ids.cpu().numpy(),
                    correct=credit.cached["contact"].detach().cpu().numpy(),
                    wrong=credit.cached["wrong_contact"].detach().cpu().numpy(),
                    terminated=result[2].cpu().numpy(),
                    truncated=result[3].cpu().numpy(),
                )
            )
            return result

        env.step = observed_step
        runner.learn(num_learning_iterations=budget.iterations, init_at_random_ep_len=False)
        torch.cuda.synchronize()
        state = runner.alg.save()
        actor_changes = changed_weight_names(actor0, cloned_tensors(state["actor_state_dict"]))
        critic_changes = changed_weight_names(critic0, cloned_tensors(state["critic_state_dict"]))
        new_columns = state["actor_state_dict"]["mlp.mlp.0.weight"][:, 168:]
        if (
            not actor_changes
            or not critic_changes
            or torch.count_nonzero(new_columns) == 0
            or any(c["attempted"] != c["completed"] for c in counts)
            or sum(c["completed"] for c in counts) != budget.optimizer_updates
            or env._sim_step_counter - steps_before != budget.controls * 10
            or len(event_rows) != budget.controls
            or runner.alg.rehearsal_updates != prior_rehearsal + budget.iterations
            or runner.alg.rehearsal_rows != (prior_rehearsal + budget.iterations) * 256
            or not torch.equal(
                actor0["mlp.prior_mean"], state["actor_state_dict"]["mlp.prior_mean"].cpu()
            )
            or not torch.equal(
                actor0["mlp.prior_denominator"],
                state["actor_state_dict"]["mlp.prior_denominator"].cpu(),
            )
        ):
            raise ValueError(
                "actual complete actor/critic/ball9 learning and rehearsal accounting required"
            )
        events_path = a.output / "training-contact-events.npz"
        np.savez_compressed(
            events_path, **{k: np.array([row[k] for row in event_rows]) for k in event_rows[0]}
        )
        source_events = []
        for source_id, name in enumerate(TRAIN_MOTION_NAMES):
            mask = np.asarray([r["source"] for r in event_rows]) == source_id
            source_events.append(
                dict(
                    source=name,
                    control_samples=int(mask.sum()),
                    correct_first_contacts=int(
                        np.asarray([r["correct"] for r in event_rows])[mask].sum()
                    ),
                    wrong_first_contacts=int(
                        np.asarray([r["wrong"] for r in event_rows])[mask].sum()
                    ),
                )
            )
        path = a.output / f"ball-aware-update{budget.iterations}.pt"
        result = dict(
            algorithm=state,
            initial_actor_state_dict=actor0,
            initial_critic_state_dict=critic0,
            ball_aware_abi=BALL_AWARE_ABI,
            actor_observation_dim=177,
            critic_observation_dim=304,
            prior_configuration=settings,
            iteration=prior_rehearsal + budget.iterations,
            iterations_this_stage=budget.iterations,
            source_head=a.expected_head,
            force_decoder="addressed",
            parent_checkpoint_hash=resume["checkpoint_hash"] if resume else qual["checkpoint_hash"],
            optimizer_state_migrated=False,
            stable_prior_rehearsal=True,
            velocity_command_anatomical_point="pelvis",
            velocity_command_position_feedback_gain=1.0,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
        )
        torch.save(result, path)
        loaded = torch.load(path, map_location="cpu", weights_only=True)
        for part in ("actor_state_dict", "critic_state_dict"):
            if any(
                not torch.equal(v.cpu(), loaded["algorithm"][part][k])
                for k, v in state[part].items()
            ):
                raise ValueError("safe actor/critic learned checkpoint round-trip differs")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="BALL_AWARE_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED"
                if budget.course
                else "BALL_AWARE_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED",
                source_head=a.expected_head,
                force_decoder="addressed",
                input_file_hashes=pins,
                ball_aware_abi=BALL_AWARE_ABI,
                actor_observation_dim=177,
                critic_observation_dim=304,
                gpu_world_substeps=(env._sim_step_counter - steps_before) * budget.worlds,
                policy_control_samples=budget.samples,
                iterations=budget.iterations,
                num_envs=budget.worlds,
                seed=budget.seed,
                previous_iterations=prior_rehearsal,
                event_file_hash=digest(events_path),
                training_source_events=source_events,
                new_optimizer_steps=sum(c["completed"] for c in counts),
                optimizer_counts=counts,
                changed_actor_weights=actor_changes,
                changed_critic_weights=critic_changes,
                ball9_weight_norm=float(new_columns.norm()),
                native_rehearsal=state["native_rehearsal"],
                checkpoint_file=path.name,
                checkpoint_hash=digest(path),
                weights_only_checkpoint_reload=True,
                parent_optimizer_state_loaded=budget.course,
                fresh_simulation_not_full_body_or_rng_resume=True,
                velocity_command_anatomical_point="pelvis",
                velocity_command_position_feedback_gain=1.0,
                parent_checkpoint_hash=resume["checkpoint_hash"]
                if resume
                else qual["checkpoint_hash"],
                new_native_steps=0,
                goal_complete=False,
                football_skill_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                optimizer_counts=counts,
                attempted_gpu_world_substeps=(getattr(env, "_sim_step_counter", 0) - steps_before)
                * budget.worlds,
                new_native_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
