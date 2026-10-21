"""Fixed SIM native-contact24 PPO pilot; fresh177 critic, no automatic promotion."""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--actor-kind", choices=("shared", "bounded-residual"), default="shared")
    p.add_argument("--adapter-proof-root", type=Path)
    p.add_argument("--adapter-proof-hash")
    p.add_argument("--training-seed", type=int, choices=(7602, 7605, 7606, 7607))
    p.add_argument("--course-kind", choices=("contact24", "contact-recovery"), default="contact24")
    for key in ("recovery-proof", "recovery-fault"):
        p.add_argument("--" + key + "-root", type=Path)
        p.add_argument("--" + key + "-hash")
    p.add_argument("--resume-root", type=Path)
    p.add_argument("--resume-hash")
    p.add_argument("--resume-mode", choices=("qualify21", "train132"))
    p.add_argument("--resume-proof", type=Path)
    p.add_argument("--resume-proof-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen learner and unused output required")
    f = a.qualification_root / "complete.json"
    if digest(f) != a.qualification_hash:
        raise ValueError("exact native course qualification required")
    qual = json.loads(f.read_text())
    if (
        qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL"
        or any(
            qual.get(k) is not True
            for k in (
                "independent_repeat_exact",
                "active_reset_rejected",
                "credit_history_cleared",
                "failed_world_frozen",
            )
        )
        or qual.get("protocol") != "native-contact24-v1"
        or qual.get("critic_abi") != 177
        or len(qual.get("episodes", [])) != 2
    ):
        raise ValueError("normal episodes, source guards and actual fault/reset proof required")
    pins = {**qual["input_file_hashes"], str(f): a.qualification_hash}
    for i, h in enumerate(qual["raw_hashes"]):
        pins[str(a.qualification_root / f"episode{i}.npz")] = h
    pins[str(a.qualification_root / "fault-fixture.npz")] = qual["fixture_hash"]
    residual = a.actor_kind == "bounded-residual"
    option = a.course_kind == "contact-recovery"
    resume = a.resume_mode is not None
    resume_checkpoint = None
    if resume:
        if not option or a.resume_root is None or not a.resume_hash:
            raise ValueError("explicit finite-option source required for continuation")
        source_receipt = a.resume_root / "complete.json"
        if digest(source_receipt) != a.resume_hash:
            raise ValueError("exact completed source learner required")
        source = json.loads(source_receipt.read_text())
        if (
            source.get("status") != "NATIVE_CONTACT24_RECOVERY60_PPO_CANDIDATE_REQUIRES_FULL_EXAM"
            or source.get("training_seed") != 7607
            or len(source["results"]) != 20
        ):
            raise ValueError("fixed20 finite-option pilot required")
        pins.update(source["input_file_hashes"])
        pins[str(source_receipt)] = a.resume_hash
        ck = next(x for x in source["checkpoints"] if x["episode"] == 20)
        resume_checkpoint = a.resume_root / ck["file"]
        pins[str(resume_checkpoint)] = ck["hash"]
        if a.resume_mode == "train132":
            if not a.resume_proof or digest(a.resume_proof) != a.resume_proof_hash:
                raise ValueError("paired actual resumed episode proof required before long run")
            rp = json.loads(a.resume_proof.read_text())
            if (
                rp.get("status") != "NATIVE_OPTION_RESUMED_EPISODE_EXACT_NOT_TASK_PASS"
                or rp.get("resume_checkpoint_hash") != ck["hash"]
                or rp.get("all_state_and_trajectory_exact") is not True
            ):
                raise ValueError("complete paired resumed physical/optimizer proof required")
            pins.update(rp["input_file_hashes"])
            pins[str(a.resume_proof)] = a.resume_proof_hash
            for relative in (
                "scripts/rsi_train_native_contact_course.py",
                "src/rosclaw_soccer/rsi/contact_option_resume.py",
                "src/rosclaw_soccer/rsi/contact_option_rollout.py",
            ):
                if rp["qualified_source_hashes"].get(relative) != digest(Path(relative)):
                    raise ValueError("same actual qualified continuation implementation required")
        elif a.resume_proof or a.resume_proof_hash:
            raise ValueError("boundary attempt does not inherit a long-run proof")
    elif any((a.resume_root, a.resume_hash, a.resume_proof, a.resume_proof_hash)):
        raise ValueError("resume arguments require explicit continuation intent")
    if option and (not residual or a.training_seed not in (None, 7607)):
        raise ValueError("new finite-option pilot requires frozen residual and fixed seed7607")
    if not option and a.training_seed == 7607:
        raise ValueError("seed7607 belongs only to the new finite-option pilot")
    for key, status in (
        ("recovery_proof", "NATIVE_CONTACT24_RECOVERY60_PROBE_NOT_LEARNER"),
        ("recovery_fault", "NATIVE_RECOVERY_FAILED_WORLD_PATHS_QUALIFIED_NOT_SKILL"),
    ):
        root, h = getattr(a, key + "_root"), getattr(a, key + "_hash")
        if option:
            if root is None or not h or digest(root / "complete.json") != h:
                raise ValueError("actual continuous recovery and failure qualification required")
            report = json.loads((root / "complete.json").read_text())
            checks = (
                ("prefix_exact", "handoff_state_exact", "repeats_exact", "strict_replays_exact")
                if key == "recovery_proof"
                else (
                    "failed_state_clock_observation_frozen",
                    "inference_only_active",
                    "strict_replay_exact",
                )
            )
            if report["status"] != status or any(report.get(k) is not True for k in checks):
                raise ValueError("recovery evidence must prove all declared invariants")
            relative = "src/rosclaw_soccer/rsi/native_recovery_tail.py"
            old_hashes = {
                v for f, v in report["input_file_hashes"].items() if f.endswith("/" + relative)
            }
            if old_hashes != {digest(Path(relative))}:
                raise ValueError("same actually qualified recovery code required")
            pins.update(report["input_file_hashes"])
            pins[str(root / "complete.json")] = h
            if key == "recovery_proof":
                for i, raw_hash in enumerate(report["raw_hashes"]):
                    pins[str(root / f"option{i}.npz")] = raw_hash
            else:
                if report["head_failed_worlds"] != [0] or report["tail_failed_worlds"] != [1]:
                    raise ValueError("actual head and recovery failures must have been observed")
                pins[str(root / "fault.npz")] = report["raw_hash"]
        elif root is not None or h:
            raise ValueError("new recovery evidence cannot relabel the old contact24 pilot")
    if residual:
        if not a.adapter_proof_root or not a.adapter_proof_hash:
            raise ValueError("residual training requires its own actual zero-influence proof")
        proof_path = a.adapter_proof_root / "complete.json"
        if digest(proof_path) != a.adapter_proof_hash:
            raise ValueError("immutable residual proof required")
        adapter_proof = json.loads(proof_path.read_text())
        if (
            adapter_proof.get("status")
            != "NATIVE_BOUNDED_CONTACT_ADAPTER_ZERO_INFLUENCE_QUALIFIED_NOT_SKILL"
            or any(
                adapter_proof.get(k) is not True
                for k in (
                    "zero_influence_all_inputs",
                    "modified_finite_adapter_far_and_missing_perception_exact",
                    "full_paired_physics_exact",
                    "frozen_foundation_exact",
                )
            )
            or adapter_proof["input_file_hashes"].get(str(f)) != a.qualification_hash
        ):
            raise ValueError("same native course and actual protected adapter proof required")
        pins.update(adapter_proof["input_file_hashes"])
        pins[str(proof_path)] = a.adapter_proof_hash
        pins[str(a.adapter_proof_root / "function-test.npz")] = adapter_proof["function_test_hash"]
        for label, h in adapter_proof["paired_raw_hashes"].items():
            pins[str(a.adapter_proof_root / (label + ".npz"))] = h
        adapter_source = next(
            Path(k).parent.parent
            for k in adapter_proof["input_file_hashes"]
            if k.endswith("/scripts/rsi_qualify_native_contact_adapter.py")
        )
        relative = "src/rosclaw_soccer/rsi/native_contact_adapter.py"
        if digest(Path.cwd() / relative) != pins[str(adapter_source / relative)]:
            raise ValueError("same actually qualified adapter implementation required")
    elif a.adapter_proof_root or a.adapter_proof_hash:
        raise ValueError("adapter evidence cannot qualify a shared-actor experiment")
    if not residual and a.training_seed is not None:
        raise ValueError("shared pilot retains its original fixed seed")
    qualified_root = next(
        Path(k).parent.parent
        for k in pins
        if k.endswith("/scripts/rsi_qualify_native_contact_course.py")
    )
    for rel in (
        "src/rosclaw_soccer/rsi/native_contact_course.py",
        "src/rosclaw_soccer/rsi/native_course_tracking.py",
        "src/rosclaw_soccer/rsi/native_physics_batch.py",
        "src/rosclaw_soccer/rsi/native_reference_observation.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
        "scripts/rsi_native_course_inputs.py",
    ):
        if digest(Path.cwd() / rel) != pins[str(qualified_root / rel)]:
            raise ValueError("learner must use exact actually qualified course/PD/contact source")
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    seed, actor_lr = (
        ((a.training_seed or (7607 if option else 7602)), 3e-4) if residual else (7601, 3e-5)
    )
    protocol = "native-contact24-recovery60-v1" if option else "native-contact24-v1"
    if resume:
        protocol = "native-contact24-recovery60-continuation-v1"
    start_episode = 20 if resume else 0
    end_episode = (21 if a.resume_mode == "qualify21" else 132) if resume else 20
    saves = (
        (21,) if a.resume_mode == "qualify21" else (36, 68, 132) if resume else (4, 8, 12, 16, 20)
    )
    before_counts = dict(source["counts"]) if resume else {}
    prior_native_steps = source["native_steps"] if resume else 0
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            protocol=protocol,
            resume_mode=a.resume_mode,
            resume_source_hash=a.resume_hash,
            resume_checkpoint_hash=pins[str(resume_checkpoint)] if resume else None,
            resume_proof_hash=a.resume_proof_hash,
            start_episode=start_episode,
            end_episode=end_episode,
            counts_before_resume=before_counts,
            prior_native_steps=prior_native_steps,
            recovery_proof_hash=a.recovery_proof_hash,
            recovery_fault_hash=a.recovery_fault_hash,
            actual_recovery_controls=60 if option else 0,
            seed=seed,
            actor_kind=a.actor_kind,
            frozen_foundation=residual,
            adapter_proof_hash=a.adapter_proof_hash,
            worlds=32,
            course_lead=10,
            controls=24,
            value_only_episodes=0 if resume else 4,
            prior_value_only_episodes=4 if resume else 0,
            ppo_episodes=end_episode - 4,
            new_ppo_episodes=end_episode - start_episode if resume else 16,
            maximum_native_steps=(end_episode - start_episode) * 32 * (84 if option else 24) * 10,
            maximum_rollout_policy_rows=(end_episode - start_episode) * 32 * 24,
            maximum_recovery_foundation_rows=(end_episode - start_episode) * 32 * 60
            if option
            else 0,
            maximum_optimizer_steps=(end_episode - start_episode) * 16,
            epochs=4,
            minibatches=4,
            actor_abi=177,
            critic_abi=177,
            fresh_critic=not resume,
            fresh_optimizer=not resume,
            optimizer_resumed=resume,
            restored_native177_critic_and_Adam=resume,
            old_critic304_and_old_adam_loaded=False,
            actor_learning_rate=actor_lr,
            critic_learning_rate=3e-4,
            exploration_learning_rate=1e-5,
            gaussian_std_initial=None if resume else 0.05,
            gaussian_std_restored=resume,
            gaussian_std_bounds=[0.01, 0.1],
            gamma=0.99,
            gae_lambda=0.95,
            ppo_clip=0.1,
            target_approximate_kl=0.02,
            baseline_anchor_kl_coefficient=0.01,
            rehearsal_mse_coefficient=1.0,
            rehearsal_batch=256,
            true_terminal_bootstrap=False,
            timeout_bootstrap_from_actual_final_observation=not option,
            contact_handoff_from_actual_discounted_recovery_return=option,
            recovery_finite_terminal_value_zero=option,
            option_terminal_reward_definition="contact_option_credit.recovery_terminal_reward"
            if option
            else None,
            invalid_ended_world_samples_excluded_from_gradient_and_normalization=True,
            checkpoint_episodes=[start_episode, *saves],
            require_independent_full_native_exam_before_expansion=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    env = None
    counts = dict(
        optimizer_attempted=0,
        optimizer_completed=0,
        value_only_steps=0,
        ppo_steps=0,
        rollout_policy_rows=0,
        rollout_critic_rows=0,
        optimizer_actor_rows=0,
        optimizer_critic_rows=0,
        optimizer_baseline_rows=0,
        kl_skipped_minibatches=0,
    )
    history = []
    recovery_history = []
    try:
        import math

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.contact_option_credit import contact_option_gae
        from rosclaw_soccer.rsi.contact_option_resume import (
            IDENTITY_KEYS,
            restore_contact_option_boundary,
        )
        from rosclaw_soccer.rsi.contact_option_rollout import finish_contact_option
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
        from rosclaw_soccer.rsi.native_contact_ppo import (
            course_gae,
            make_contact_actor_critic,
            native_ppo_loss,
        )
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)
        torch.manual_seed(seed)
        np.random.seed(seed)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required for actual neural learning")

        def pinned(suffix):
            matches = [Path(k) for k in pins if k.endswith(suffix)]
            if len(matches) != 1:
                raise ValueError("one exact pinned input required: " + suffix)
            return matches[0]

        cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        cp = pinned("/ball-aware-update200.pt")
        bp = pinned("/rsi-body-bound-precontact-curriculum-20261021/precontact-resets.npz")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        rp = pinned("/rsi-pelvis-online-ppo-smoke-20261021/native-rehearsal1200.npz")
        params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
        with np.load(bp, allow_pickle=False) as z:
            bank = {k: z[k].copy() for k in z.files}
        with np.load(rp, allow_pickle=False) as z:
            old_obs = z["observation"].copy()
        if (
            old_obs.shape != (1200, 98)
            or old_obs.dtype != np.float32
            or not np.isfinite(old_obs).all()
        ):
            raise ValueError("entire qualified old native rehearsal bank required")
        rehearsal = torch.from_numpy(
            np.concatenate((old_obs, np.zeros((1200, 79), dtype=np.float32)), axis=1)
        ).cuda()
        plans = []
        for name in TRAIN_MOTION_NAMES:
            f = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + name.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(f, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        refs = load_body_references(a.reference_root, model, layout, plans)
        env = NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=32,
            lead=10,
        )
        actor = VelocityPriorActor.from_onnx(
            onnx, expected_hash=pins[str(onnx)], reference_features=79
        ).cuda()
        parent = torch.load(cp, map_location="cpu", weights_only=True)
        actor.load_state_dict(
            {k: parent["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
            strict=True,
        )
        foundation = None
        adapter_config = None
        if residual:
            from rosclaw_soccer.rsi.native_contact_adapter import NativeResidualContactActor

            actor = NativeResidualContactActor(actor).cuda()
            foundation = {k: v.detach().cpu().clone() for k, v in actor.base.state_dict().items()}
            adapter_config = dict(
                near_radius_m=actor.near_radius_m,
                far_radius_m=actor.far_radius_m,
                maximum_normalized_residual=actor.maximum_normalized_residual,
                missing_ball9_disables_adapter=True,
                hidden_dims=[128, 64],
                input_dim=177,
                output_dim=29,
            )
        policy = make_contact_actor_critic(actor).cuda()
        optimizer = torch.optim.Adam(
            [
                dict(params=[p for p in policy.actor.parameters() if p.requires_grad], lr=actor_lr),
                dict(params=list(policy.critic.parameters()), lr=3e-4),
                dict(params=[policy.log_std], lr=1e-5),
            ]
        )
        if optimizer.state:
            raise ValueError("fresh Adam required")
        if resume:
            boundary = torch.load(resume_checkpoint, map_location="cpu", weights_only=True)
            restored = restore_contact_option_boundary(
                policy,
                optimizer,
                resume_checkpoint,
                expected_hash=pins[str(resume_checkpoint)],
                expected_identity={k: boundary[k] for k in IDENTITY_KEYS},
            )
            counts = restored["counts"]
            env.episode = restored["course_episode"]
        initial_actor = {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}
        initial_critic = {
            k: v.detach().cpu().clone() for k, v in policy.critic.state_dict().items()
        }
        frozen_base = {k: v.detach().cpu().clone() for k, v in policy.baseline.state_dict().items()}
        initial_std = policy.log_std.detach().cpu().clone()
        checkpoints = []

        def checkpoint(episode):
            file = a.output / f"native-contact-episode{episode}.pt"
            value = dict(
                schema="soccer.rsi.native_contact24_recovery60_continuation_ppo.v1"
                if resume
                else "soccer.rsi.native_contact24_recovery60_residual_ppo.v1"
                if option
                else "soccer.rsi.native_contact24_residual_ppo.v1"
                if residual
                else "soccer.rsi.native_contact24_ppo.v1",
                actor_kind=a.actor_kind,
                adapter_config=adapter_config,
                adapter_proof_hash=a.adapter_proof_hash,
                protocol=protocol,
                resume_checkpoint_hash=pins[str(resume_checkpoint)] if resume else None,
                resume_source_hash=a.resume_hash,
                resume_proof_hash=a.resume_proof_hash,
                resume_mode=a.resume_mode,
                counts_before_resume=before_counts,
                prior_native_steps=prior_native_steps,
                new_native_steps=env.native_steps,
                aggregate_native_steps=prior_native_steps + env.native_steps,
                recovery_proof_hash=a.recovery_proof_hash,
                recovery_fault_hash=a.recovery_fault_hash,
                episode_count=episode,
                value_warmup_episodes=4,
                ppo_episodes=max(0, episode - 4),
                actor_observation_dim=177,
                critic_observation_dim=177,
                native_pd_position_targets=29,
                parent_checkpoint_hash=pins[str(cp)],
                original_onnx_hash=pins[str(onnx)],
                parameter_hash=pins[str(cfg)],
                native_model_hash=pins[str(mp)],
                reset_bank_hash=pins[str(bp)],
                source_head=a.expected_head,
                training_seed=seed,
                qualification_hash=a.qualification_hash,
                actor_state_dict=policy.actor.state_dict(),
                critic_state_dict=policy.critic.state_dict(),
                policy_state_dict=policy.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=[v.cpu() for v in torch.cuda.get_rng_state_all()],
                counts=dict(counts),
                course_episode=env.episode,
                fresh_optimizer=not resume,
                optimizer_resumed=resume,
                activation_ceiling="SIM_ONLY",
                promotion_authorized=False,
                hardware_authorized=False,
            )
            torch.save(value, file)
            loaded = torch.load(file, map_location="cpu", weights_only=True)
            if any(
                not torch.equal(v.detach().cpu(), loaded["policy_state_dict"][k])
                for k, v in policy.state_dict().items()
            ):
                raise ValueError("checkpoint safe readback differs")
            checkpoints.append(dict(episode=episode, file=file.name, hash=digest(file)))

        checkpoint(start_episode)
        results = []
        for episode in range(start_episode, end_episode):
            env.reset()
            history = []
            recovery_history = []
            while not env.ended.all():
                obs = torch.from_numpy(env.observation).cuda()
                with torch.no_grad():
                    if option:
                        ids = np.flatnonzero(~env.ended)
                        action = torch.zeros((env.worlds, 29), device="cuda")
                        logp = torch.zeros(env.worlds, device="cuda")
                        value = torch.zeros(env.worlds, device="cuda")
                        dist = policy.distribution(obs[ids])
                        action[ids] = dist.sample()
                        logp[ids] = dist.log_prob(action[ids]).sum(-1)
                        value[ids] = policy.value(obs[ids])
                        infer_rows = len(ids)
                    else:
                        dist = policy.distribution(obs)
                        action = dist.sample()
                        logp = dist.log_prob(action).sum(-1)
                        value = policy.value(obs)
                        infer_rows = env.worlds
                counts["rollout_policy_rows"] += infer_rows
                counts["rollout_critic_rows"] += infer_rows
                result = env.step(action.cpu().numpy().copy())
                with torch.no_grad():
                    if option:
                        ids = np.flatnonzero(~result["ended"])
                        next_value = torch.zeros(env.worlds, device="cuda")
                        if len(ids):
                            next_value[ids] = policy.value(
                                torch.from_numpy(result["observation"][ids]).cuda()
                            )
                        next_rows = len(ids)
                    else:
                        next_value = policy.value(torch.from_numpy(result["observation"]).cuda())
                        next_rows = env.worlds
                counts["rollout_critic_rows"] += next_rows
                history.append(
                    {
                        **result,
                        "action": action.cpu().numpy(),
                        "old_logp": logp.cpu().numpy(),
                        "old_value": value.cpu().numpy(),
                        "next_value": next_value.cpu().numpy(),
                    }
                )
            raw = {k: np.stack([r[k] for r in history]) for k in history[0]}
            folder = a.output / f"episode{episode:02d}"
            folder.mkdir()
            path = folder / "rollout.npz"
            np.savez_compressed(path, bank_index=env.index, **raw)
            outcomes = []
            recovery_hash = None
            if option:
                recovery, outcomes, _recovery_rows = finish_contact_option(
                    env, raw, policy.baseline.base, counters=counts, trace=recovery_history
                )
                recovery_path = folder / "recovery.npz"
                np.savez_compressed(recovery_path, bank_index=env.index, **recovery)
                recovery_hash = digest(recovery_path)
                adv, returns = contact_option_gae(
                    raw["reward"].astype(np.float32),
                    raw["old_value"],
                    raw["next_value"],
                    raw["terminated"],
                    raw["truncated"],
                    raw["valid"],
                    tail_reward=recovery["reward"],
                    tail_valid=recovery["valid"],
                    tail_terminal=recovery["terminal"],
                )
            else:
                adv, returns = course_gae(
                    raw["reward"].astype(np.float32),
                    raw["old_value"],
                    raw["next_value"],
                    raw["terminated"],
                    raw["truncated"],
                    raw["valid"],
                )
            mask = raw["valid"].reshape(-1)
            data = {
                k: torch.from_numpy(v.reshape((-1, *v.shape[2:]))[mask].copy()).cuda()
                for k, v in dict(
                    obs=raw["observation_before"],
                    action=raw["action"],
                    old_logp=raw["old_logp"],
                    old_value=raw["old_value"],
                    advantage=adv,
                    returns=returns,
                ).items()
            }
            data["advantage"] = (data["advantage"] - data["advantage"].mean()) / (
                data["advantage"].std(unbiased=False) + 1e-8
            )
            warmup = episode < 4
            update_rows = []
            stop = False
            for _epoch in range(4):
                if stop:
                    break
                permutation = torch.randperm(int(mask.sum()), device="cuda")
                for minibatch in torch.tensor_split(permutation, 4):
                    if not len(minibatch):
                        continue
                    old_ids = (
                        torch.arange(256, device="cuda") + counts["optimizer_completed"] * 256
                    ) % 1200
                    loss, metrics = native_ppo_loss(
                        policy,
                        data["obs"][minibatch],
                        data["action"][minibatch],
                        data["old_logp"][minibatch],
                        data["old_value"][minibatch],
                        data["advantage"][minibatch],
                        data["returns"][minibatch],
                        rehearsal[old_ids],
                        value_only=warmup,
                    )
                    counts["optimizer_critic_rows"] += len(minibatch)
                    if not warmup:
                        counts["optimizer_actor_rows"] += len(minibatch) + 256
                        counts["optimizer_baseline_rows"] += len(minibatch) + 256
                    if not warmup and float(metrics["approximate_kl"]) > 0.02:
                        counts["kl_skipped_minibatches"] += 1
                        stop = True
                        break
                    if not torch.isfinite(loss):
                        raise ValueError("nonfinite native critic/PPO loss")
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [p for group in optimizer.param_groups for p in group["params"]], 1.0
                    )
                    if not torch.isfinite(norm):
                        raise ValueError("nonfinite clipped native PPO gradient")
                    counts["optimizer_attempted"] += 1
                    optimizer.step()
                    counts["optimizer_completed"] += 1
                    counts["value_only_steps" if warmup else "ppo_steps"] += 1
                    with torch.no_grad():
                        policy.log_std.clamp_(math.log(0.01), math.log(0.1))
                    if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
                        raise ValueError("nonfinite learned weights, no checkpoint promotion")
                    if residual and any(
                        not torch.equal(v.cpu(), foundation[k])
                        for k, v in policy.actor.base.state_dict().items()
                    ):
                        raise ValueError(
                            "immutable residual foundation changed during optimization"
                        )
                    update_rows.append({k: float(v) for k, v in metrics.items()})
            if any(
                not torch.equal(v.cpu(), frozen_base[k])
                for k, v in policy.baseline.state_dict().items()
            ):
                raise ValueError("frozen stability baseline changed")
            if warmup and (
                any(
                    not torch.equal(v.cpu(), initial_actor[k])
                    for k, v in policy.actor.state_dict().items()
                )
                or not torch.equal(policy.log_std.cpu(), initial_std)
                or any(p in optimizer.state for p in policy.actor.parameters())
                or policy.log_std in optimizer.state
            ):
                raise ValueError("critic-only warmup touched actor or exploration")
            row = dict(
                episode=episode,
                mode="value-only" if warmup else "PPO",
                valid_samples=int(mask.sum()),
                raw_hash=digest(path),
                recovery_raw_hash=recovery_hash,
                recovery_outcomes=outcomes,
                reward_sum=float(raw["reward"].sum()),
                terminations=int(raw["terminated"].sum()),
                positive_contact_worlds=int((raw["credit_contact"] > 0).any(axis=0).sum()),
                forbidden_contact_worlds=int(
                    (raw["credit_forbidden_contact"] > 0).any(axis=0).sum()
                ),
                native_steps=env.native_steps,
                counts=dict(counts),
                minibatches=update_rows,
            )
            write_once(folder / "result.json", row)
            results.append(row)
            print(json.dumps({k: v for k, v in row.items() if k != "minibatches"}), flush=True)
            if episode + 1 in saves:
                checkpoint(episode + 1)
        actor_changed = [
            k
            for k, v in policy.actor.state_dict().items()
            if not torch.equal(v.cpu(), initial_actor[k])
        ]
        critic_changed = [
            k
            for k, v in policy.critic.state_dict().items()
            if not torch.equal(v.cpu(), initial_critic[k])
        ]
        if (
            not actor_changed
            or not critic_changed
            or counts["optimizer_attempted"] != counts["optimizer_completed"]
            or not counts["ppo_steps"]
            or not counts["value_only_steps"]
        ):
            raise ValueError("real completed actor/critic optimization required")
        if residual and any(not k.startswith("adapter.") for k in actor_changed):
            raise ValueError("residual learning changed non-adapter state")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "NATIVE_OPTION_BOUNDARY_ATTEMPT_REQUIRES_PAIRED_COMPARISON"
                    if a.resume_mode == "qualify21"
                    else "NATIVE_OPTION_CONTINUATION132_REQUIRES_FULL_EXAM"
                )
                if resume
                else "NATIVE_CONTACT24_RECOVERY60_PPO_CANDIDATE_REQUIRES_FULL_EXAM"
                if option
                else "NATIVE_CONTACT24_RESIDUAL_PPO_CANDIDATE_REQUIRES_FULL_EXAM"
                if residual
                else "NATIVE_CONTACT24_PPO_CANDIDATE_REQUIRES_FULL_EXAM",
                actor_kind=a.actor_kind,
                training_seed=seed,
                adapter_config=adapter_config,
                adapter_proof_hash=a.adapter_proof_hash,
                input_file_hashes=pins,
                results=results,
                checkpoints=checkpoints,
                actor_changed=actor_changed,
                critic_changed=critic_changed,
                native_steps=env.native_steps,
                counts_before_resume=before_counts,
                prior_native_steps=prior_native_steps,
                aggregate_native_steps=prior_native_steps + env.native_steps,
                new_optimizer_updates=counts["optimizer_completed"]
                - before_counts.get("optimizer_completed", 0),
                resume_checkpoint_hash=pins[str(resume_checkpoint)] if resume else None,
                resume_source_hash=a.resume_hash,
                resume_proof_hash=a.resume_proof_hash,
                resume_mode=a.resume_mode,
                end_episode=end_episode,
                counts=counts,
                actor_abi=177,
                critic_abi=177,
                protocol=protocol,
                recovery_proof_hash=a.recovery_proof_hash,
                recovery_fault_hash=a.recovery_fault_hash,
                full_native_exam_passed=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as exc:
        failure = dict(
            error=repr(exc),
            native_steps=env.native_steps if env else 0,
            counts=counts,
            hardware_authorized=False,
        )
        print(json.dumps(failure), flush=True)
        write_once(a.output / "failed.json", failure)
        if history:
            # Preserve completed transitions even if the current control/update
            # failed before its normal episode publication. Never overwrite.
            partial = {k: np.stack([r[k] for r in history]) for k in history[0]}
            np.savez_compressed(a.output / "failed-partial-episode.npz", **partial)
        if recovery_history:
            np.savez_compressed(
                a.output / "failed-recovery-partial.npz",
                **{k: np.stack([r[k] for r in recovery_history]) for k in recovery_history[0]},
            )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
