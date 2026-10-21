"""Bounded research-only full soccer recurrent PPO on native MuJoCo.

All assets are external CC-BY-NC research inputs. No publicity/deployment
authorization. Complete episode BPTT, fixed anchor and untouched physics.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher-root", type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--qualification-root", type=Path, required=True)
    p.add_argument("--qualification-hash", required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--iterations", type=int, choices=(4, 8, 16), required=True)
    p.add_argument("--post-update-trust", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and immutable exact training source required")
    teacher_file = a.teacher_root / "complete.json"
    qualification_file = a.qualification_root / "complete.json"
    if digest(teacher_file) != a.teacher_hash or digest(qualification_file) != a.qualification_hash:
        raise ValueError("exact completed native teacher and independent qualification required")
    teacher = json.loads(teacher_file.read_text())
    qualification = json.loads(qualification_file.read_text())
    if (
        teacher["status"] != "PAID_RESEARCH_NATIVE_TORCH_PROBE_NOT_PROMOTED"
        or qualification["status"] != "PAID_TORCH_FOLLOWUP_CHAIN_CLOSED_NOT_PROMOTED"
        or qualification["candidate_hash"] != a.teacher_hash
        or qualification["comparison_status"] != "PAID_BACKEND_NATIVE_PARITY_PASSED"
    ):
        raise ValueError("full trainable teacher physical qualification required")
    pins = {
        **teacher["input_file_hashes"],
        **qualification["input_file_hashes"],
        str(teacher_file): a.teacher_hash,
        str(qualification_file): a.qualification_hash,
    }
    for name, key in (("native-audit", "native_audit_hash"), ("comparison", "comparison_hash")):
        pins[str(a.qualification_root / name / "complete.json")] = qualification[key]
    for relative, h in teacher["raw_hashes"].items():
        pins[str(a.teacher_root / relative)] = h
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    base_cases = [
        r for r in teacher["results"] if r["profile"] == "cached-standard" and r["repeat"] == 0
    ]
    names = list(dict.fromkeys(r["source"] for r in base_cases))
    if len(base_cases) != 16 or len(names) != 8:
        raise ValueError("complete fixed16 eight-source standard-ball baseline required")
    train, dev = [], []
    for name in names:
        sign = 1 if "left" in name else -1
        for i, (x, y, vx, vy) in enumerate(
            (
                (0.625, 0.125, 0, 0),
                (0.675, 0.175, 0, 0),
                (0.95, -0.025, -0.2, 0.1),
                (1.05, 0.025, -0.25, 0.08),
            )
        ):
            train.append(
                dict(
                    source=name, variant=i, ball_local=[x, sign * y], velocity_local=[vx, sign * vy]
                )
            )
        for i, (x, y, vx, vy) in enumerate(((0.65, 0.2, 0, 0), (1.0, 0.05, -0.15, 0.12))):
            dev.append(
                dict(
                    source=name, variant=i, ball_local=[x, sign * y], velocity_local=[vx, sign * vy]
                )
            )
    a.output.mkdir()
    config = dict(
        seed=20261021,
        iterations=a.iterations,
        train_episodes_per_iteration=32,
        critic_only_iterations=1,
        ppo_epochs=3,
        episode_minibatch_size=4,
        lr=0.00001,
        gamma=0.995,
        gae_lambda=0.95,
        initial_std=0.02,
        maximum_kl=0.015,
        clip_ratio=0.1,
        max_grad_norm=0.5,
        replay_anchor_coefficient=0.02,
        full_unpadded_episode_BPTT=True,
        real_source_plus_held_tail_natural_terminal=True,
        frozen_normalizer=True,
        fixed_body_stop_height=0.65,
        fixed_body_stop_upright=0.9,
        nonfoot_threshold_N=1.0,
        actor_all29_RNN_and_MLP_trainable=True,
        post_update_trust=a.post_update_trust,
        exact_TRAIN_global_KL=a.post_update_trust,
    )
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            train_cases=train,
            dev_cases=dev,
            old_cases=[
                dict(
                    source=r["source"],
                    case=r["case"],
                    ball=r["ball_world_initial"],
                    goal=r["goal_world"],
                )
                for r in base_cases
            ],
            reward=(
                "bodyfailure:-12; missing/wrongfoot/unobserved:-8-6*nonfoot; otherwise "
                "2*(6-min(12,target_distance3D_at_firstfoot10N_plus1.2s))-6*nonfoot"
                "-min(3,2*nonfoot_impulse)+4*clean-.5*retreat; "
                "terminal reward discounted by actual heldtail controls"
            ),
            commercial_publicity_authorized=False,
            promotion_authorized=False,
            hardware_authorized=False,
            original_formal52_unchanged=True,
            fresh_exam_claim=False,
        ),
    )
    episodes, history = [], []
    counts = dict(
        rollout_actor_rows=0,
        rollout_critic_rows=0,
        update_actor_rows=0,
        update_anchor_rows=0,
        update_critic_rows=0,
        optimizer_updates=0,
        actor_optimizer_updates=0,
        trust_actor_rows=0,
    )
    try:
        import copy
        import math

        import numpy as np
        import torch
        from torch import nn

        from rosclaw_soccer.rsi.native_contact_window import native_contact_window
        from rosclaw_soccer.rsi.paid_native_episode import PaidNativeEpisode
        from rosclaw_soccer.rsi.paid_recurrent_ppo import (
            backtracked_adam_update,
            finite_option_gae,
            recurrent_episode_loss,
        )
        from rosclaw_soccer.rsi.paid_teacher_policy import PaidResearchPolicy
        from rosclaw_soccer.rsi.paid_trainable_actor import PUBLISHED_HASH, PaidTrainableActor

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        rng = np.random.default_rng(config["seed"])
        sys.path.insert(0, str(a.reference_root))
        from exp.mujoco_soccer.constants import BODY_NAMES, JOINT_EFFORT_LIMIT_ARRAY
        from exp.mujoco_soccer.math_utils import quat_apply
        from exp.mujoco_soccer.motion import load_motion_clips
        from exp.mujoco_soccer.observations import build_observation
        from exp.mujoco_soccer.scene import MujocoSoccer

        policy_path = a.reference_root / "ckp/policy_30000.onnx"
        if pins.get(str(policy_path)) != PUBLISHED_HASH:
            raise ValueError("exact external research actor must be source-bound")
        contract = PaidResearchPolicy(policy_path, expected_hash=PUBLISHED_HASH, horizon=1)
        clips = {
            name: load_motion_clips(a.reference_root / "motions/soccer-standard" / name)[0]
            for name in names
        }
        xml = a.teacher_root / "cached-standard/declared-scene.xml"
        if str(xml) not in pins:
            raise ValueError("exact independently qualified standard physics required")
        envs = {name: MujocoSoccer(xml) for name in names}

        def make(case):
            name = case["source"]
            ep = PaidNativeEpisode(
                envs[name],
                clips[name],
                contract.parameters,
                contract.hg_from_policy,
                JOINT_EFFORT_LIMIT_ARRAY,
                build_observation,
                BODY_NAMES.index("torso_link"),
            )
            quat = clips[name].body_quat_w[0, 0].astype(np.float64)
            if "ball_local" in case:
                ball = quat_apply(quat, np.r_[case["ball_local"], 0.0])
                ball[2] = 0.11
                velocity = quat_apply(quat, np.r_[case["velocity_local"], 0.0])[:2]
                goal = quat_apply(quat, np.array([6.0, 0.0, 0.0]))
                goal[2] = 0.11
            else:
                ball, goal = np.array(case["ball_world_initial"]), np.array(case["goal_world"])
                local_velocity = (
                    [0, 0] if case["case"] % 2 == 0 else [-0.2, 0.1 if "left" in name else -0.1]
                )
                velocity = quat_apply(quat, np.r_[local_velocity, 0.0])[:2]
            ep.reset(ball=ball, velocity=velocity, goal=goal)
            episodes.append(ep)
            return ep

        # Admission of ALL proposed positions before training; no case can be
        # clipped, resampled or removed after its outcome becomes visible.
        for case in train + dev:
            make(case)
        # Exact physics/observation regression of the reusable environment on
        # all original16 trajectories; no inference or learning in this check.
        for case in base_cases:
            ep = make(case)
            path = a.teacher_root / f"cached-standard/case{case['case']}-repeat0.npz"
            with np.load(path, allow_pickle=False) as z:
                for frame in range(len(z["target"])):
                    action = None
                    if frame < len(z["NN_action"]):
                        if not np.array_equal(ep.observation(), z["NN_observation"][frame]):
                            raise ValueError("reusable environment changed causal observations")
                        action = z["NN_action"][frame].copy()
                    row = ep.step(action)
                    if any(
                        not np.array_equal(row[k], z[k][frame])
                        for k in ("qpos", "qvel", "forces", "torque", "target")
                    ):
                        raise ValueError("reusable environment changed actual teacher physics")
                if not ep.done:
                    raise ValueError("teacher source terminal boundary differs")
        write_once(
            a.output / "environment-qualified.json",
            dict(
                all16_raw_trajectories_exact=True,
                native_steps=sum(e.native_steps for e in episodes),
                new_neural_rows=0,
            ),
        )

        actor = PaidTrainableActor(policy_path, expected_hash=PUBLISHED_HASH).eval()
        anchor = copy.deepcopy(actor).eval()
        for parameter in anchor.parameters():
            parameter.requires_grad_(False)
        critic = nn.Sequential(
            nn.Linear(160, 128), nn.ELU(), nn.Linear(128, 64), nn.ELU(), nn.Linear(64, 1)
        )
        nn.init.zeros_(critic[-1].weight)
        nn.init.zeros_(critic[-1].bias)
        log_std = nn.Parameter(torch.full((29,), math.log(config["initial_std"])))
        parameters = list(actor.parameters()) + list(critic.parameters()) + [log_std]
        optimizer = torch.optim.Adam(parameters, lr=config["lr"])

        def checkpoint(iteration):
            path = a.output / f"recurrent-iteration{iteration}.pt"
            if path.exists():
                raise ValueError("checkpoint overwrite forbidden")
            state = dict(
                schema="soccer.paid_recurrent_research.v1",
                iteration=iteration,
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                config=config,
                actor=actor.state_dict(),
                anchor=anchor.state_dict(),
                critic=critic.state_dict(),
                log_std=log_std.detach().clone(),
                optimizer=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                optimizer_updates=counts["optimizer_updates"],
                actor_optimizer_updates=counts["actor_optimizer_updates"],
                native_steps=sum(e.native_steps for e in episodes),
                counts=counts.copy(),
                commercial_publicity_authorized=False,
                hardware_authorized=False,
            )
            torch.save(state, path)
            checked = torch.load(path, weights_only=True, map_location="cpu")
            if checked["schema"] != state["schema"]:
                raise ValueError("safe checkpoint reload failed")
            return digest(path)

        def collect(case, folder, stochastic):
            ep = make(case)
            state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
            controls, neural = [], []
            while not ep.done:
                action = None
                if ep.frame < ep.clip.length:
                    obs = ep.observation()
                    before = [s.numpy().copy() for s in state]
                    with torch.no_grad():
                        mean, state = actor(torch.from_numpy(obs).reshape(1, 1, 160), state)
                        counts["rollout_actor_rows"] += 1
                        if stochastic:
                            dist = torch.distributions.Normal(
                                mean[0, 0], log_std.clamp(math.log(0.01), math.log(0.05)).exp()
                            )
                            chosen = dist.sample()
                            logp = float(dist.log_prob(chosen).sum())
                            value = float(
                                critic(
                                    (torch.from_numpy(obs) - actor.mean) / actor.denominator
                                ).reshape(())
                            )
                            counts["rollout_critic_rows"] += 1
                        else:
                            chosen, logp, value = mean[0, 0], 0.0, 0.0
                    action = chosen.numpy().copy()
                    neural.append(
                        dict(
                            obs=obs,
                            action=action,
                            mean=mean[0, 0].numpy().copy(),
                            logp=logp,
                            value=value,
                            h_before=before[0],
                            c_before=before[1],
                            h_after=state[0].numpy().copy(),
                            c_after=state[1].numpy().copy(),
                        )
                    )
                controls.append(ep.step(action))
            raw = {k: np.array([row[k] for row in controls]) for k in controls[0]}
            raw.update({"NN_" + k: np.array([row[k] for row in neural]) for k in neural[0]})
            raw.update({"initial_" + k: v for k, v in ep.initial.items()})
            q, v, force = [
                raw[k].reshape(-1, width)
                for k, width in (("qpos", 43), ("qvel", 41), ("forces", 3))
            ]
            post = native_contact_window(
                force=force[:, :2],
                canonical_qpos=q,
                canonical_qvel=v,
                goal=ep.goal,
                intended_foot=0 if "left" in case["source"] else 1,
            )
            nonfoot = bool((force[:, 2] > 1).any())
            impulse = float((force[:, 2] * 0.002).sum())
            failed = bool(raw["body_failed"].any())
            if failed:
                reward = -12.0
            elif not (post["post_contact_quality_observed"] and post["correct_foot_first"]):
                reward = -8.0 - 6 * nonfoot
            else:
                index = int(post["first_foot_contact_substep"]) + 600
                distance = float(np.linalg.norm(q[index, 36:39] - ep.goal))
                reward = (
                    2 * (6 - min(12, distance))
                    - 6 * nonfoot
                    - min(3, 2 * impulse)
                    + 4 * (not nonfoot)
                    - 0.5 * post["maximum_pelvis_retreat_after_contact_m"]
                )
            adv, returns = finite_option_gae(
                raw["NN_value"].astype(np.float32), reward, len(controls) - len(neural)
            )
            raw["NN_advantage"], raw["NN_returns"] = adv, returns
            file = folder.with_suffix(".npz")
            np.savez_compressed(file, **raw)
            summary = dict(
                source=case["source"],
                variant=case.get("variant", case.get("case")),
                raw_hash=digest(file),
                native_steps=len(controls) * 10,
                neural_rows=len(neural),
                body_failed=failed,
                forbidden=nonfoot,
                nonfoot_impulse_Ns=impulse,
                terminal_reward=reward,
                postcontact=post,
            )
            data = {
                k: torch.from_numpy(raw["NN_" + k].astype(np.float32))
                for k in ("obs", "action", "mean", "logp", "value", "advantage", "returns")
            }
            return summary, data

        def exam(cases, name):
            folder = a.output / name
            folder.mkdir()
            summaries = [
                collect(case, folder / f"case{i}", False)[0] for i, case in enumerate(cases)
            ]
            write_once(
                folder / "complete.json",
                dict(results=summaries, goal_complete=False, commercial_publicity_authorized=False),
            )
            return summaries

        initial_cp = checkpoint(0)
        initial_dev = exam(dev, "initial-dev")
        for iteration in range(a.iterations):
            folder = a.output / f"iteration{iteration + 1}"
            folder.mkdir()
            samples = [collect(case, folder / f"case{i}", True) for i, case in enumerate(train)]
            advantage = torch.cat([data["advantage"] for _, data in samples])
            for _, data in samples:
                data["advantage"] = (data["advantage"] - advantage.mean()) / advantage.std(
                    unbiased=False
                ).clamp_min(1e-6)
            losses, batches = [], []
            value_only = iteration < config["critic_only_iterations"]
            if a.post_update_trust:
                for group in optimizer.param_groups:
                    group["lr"] = config["lr"]
            old_std = log_std.detach().clamp(math.log(0.01), math.log(0.05)).exp().clone()

            def exact_global_kl(samples=samples, old_std=old_std):
                weighted, rows = 0.0, 0
                with torch.no_grad():
                    std = log_std.clamp(math.log(0.01), math.log(0.05)).exp()
                    for _, data in samples:
                        obs = data["obs"]
                        mean, _ = actor(
                            obs[:, None], (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
                        )
                        counts["trust_actor_rows"] += len(obs)
                        # Exact diagonal-Gaussian old||new KL at each actual
                        # TRAIN prefix; roundoff at zero is clamped, not a
                        # changed acceptance threshold or altered ball metric.
                        kl = (
                            torch.log(std / old_std)
                            + (old_std.square() + (data["mean"] - mean[:, 0]).square())
                            / (2 * std.square())
                            - 0.5
                        ).sum(-1)
                        weighted += float(kl.clamp_min(0).sum())
                        rows += len(obs)
                return weighted / rows

            kl_stop = False
            for epoch in range(config["ppo_epochs"]):
                indices = rng.permutation(len(samples)).tolist()
                for start in range(0, len(indices), config["episode_minibatch_size"]):
                    batch = indices[start : start + config["episode_minibatch_size"]]
                    optimizer.zero_grad(set_to_none=True)
                    total_rows = sum(len(samples[i][1]["obs"]) for i in batch)
                    weighted_kl = 0.0
                    for i in batch:
                        data = samples[i][1]
                        loss, stats = recurrent_episode_loss(
                            actor, anchor, critic, log_std, data, value_only=value_only
                        )
                        weight = len(data["obs"]) / total_rows
                        (weight * loss).backward()
                        weighted_kl += weight * float(stats["approximate_kl"])
                        counts["update_actor_rows"] += stats["actor_rows"]
                        counts["update_anchor_rows"] += stats["anchor_rows"]
                        counts["update_critic_rows"] += stats["critic_rows"]
                        losses.append(float(loss.detach()))
                    if (
                        not value_only
                        and not a.post_update_trust
                        and weighted_kl > config["maximum_kl"]
                    ):
                        optimizer.zero_grad(set_to_none=True)
                        kl_stop = True
                        batches.append(
                            dict(epoch=epoch, episodes=batch, kl=weighted_kl, applied=False)
                        )
                        break
                    norm = torch.nn.utils.clip_grad_norm_(
                        parameters, config["max_grad_norm"], error_if_nonfinite=True
                    )
                    trust = None
                    if a.post_update_trust and not value_only:
                        trust = backtracked_adam_update(
                            optimizer, parameters, exact_global_kl, maximum_kl=config["maximum_kl"]
                        )
                        if not trust["applied"]:
                            optimizer.zero_grad(set_to_none=True)
                            kl_stop = True
                            batches.append(
                                dict(
                                    epoch=epoch,
                                    episodes=batch,
                                    kl=weighted_kl,
                                    applied=False,
                                    post_update_trust=trust,
                                )
                            )
                            break
                    else:
                        optimizer.step()
                    counts["optimizer_updates"] += 1
                    counts["actor_optimizer_updates"] += int(not value_only)
                    batches.append(
                        dict(
                            epoch=epoch,
                            episodes=batch,
                            kl=weighted_kl,
                            applied=True,
                            unclipped_grad_norm=float(norm),
                            post_update_trust=trust,
                        )
                    )
                    if a.post_update_trust and optimizer.param_groups[0]["lr"] < 1e-8:
                        kl_stop = True
                        break
                if kl_stop:
                    break
            if not torch.equal(actor.mean, anchor.mean) or not torch.equal(
                actor.denominator, anchor.denominator
            ):
                raise ValueError("frozen normalization changed")
            result = dict(
                iteration=iteration + 1,
                value_only=value_only,
                results=[s for s, _ in samples],
                optimizer_batches=batches,
                kl_stop=kl_stop,
                mean_loss=float(np.mean(losses)),
                counts=counts.copy(),
                native_steps=sum(e.native_steps for e in episodes),
                checkpoint_hash=checkpoint(iteration + 1),
            )
            write_once(folder / "complete.json", result)
            history.append(result)
            print(
                json.dumps(
                    dict(
                        iteration=iteration + 1,
                        counts=counts,
                        stable=sum(not s["body_failed"] for s, _ in samples),
                        clean=sum(
                            not s["body_failed"]
                            and not s["forbidden"]
                            and s["postcontact"]["correct_foot_first"]
                            for s, _ in samples
                        ),
                        kl_stop=kl_stop,
                    )
                ),
                flush=True,
            )
        final_dev = exam(dev, "final-dev")
        final_old = exam(base_cases, "final-old16")
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PAID_RECURRENT_PPO_RESEARCH_CLOSED_NOT_PROMOTED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                config=config,
                initial_checkpoint_hash=initial_cp,
                final_checkpoint_hash=digest(a.output / f"recurrent-iteration{a.iterations}.pt"),
                history=history,
                initial_dev=initial_dev,
                final_dev=final_dev,
                final_old16=final_old,
                native_steps=sum(e.native_steps for e in episodes),
                counts=counts,
                original_formal52_unchanged=True,
                goal_complete=False,
                promotion_authorized=False,
                commercial_publicity_authorized=False,
                hardware_authorized=False,
                final_rollouts_not_yet_independently_replayed=True,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                native_steps=sum(e.native_steps for e in episodes),
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
