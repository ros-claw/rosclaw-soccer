"""Bounded SIM learning across earlier approach states and measured recovery.

New task/critic/optimizer, actor initialized ONLY from the fixed132 research
candidate. Never promotes that candidate or accepts old24 qualification alone.
Fixed8 critic warmup+32PPO episodes, no DEV data or checkpoint selection.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    p = argparse.ArgumentParser(description=__doc__)
    for key in ("course", "fault", "parent"):
        p.add_argument("--" + key + "-root", type=Path, required=True)
        p.add_argument("--" + key + "-hash", required=True)
    for key in ("course-audit", "fault-audit"):
        p.add_argument("--" + key, type=Path, required=True)
        p.add_argument("--" + key + "-hash", required=True)
    p.add_argument("--reference-root", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--mode", choices=("smoke2", "train40"), required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("unused output and frozen clean source required")
    pins, reports = {}, {}
    for key in ("course", "fault", "parent"):
        root, h = getattr(a, key + "_root"), getattr(a, key + "_hash")
        path = root / "complete.json"
        if digest(path) != h:
            raise ValueError("exact completed prerequisite required: " + key)
        report = json.loads(path.read_text())
        pins.update(report["input_file_hashes"])
        pins[str(path)] = h
        reports[key] = report
        if key != "parent":
            dp = root / "declaration.json"
            pins[str(dp)] = digest(dp)
            reports[key + "_declaration"] = json.loads(dp.read_text())
            for file, hash_value in report["raw_hashes"].items():
                pins[str(root / file)] = hash_value
    for key in ("course", "fault"):
        file, h = getattr(a, key + "_audit"), getattr(a, key + "_audit_hash")
        if digest(file) != h:
            raise ValueError("exact independent raw audit required")
        audit = json.loads(file.read_text())
        if (
            audit["schema"] != "soccer.rsi.native_approach_independent_audit.v1"
            or audit["input_file_hashes"].get(str(getattr(a, key + "_root") / "complete.json"))
            != getattr(a, key + "_hash")
            or audit["measured_handoff_and_body_failure_exact"] is not True
            or audit["whole_repeats_exact"] is not True
        ):
            raise ValueError("audit does not prove this actual new approach source")
        pins.update(audit["input_file_hashes"])
        pins[str(file)] = h
        reports[key + "_audit"] = audit
    for key in ("course", "fault"):
        report = reports[key]
        if report["status"] != "NATIVE_APPROACH_MEASURED_CONTACT_RECOVERY_PROBE_NOT_LEARNER" or any(
            report[k] is not True
            for k in (
                "continuous_handoff_exact",
                "strict_replays_exact",
                "full_repeats_exact",
                "stopped_worlds_frozen",
            )
        ):
            raise ValueError("complete new native approach mechanics required")
    cd, fd = reports["course_declaration"], reports["fault_declaration"]
    if (
        cd["worlds"] != 64
        or cd["leads"] != [10, 20, 40, 80]
        or cd.get("fault_injection") is not None
        or cd.get("foundation_inference_batch_rows") != 1
        or reports["course_audit"]["original_full_reference_prefix_controls_exact"] < 1
        or fd["worlds"] != 8
        or fd["leads"] != [80]
        or not fd.get("fault_injection")
    ):
        raise ValueError(
            "full256 reset coverage and actual separate head/tail fault proof required"
        )
    parent = reports["parent"]
    if (
        parent["status"] != "NATIVE_OPTION_CONTINUATION132_REQUIRES_FULL_EXAM"
        or parent["end_episode"] != 132
        or [x["episode"] for x in parent["results"]] != list(range(20, 132))
    ):
        raise ValueError("only fixed actual132 research actor initialization allowed")
    cps = [x for x in parent["checkpoints"] if x["episode"] == 132]
    if len(cps) != 1:
        raise ValueError("one fixed parent132 checkpoint required")
    checkpoint = a.parent_root / cps[0]["file"]
    pins[str(checkpoint)] = cps[0]["hash"]
    for relative in (
        "rsi/native_approach_contact.py",
        "rsi/native_contact_course.py",
        "rsi/native_recovery_tail.py",
        "rsi/native_physics_batch.py",
    ):
        file = Path.cwd() / "src/rosclaw_soccer" / relative
        existing = {
            h
            for path, h in reports["course"]["input_file_hashes"].items()
            if path.endswith("/src/rosclaw_soccer/" + relative)
        }
        if digest(file) not in existing:
            raise ValueError("new course implementation differs from qualified source")
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in folder.rglob("*.py")})
    check_pins(pins)
    end = 2 if a.mode == "smoke2" else 40
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            protocol="native-approach-measured-contact-recovery60-ppo-v1",
            mode=a.mode,
            training_seed=7608,
            worlds=32,
            episodes=end,
            value_only_episodes=min(end, 8),
            PPO_episodes=max(end - 8, 0),
            maximum_new_native_steps=end * 32 * 360 * 10,
            maximum_optimizer_updates=end * 16,
            lead_schedule="(10,20,40,80)[episode//2%4]",
            gamma=0.995,
            GAE_lambda=0.95,
            head_rewards="approach_head_penalty: no alive bonus",
            terminal_reward="actual recovery_terminal_reward at last actual recovery control",
            actor_init_parent132_hash=cps[0]["hash"],
            fresh_critic=True,
            fresh_optimizer=True,
            optimizer_resumed=False,
            old_optimizer_or_critic_inherited=False,
            inactive_worlds_no_physics_no_NN=True,
            far_ball_actions_deterministic=True,
            near_exploration_scaled_by_actual_gate=True,
            foundation_recovery_cpu_single_row=True,
            no_DEV_or_privateFresh=True,
            checkpoint_episodes=([0, 2] if end == 2 else [0, 8, 24, 40]),
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )
    env = None
    counts = dict(
        optimizer_attempted=0,
        optimizer_completed=0,
        value_only_steps=0,
        PPO_steps=0,
        actor_update_steps=0,
        kl_skips=0,
        head_policy_rows=0,
        head_critic_rows=0,
        recovery_foundation_rows=0,
        optimizer_critic_rows=0,
        optimizer_plastic_rows=0,
    )
    history, tail_trace = [], []
    try:
        import copy
        import math

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.rsi.approach_option_credit import (
            approach_head_penalty,
            approach_option_gae,
        )
        from rosclaw_soccer.rsi.approach_option_ppo import approach_ppo_loss, approach_sample
        from rosclaw_soccer.rsi.approach_option_rollout import finish_approach_option
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
        from rosclaw_soccer.rsi.native_approach_contact import NativeApproachContactCourse
        from rosclaw_soccer.rsi.native_contact_checkpoint import NativeContinuedContactSession
        from rosclaw_soccer.rsi.native_contact_ppo import make_contact_actor_critic
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from scripts.rsi_native_course_inputs import load_body_references

        torch.set_num_threads(1)
        torch.manual_seed(7608)
        np.random.seed(7608)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required for bounded neural optimization")

        def pinned(suffix):
            found = [Path(x) for x in pins if x.endswith(suffix)]
            if len(found) != 1:
                raise ValueError("one exact actual source required: " + suffix)
            return found[0]

        def load(path):
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
        config = pinned("/policy/velocity/v0/params/deploy.yaml")
        mp = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
        original_parent = pinned("/ball-aware-update200.pt")
        session = NativeContinuedContactSession(
            onnx,
            config,
            expected_model_hash=pins[str(onnx)],
            expected_config_hash=pins[str(config)],
            checkpoint=checkpoint,
            expected_checkpoint_hash=pins[str(checkpoint)],
            parent_checkpoint=original_parent,
            expected_parent_hash=pins[str(original_parent)],
            expected_native_model_hash=pins[str(mp)],
            expected_resume_checkpoint_hash=parent["resume_checkpoint_hash"],
            expected_resume_proof_hash=parent["resume_proof_hash"],
            expected_resume_source_hash=parent["resume_source_hash"],
            expected_recovery_proof_hash=parent["recovery_proof_hash"],
            expected_recovery_fault_hash=parent["recovery_fault_hash"],
            arm="final",
        )
        actor = copy.deepcopy(session.model)
        if sum(par.requires_grad for par in actor.adapter.parameters()) != 6 or any(
            par.requires_grad for par in actor.base.parameters()
        ):
            raise ValueError("exactly six plastic adapter tensors and immutable base required")
        # Fresh statistics only for the changed task. No rejected optimizer/RNG
        # or short-task critic is quietly reused.
        policy = make_contact_actor_critic(actor).cuda()
        foundation = copy.deepcopy(session.model.base).cpu().eval()
        for par in foundation.parameters():
            par.requires_grad_(False)
        baseline_before = {k: v.cpu().clone() for k, v in policy.baseline.state_dict().items()}
        base_before = {k: v.cpu().clone() for k, v in policy.actor.base.state_dict().items()}
        actor_initial = {k: v.cpu().clone() for k, v in policy.actor.state_dict().items()}
        std_initial = policy.log_std.detach().cpu().clone()
        optimizer = torch.optim.Adam(
            [
                dict(
                    params=[par for par in policy.actor.parameters() if par.requires_grad], lr=3e-4
                ),
                dict(params=policy.critic.parameters(), lr=3e-4),
                dict(params=[policy.log_std], lr=1e-5),
            ]
        )
        model = mujoco.MjModel.from_binary_path(str(mp))
        layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
        bank = load(pinned("/precontact-resets.npz"))
        plans = [
            load(
                pinned(
                    "/rsi-train-starts-body-bound-index-check-20261021/"
                    + n.removesuffix(".npz")
                    + "-plan.npz"
                )
            )
            for n in TRAIN_MOTION_NAMES
        ]
        refs = load_body_references(a.reference_root, model, layout, plans)
        env = NativeApproachContactCourse(
            model=model,
            params=session.params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=32,
            lead=10,
            threads=8,
        )
        checkpoints = []

        def save(episode):
            file = a.output / f"approach-episode{episode}.pt"
            state = dict(
                schema="soccer.rsi.native_approach_contact_recovery_ppo.v1",
                protocol="native-approach-measured-contact-recovery60-ppo-v1",
                episode_count=episode,
                source_head=a.expected_head,
                training_seed=7608,
                mode=a.mode,
                actor_observation_dim=177,
                critic_observation_dim=177,
                native_pd_position_targets=29,
                initialization_checkpoint_hash=pins[str(checkpoint)],
                original_onnx_hash=pins[str(onnx)],
                parameter_hash=pins[str(config)],
                native_model_hash=pins[str(mp)],
                reset_bank_hash=pins[str(pinned("/precontact-resets.npz"))],
                course_proof_hash=a.course_hash,
                fault_proof_hash=a.fault_hash,
                course_audit_hash=a.course_audit_hash,
                fault_audit_hash=a.fault_audit_hash,
                fresh_critic=True,
                fresh_optimizer=True,
                optimizer_resumed=False,
                activation_ceiling="SIM_ONLY",
                promotion_authorized=False,
                hardware_authorized=False,
                actor_state_dict=policy.actor.state_dict(),
                policy_state_dict=policy.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                torch_rng=torch.get_rng_state(),
                cuda_rng=[x.cpu() for x in torch.cuda.get_rng_state_all()],
                counts=dict(counts),
                native_steps=env.native_steps,
                course_episode=env.episode,
                gamma=0.995,
                GAE_lambda=0.95,
                maximum_normalized_residual=0.2,
            )
            torch.save(state, file)
            read = torch.load(file, map_location="cpu", weights_only=True)
            if any(
                not torch.equal(v.detach().cpu(), read["policy_state_dict"][k])
                for k, v in policy.state_dict().items()
            ):
                raise ValueError("safe checkpoint tensor readback differs")
            checkpoints.append(dict(episode=episode, file=file.name, hash=digest(file)))

        save(0)
        results = []
        for episode in range(end):
            env.lead = (10, 20, 40, 80)[episode // 2 % 4]
            env.reset()
            history, tail_trace = [], []
            while not env.ended.all():
                ids = np.flatnonzero(~env.ended)
                sample = dict(
                    action=np.zeros((32, 29), dtype=np.float32),
                    old_logp=np.zeros(32, dtype=np.float32),
                    old_value=np.zeros(32, dtype=np.float32),
                    plastic=np.zeros(32, dtype=bool),
                )
                with torch.no_grad():
                    values = approach_sample(policy, torch.from_numpy(env.observation[ids]).cuda())
                for key, tensor in values.items():
                    sample[key][ids] = tensor.cpu().numpy()
                counts["head_policy_rows"] += len(ids)
                counts["head_critic_rows"] += len(ids)
                step = env.step(sample["action"])
                step["reward"] = approach_head_penalty(step)
                next_value = np.zeros(32, dtype=np.float32)
                following = np.flatnonzero(~step["ended"])
                if len(following):
                    with torch.no_grad():
                        next_value[following] = (
                            policy.value(torch.from_numpy(step["observation"][following]).cuda())
                            .cpu()
                            .numpy()
                        )
                    counts["head_critic_rows"] += len(following)
                history.append(dict(**step, **sample, next_value=next_value))
            raw = {k: np.stack([row[k] for row in history]) for k in history[0]}
            folder = a.output / f"episode{episode:02d}"
            folder.mkdir()
            file = folder / "head.npz"
            np.savez_compressed(file, bank_index=env.index, **raw)
            tail, outcomes = finish_approach_option(
                env, raw, foundation, counters=counts, trace=tail_trace
            )
            tailfile = folder / "tail.npz"
            np.savez_compressed(tailfile, bank_index=env.index, **tail)
            advantage, returns = approach_option_gae(
                raw["reward"],
                raw["old_value"],
                raw["next_value"],
                raw["terminated"],
                raw["truncated"],
                raw["valid"],
                tail_reward=tail["reward"],
                tail_valid=tail["valid"],
                tail_terminal=tail["terminal"],
            )
            mask = raw["valid"].reshape(-1)
            data = {
                k: torch.from_numpy(v.reshape((-1, *v.shape[2:]))[mask].copy()).cuda()
                for k, v in dict(
                    obs=raw["observation_before"],
                    action=raw["action"],
                    old_logp=raw["old_logp"],
                    old_value=raw["old_value"],
                    plastic=raw["plastic"],
                    advantage=advantage,
                    returns=returns,
                ).items()
            }
            near = data["plastic"]
            if near.any():
                adv = data["advantage"][near]
                data["advantage"][near] = (adv - adv.mean()) / (adv.std(unbiased=False) + 1e-8)
            updates = []
            stop, warmup = False, episode < 8
            for _ in range(4):
                if stop:
                    break
                for mb in torch.tensor_split(torch.randperm(int(mask.sum()), device="cuda"), 4):
                    if not len(mb):
                        continue
                    loss, stats = approach_ppo_loss(
                        policy, {k: v[mb] for k, v in data.items()}, value_only=warmup
                    )
                    counts["optimizer_critic_rows"] += len(mb)
                    counts["optimizer_plastic_rows"] += int(stats["plastic_rows"])
                    if not warmup and float(stats["approximate_kl"]) > 0.02:
                        counts["kl_skips"] += 1
                        stop = True
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(
                        [par for group in optimizer.param_groups for par in group["params"]], 1.0
                    )
                    if not torch.isfinite(norm):
                        raise ValueError("nonfinite actual approach gradient")
                    counts["optimizer_attempted"] += 1
                    optimizer.step()
                    counts["optimizer_completed"] += 1
                    counts["value_only_steps" if warmup else "PPO_steps"] += 1
                    counts["actor_update_steps"] += int(
                        not warmup and bool(stats["plastic_rows"] > 0)
                    )
                    with torch.no_grad():
                        policy.log_std.clamp_(math.log(0.01), math.log(0.1))
                    if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
                        raise ValueError("nonfinite actual approach weights")
                    if any(
                        not torch.equal(v.cpu(), base_before[k])
                        for k, v in policy.actor.base.state_dict().items()
                    ) or any(
                        not torch.equal(v.cpu(), baseline_before[k])
                        for k, v in policy.baseline.state_dict().items()
                    ):
                        raise ValueError("immutable backbone/initial132 anchor changed")
                    updates.append({k: float(v) for k, v in stats.items()})
            if warmup and (
                any(
                    not torch.equal(v.cpu(), actor_initial[k])
                    for k, v in policy.actor.state_dict().items()
                )
                or not torch.equal(policy.log_std.cpu(), std_initial)
                or any(par in optimizer.state for par in policy.actor.parameters())
                or policy.log_std in optimizer.state
            ):
                raise ValueError("critic warmup touched actual actor or exploration")
            row = dict(
                episode=episode,
                lead=env.lead,
                head_hash=digest(file),
                tail_hash=digest(tailfile),
                native_steps=env.native_steps,
                head_rows=int(mask.sum()),
                plastic_rows=int(near.sum()),
                counts=dict(counts),
                updates=updates,
                outcomes=outcomes,
            )
            write_once(folder / "result.json", row)
            results.append(row)
            print(
                json.dumps(
                    dict(
                        episode=episode,
                        lead=env.lead,
                        native_steps=env.native_steps,
                        updates=counts["optimizer_completed"],
                        actor_updates=counts["actor_update_steps"],
                        positive_reward=sum(x["recovery_reward"] > 0 for x in outcomes),
                        quality_reward=sum(x["recovery_reward"] >= 10 for x in outcomes),
                    )
                ),
                flush=True,
            )
            if episode + 1 in ([2] if end == 2 else [8, 24, 40]):
                save(episode + 1)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE_APPROACH_SMOKE_CRITIC_ONLY"
                if end == 2
                else "NATIVE_APPROACH40_REQUIRES_FULL_EXAM",
                input_file_hashes=pins,
                protocol="native-approach-measured-contact-recovery60-ppo-v1",
                end_episode=end,
                mode=a.mode,
                results=results,
                checkpoints=checkpoints,
                counts=counts,
                native_steps=env.native_steps,
                optimizer_updates=counts["optimizer_completed"],
                actor_update_steps=counts["actor_update_steps"],
                fresh_critic=True,
                fresh_optimizer=True,
                actor_init_parent132_hash=pins[str(checkpoint)],
                training_seed=7608,
                full_native_exam_passed=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        import numpy as np

        for name, trace in (("failed-head", history), ("failed-tail", tail_trace)):
            if trace:
                np.savez_compressed(
                    a.output / (name + ".npz"),
                    **{k: np.stack([x[k] for x in trace]) for k in trace[0]},
                )
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                counts=counts,
                native_steps=env.native_steps if env else 0,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
