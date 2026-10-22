"""First bounded real AR(1) actor-critic update over audited pure-learner options.

Four PPO optimization epochs on ONE actual on-policy batch, not four new
rollout generations. All failures stay in the objective. Final-only model.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ("experience-root", "audit-root", "output"):
        p.add_argument("--" + n, type=Path, required=True)
    for n in ("audit-hash", "expected-head"):
        p.add_argument("--" + n, required=True)
    p.add_argument("--failed-predecessor-root", type=Path)
    p.add_argument("--failed-predecessor-hash")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    def clean():
        if (
            subprocess.check_output(["git", "status", "--porcelain"])
            or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
            != a.expected_head
        ):
            raise ValueError("clean frozen actual actor-critic source required")

    clean()
    rp, ap, dp = (
        a.experience_root / "complete.json",
        a.audit_root / "complete.json",
        a.experience_root / "declaration.json",
    )
    if a.output.exists() or any(x.is_symlink() for x in (rp, ap, dp)) or digest(ap) != a.audit_hash:
        raise ValueError("exact closed experience and independent own-native/law audit required")
    r, ar, d = [json.loads(x.read_text()) for x in (rp, ap, dp)]
    from rosclaw_soccer.rsi.motor_exploration_curriculum import declared_motor_curriculum

    curriculum = declared_motor_curriculum(d)
    if (
        ar["status"] != curriculum.audit_status
        or ar["source_hash"] != digest(rp)
        or ar["counts"]["native"] != r["actual_native_steps"]
        or ar["counts"]["neural"] != r["actual_neural_decisions"]
        or r["declaration_hash"] != digest(dp)
        or len(r["records"]) != curriculum.worlds
        or len(ar["results"]) != curriculum.worlds
        or d["rho"] != 0.9
        or d["sigma_policy_units"] != 0.005
        or any(x["rejected"] for x in ar["results"])
        or any(x["rejection"] is not None or x["assisted"] for x in r["records"])
        or any(
            v[k] is not False
            for v in (ar, r, d)
            for k in ("hardware_authorized", "promotion_authorized")
        )
    ):
        raise ValueError("all eight unmodified audited actually executed options required")
    pins = dict(ar["input_file_hashes"])
    for x in (rp, ap, dp):
        pins[str(x)] = digest(x)
    if bool(a.failed_predecessor_root) != bool(a.failed_predecessor_hash):
        raise ValueError("exact failed-predecessor binding required")
    if a.failed_predecessor_root is not None:
        failure_path = a.failed_predecessor_root / "failed.json"
        if failure_path.is_symlink() or digest(failure_path) != a.failed_predecessor_hash:
            raise ValueError("exact preserved failed training receipt required")
        failure = json.loads(failure_path.read_text())
        if (
            failure["accepted_optimizer_steps"] != 0
            or failure["error"] != "ValueError('invalid whole-history AR likelihood ratio')"
        ):
            raise ValueError("recovery only fixes the pre-acceptance KL evaluation fault")
        pins[str(failure_path)] = a.failed_predecessor_hash
        predecessor_decl = a.failed_predecessor_root / "declaration.json"
        predecessor = json.loads(predecessor_decl.read_text())
        if predecessor["experience_audit_hash"] != a.audit_hash:
            raise ValueError("recovery requires the same still-on-policy actual batch")
        pins[str(predecessor_decl)] = digest(predecessor_decl)

    def receipt(h):
        paths = [Path(x) for x, v in pins.items() if v == h and x.endswith("/complete.json")]
        if len(paths) != 1:
            raise ValueError("unique exact predecessor required")
        return paths[0], json.loads(paths[0].read_text())

    wp, warm = receipt(r["training_hash"])
    continual = d.get("checkpoint_kind", "supervised") == "ar1-actor-critic"
    if continual and (
        d.get("generation_index") != warm["config"].get("source_generation_index", 0) + 1
        or a.failed_predecessor_root is not None
    ):
        raise ValueError("fresh next-generation data required for exact continual warm start")
    pp, parent_record = receipt(warm["parent_lineage_hash"])
    data_path, data = receipt(warm["data_hash"])
    for folder in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(x): digest(x) for x in folder.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    config = dict(
        protocol="dual256.AR1_continual_actor_critic.v2"
        if continual
        else "dual256.AR1_audited_option_actor_critic.v1",
        epochs=4,
        actor_lr=3e-5,
        critic_lr=0.001,
        rho=0.9,
        sigma_policy=0.005,
        max_conditional_kl=0.015,
        old14_anchor_weight=0.02,
        seed=74220,
        gamma=0.995,
        neural_standing_delay_observed=True,
        whole_private_history_from_zero=True,
        final_only=True,
        source_batch_generations=1,
        reward="coarse_contact_skill_return.v1_NOT_goal_plane_or_team_score",
        device="cpu",
        source_batch_size=curriculum.worlds,
        source_curriculum=curriculum.name,
    )
    if continual:
        config.update(
            source_generation_index=d["generation_index"],
            optimizer_state_carried=True,
            critic_state_carried=True,
            learning_rates="exact_predecessor_effective_rates_not_reset",
        )
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=config,
            experience_hash=digest(rp),
            experience_audit_hash=a.audit_hash,
            implementation_recovery_failure_hash=a.failed_predecessor_hash,
            warm_start_hash=r["training_hash"],
            parent_lineage_hash=warm["parent_lineage_hash"],
            data_hash=warm["data_hash"],
            actor_training_admitted=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    accepted = 0
    epochs = []
    queries = 0
    try:
        import copy
        import math

        import numpy as np
        import torch

        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import (
            load_dual_candidate_for_sim_audit,
            same_training_state,
        )
        from rosclaw_soccer.rsi.dual_recurrent_actor_critic import (
            MotorCritic,
            own_history_conditional_kl,
            own_history_loss,
        )
        from rosclaw_soccer.rsi.motor_option_credit import (
            contact_skill_return,
            terminal_option_returns,
        )
        from rosclaw_soccer.rsi.paid_progressive_student import load_paid_student
        from rosclaw_soccer.rsi.paid_recurrent_ppo import backtracked_adam_update
        from rosclaw_soccer.rsi.paid_research_setup import open_paid_bench

        torch.set_num_threads(1)
        torch.set_num_interop_threads(1)
        torch.manual_seed(config["seed"])
        torch.use_deterministic_algorithms(True)
        _, policy, scene = open_paid_bench(pins, [x["case"] for x in d["contexts"]])
        parent, _ = load_paid_student(
            policy,
            checkpoint=Path(parent_record["checkpoint_path"]),
            checkpoint_hash=parent_record["checkpoint_hash"],
            training_receipt=pp,
            training_receipt_hash=warm["parent_lineage_hash"],
            scene_hash=scene,
        )
        model, _ = load_dual_candidate_for_sim_audit(
            parent,
            parent_lineage_hash=warm["parent_lineage_hash"],
            root=wp.parent,
            expected_receipt_hash=r["training_hash"],
            checkpoint_kind=d.get("checkpoint_kind", "supervised"),
        )
        model.plastic.requires_grad_(True)
        critic = MotorCritic().eval()
        warm_state = None
        initial_steps = 0
        if continual:
            warm_path = Path(warm["checkpoint_path"])
            if digest(warm_path) != warm["checkpoint_hash"]:
                raise ValueError("exact validated continual critic/optimizer source required")
            warm_state = torch.load(warm_path, map_location="cpu", weights_only=True)
            critic.load_state_dict(warm_state["critic"], strict=True)
            initial_steps = warm.get("cumulative_optimizer_steps", warm["accepted_optimizer_steps"])
        log_std = torch.full((29,), math.log(0.005), dtype=torch.float32)
        frozen = {k: v.clone() for k, v in parent.state_dict().items()}
        initial = {k: v.clone() for k, v in model.plastic.state_dict().items()}
        critic_initial = {k: v.clone() for k, v in critic.state_dict().items()}
        episodes = []
        sources = []

        def npz(path, h):
            if path.is_symlink() or digest(path) != h:
                raise ValueError("exact actual history required")
            pins[str(path)] = h
            with np.load(path, allow_pickle=False) as z:
                return {k: z[k].copy() for k in z.files}

        for row in r["records"]:
            blobs = {
                name: npz(a.experience_root / art["path"], art["hash"])
                for name, art in row["artifacts"].items()
            }
            kick, stand = blobs["kick"], blobs.get("stand")
            n = len(kick["NN_obs"])
            if (
                stand is None
                or len(stand["qpos"]) != 100
                or kick["body_failed"].any()
                or not row["stand"]["healthy"]
            ):
                raise ValueError(
                    "pilot requires complete actual option with observed neural standing"
                )
            reward = contact_skill_return(row["result"])
            returns = terminal_option_returns(n, 100, reward)
            ep = dict(
                observations=torch.from_numpy(kick["NN_obs"]),
                actions=torch.from_numpy(kick["NN_action"]),
                old_means=torch.from_numpy(kick["NN_nominal_action"]),
                old_logp=torch.from_numpy(kick["AR_logp"]),
                old_values=torch.zeros(n),
                returns=torch.from_numpy(returns),
                advantages=torch.from_numpy(returns.copy()),
                assisted=False,
                all_actor_actions_actually_executed=True,
                authority_rejected=False,
                natural_terminal_observed=True,
            )
            # Recompute the PREDECESSOR critic on the actual policy history.
            # No teacher h/c or future weights enter this fixed old baseline.
            with torch.no_grad():
                private = model.initial_state()
                values = []
                for i, actual in enumerate(ep["observations"]):
                    if not np.array_equal(
                        private[0].numpy(), kick["NN_h_before"][i]
                    ) or not np.array_equal(private[1].numpy(), kick["NN_c_before"][i]):
                        raise ValueError(
                            "own actual old-value history differs from behavior policy"
                        )
                    mean, private, e = model.forward_with_evidence(
                        actual.reshape(1, 1, 160), private
                    )
                    if not np.array_equal(mean[0, 0].numpy(), kick["NN_nominal_action"][i]):
                        raise ValueError("old values must use exact actual behavior weights")
                    values.append(critic(e["critic_features"][:, 0])[0])
                old_values = torch.stack(values)
                if not continual and old_values.count_nonzero():
                    raise ValueError("declared initial critic not zero")
                if not torch.isfinite(old_values).all():
                    raise ValueError("finite persistent critic baseline required")
            ep["old_values"] = old_values
            ep["advantages"] = ep["returns"] - old_values
            episodes.append(ep)
            sources.append(
                dict(
                    context=row["context"],
                    seed=row["seed"],
                    terminal_reward=reward,
                    rows=n,
                    standing_delay=100,
                    initial_value_mean=float(old_values.mean()),
                    baseline_recomputed_from_own_behavior_history=True,
                )
            )
        advantages = torch.cat([ep["advantages"] for ep in episodes])
        av, sd = advantages.mean(), advantages.std(unbiased=False).clamp_min(1e-8)
        for ep in episodes:
            ep["advantages"] = (ep["advantages"] - av) / sd
        for i, ep in enumerate(episodes):
            path = a.output / f"credit{i}.npz"
            np.savez_compressed(
                path,
                old_values=ep["old_values"].numpy(),
                returns=ep["returns"].numpy(),
                advantages=ep["advantages"].numpy(),
            )
            sources[i].update(credit_path=path.name, credit_hash=digest(path))
        old = []
        for row in data["records"]:
            if row["role"] == "acquired_parent_skill":
                z = npz(data_path.parent / row["raw_path"], row["raw_hash"])
                old.append({k: torch.from_numpy(z[k]) for k in ("observations", "targets")})
        if len(old) != 14:
            raise ValueError("all14 original acquired skills must enter retention anchor")
        actor_params = list(model.plastic_parameters())
        critic_params = list(critic.parameters())
        optimizer = torch.optim.Adam(
            [
                dict(params=actor_params, lr=config["actor_lr"]),
                dict(params=critic_params, lr=config["critic_lr"]),
            ]
        )
        if continual:
            optimizer.load_state_dict(copy.deepcopy(warm_state["optimizer"]))
            if not same_training_state(optimizer.state_dict(), warm_state["optimizer"]):
                raise ValueError("optimizer moments/counters/effective LRs did not carry exactly")

        def cpu_tree(x):
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().clone()
            if isinstance(x, dict):
                return {k: cpu_tree(v) for k, v in x.items()}
            if isinstance(x, list):
                return [cpu_tree(v) for v in x]
            if isinstance(x, tuple):
                return tuple(cpu_tree(v) for v in x)
            return copy.deepcopy(x)

        first = a.output / "onpolicy-initial.pt"
        torch.save(
            dict(
                schema="soccer.dual_motor.ar1_ppo.v2"
                if continual
                else "soccer.dual_motor.ar1_ppo.v1",
                plastic_network=initial,
                critic=critic_initial,
                optimizer_updates=initial_steps,
                optimizer=cpu_tree(optimizer.state_dict()),
                warm_start_hash=r["training_hash"],
                parent_lineage_hash=warm["parent_lineage_hash"],
            ),
            first,
        )

        def evaluate_kl():
            with torch.no_grad():
                return sum(
                    own_history_conditional_kl(model, ep, log_std, rho=0.9) for ep in episodes
                ) / len(episodes)

        for epoch in range(4):
            optimizer.zero_grad(set_to_none=True)
            losses = []
            for ep in episodes:
                loss, _ = own_history_loss(model, critic, ep, log_std, rho=0.9)
                (loss / len(episodes)).backward()
                losses.append(float(loss.detach()))
            anchors = []
            for ep in old:
                means, _ = model(ep["observations"][:, None], model.initial_state())
                anchor = 0.5 * ((means[:, 0] - ep["targets"]) / 0.02).square().sum(-1).mean()
                (0.02 * anchor / 14).backward()
                anchors.append(float(anchor.detach()))
            ga = torch.nn.utils.clip_grad_norm_(actor_params, 1.0)
            gc = torch.nn.utils.clip_grad_norm_(critic_params, 1.0)
            if not torch.isfinite(ga) or not torch.isfinite(gc):
                raise ValueError("nonfinite actor/critic gradient")
            decision = backtracked_adam_update(
                optimizer, actor_params + critic_params, evaluate_kl, maximum_kl=0.015
            )
            accepted += int(decision["applied"])
            info = dict(
                epoch=epoch + 1,
                accepted_steps=accepted,
                mean_loss=sum(losses) / len(episodes),
                old14_anchor=sum(anchors) / 14,
                actor_gradient=float(ga),
                critic_gradient=float(gc),
                decision=decision,
            )
            epochs.append(info)
            write_once(a.output / f"epoch{epoch + 1}.json", info)
            print(json.dumps(info), flush=True)
        if not accepted:
            raise ValueError("no accepted actor-critic update; no learned candidate")
        final = cpu_tree(model.plastic.state_dict())
        critic_final = cpu_tree(critic.state_dict())
        if any(not torch.equal(v, frozen[k]) for k, v in parent.state_dict().items()):
            raise ValueError("frozen parent changed")
        if any(not torch.equal(final[k], initial[k]) for k in ("mean", "denominator")):
            raise ValueError("normalization changed")
        changed = [k for k, v in final.items() if not torch.equal(v, initial[k])]
        changed_critic = [
            k for k, v in critic_final.items() if not torch.equal(v, critic_initial[k])
        ]
        if len(changed) != 16 or not changed_critic:
            raise ValueError("actual actor RNN and critic changes required")
        last = a.output / "onpolicy-final.pt"
        torch.save(
            dict(
                schema="soccer.dual_motor.ar1_ppo.v2"
                if continual
                else "soccer.dual_motor.ar1_ppo.v1",
                plastic_network=final,
                critic=critic_final,
                optimizer=cpu_tree(optimizer.state_dict()),
                optimizer_updates=initial_steps + accepted,
                config=config,
                warm_start_hash=r["training_hash"],
                parent_lineage_hash=warm["parent_lineage_hash"],
            ),
            last,
        )
        model.eval().requires_grad_(False)
        replay = []
        for row in data["records"]:
            z = npz(data_path.parent / row["raw_path"], row["raw_hash"])
            state = model.initial_state()
            means = []
            for obs in z["observations"]:
                with torch.no_grad():
                    mean, state = model(torch.from_numpy(obs).reshape(1, 1, 160), state)
                queries += 1
                means.append(mean[0, 0].numpy().copy())
            means = np.stack(means)
            path = a.output / f"cpu-sequence{row['index']}.npz"
            np.savez_compressed(path, means=means)
            replay.append(
                dict(
                    index=row["index"],
                    role=row["role"],
                    rows=len(means),
                    motor_MSE=float(np.mean(((means - z["targets"]) * z["action_scale"]) ** 2)),
                    prediction_path=path.name,
                    prediction_hash=digest(path),
                )
            )
        check_pins(pins)
        clean()
        write_once(
            a.output / "complete.json",
            dict(
                status="DUAL_AR1_ACTOR_CRITIC_ACTUALLY_UPDATED_PENDING_PHYSICAL_EXAM",
                source_head=a.expected_head,
                input_file_hashes=pins,
                config=config,
                declaration_hash=digest(a.output / "declaration.json"),
                experience_hash=digest(rp),
                experience_audit_hash=a.audit_hash,
                warm_start_hash=r["training_hash"],
                parent_lineage_hash=warm["parent_lineage_hash"],
                data_hash=warm["data_hash"],
                initial_checkpoint_path=str(first),
                initial_checkpoint_hash=digest(first),
                checkpoint_path=str(last),
                checkpoint_hash=digest(last),
                accepted_optimizer_steps=accepted,
                initial_optimizer_steps=initial_steps,
                cumulative_optimizer_steps=initial_steps + accepted,
                epochs=epochs,
                episode_credit=sources,
                changed_parameter_tensors=changed,
                changed_critic_tensors=changed_critic,
                cpu_replay=replay,
                cpu_offline_decisions=queries,
                physical_steps_executed=0,
                actual_source_native_steps=r["actual_native_steps"],
                new_rollout_generations=0,
                online_RL_claim=True,
                old_retention_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                accepted_optimizer_steps=accepted,
                epochs=epochs,
                cpu_offline_decisions=queries,
                automatic_retry=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
