"""Four bounded PPO epochs on ONE fully own-audited stochastic24 batch.

Actually updates plastic-head+critic weights. No new rollouts here, no teacher
imitation, no old-skill/physical/4v4 success claim and no runtime deployment.
"""

import argparse
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("experience-root", "own-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    a = p.parse_args()
    from rosclaw_soccer.rsi.refinement_onpolicy_learning import (
        COMMITMENTS,
        KIND,
        SCHEMA,
        STATUS,
        learning_config,
        tensor_state_hash,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins
    from scripts.rsi_refinement_onpolicy_data import admit, episodes

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only PPO output required")
    source, own, pins, scope, bank, _, _ = admit(
        a.experience_root, a.own_root, a.own_hash, a.state_id
    )
    _, behavior = receipt(pins, scope["student_training_hash"])
    cfg = learning_config(a.state_id)
    commitments = dict(
        behavior_training_hash=scope["student_training_hash"],
        behavior_checkpoint_hash=behavior["checkpoint_hash"],
        parent_lineage_hash=behavior["parent_lineage_hash"],
        experience_hash=digest(a.experience_root / "complete.json"),
        experience_own_hash=a.own_hash,
    )
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    accepted, history = 0, []
    try:
        import math

        import torch

        from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
        from rosclaw_soccer.rsi.paid_recurrent_ppo import backtracked_adam_update
        from rosclaw_soccer.rsi.refinement_actor_critic import (
            refinement_conditional_kl,
            refinement_own_history_loss,
        )
        from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import validate_checkpoint
        from scripts.rsi_stroke_search_common import open_motor

        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, actor = open_motor(
            pins,
            scope["student_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-refinement-v1",
        )
        torch.use_deterministic_algorithms(True)
        torch.manual_seed(cfg["seed"])
        critic = MotorCritic()
        initial_head = {k: v.detach().cpu().clone() for k, v in actor.head.state_dict().items()}
        initial_critic = {k: v.detach().cpu().clone() for k, v in critic.state_dict().items()}
        commitments["frozen_parent_hash"] = tensor_state_hash(dict(actor.parent.state_dict()))
        if set(commitments) != COMMITMENTS:
            raise ValueError("all immutable initial/data identities required")
        declared = dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            config=cfg,
            **commitments,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        )
        write_once(a.output / "declaration.json", declared)
        scale = actor.parent.memory.parent.get_buffer("action_scale").clone()
        batch = episodes(a.experience_root, source, scale)
        std = torch.full((29,), math.log(cfg["sigma_policy_units"]), dtype=torch.float32)
        # Every initial density/reference/private-memory check precedes any update.
        for ep in batch:
            with torch.no_grad():
                mean, _ = actor(ep["observations"][:, None], actor.initial_state())
                if not torch.allclose(mean[:, 0], ep["old_means"], atol=2e-5, rtol=1e-6):
                    raise ValueError(
                        "behavior whole-own replay differs before optimizer allocation"
                    )
                refinement_own_history_loss(actor, critic, ep, std, rho=0.9)
        actor.head.requires_grad_(True)
        actor.train()
        head_parameters = list(actor.head.parameters())
        value_parameters = list(critic.parameters())
        optimizer = torch.optim.Adam(
            [
                {"params": head_parameters, "lr": cfg["actor_lr"]},
                {"params": value_parameters, "lr": cfg["critic_lr"]},
            ]
        )
        if optimizer.state_dict()["state"]:
            raise ValueError("new generation0 optimizer must be empty")
        total_rows = sum(len(e["observations"]) for e in batch)

        def kl():
            return (
                sum(
                    len(e["observations"]) * refinement_conditional_kl(actor, e, std, rho=0.9)
                    for e in batch
                )
                / total_rows
            )

        for epoch in range(1, 5):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for ep in batch:
                loss, _ = refinement_own_history_loss(actor, critic, ep, std, rho=0.9)
                weight = len(ep["observations"]) / total_rows
                (loss * weight).backward()
                loss_value += float(loss.detach()) * weight
            actor_norm = float(torch.nn.utils.clip_grad_norm_(head_parameters, 1.0))
            critic_norm = float(torch.nn.utils.clip_grad_norm_(value_parameters, 1.0))
            if not all(math.isfinite(v) for v in (loss_value, actor_norm, critic_norm)):
                raise ValueError("finite actual actor/critic gradients required")
            update = backtracked_adam_update(
                optimizer,
                [*head_parameters, *value_parameters],
                kl,
                maximum_kl=0.015,
                maximum_attempts=8,
            )
            accepted += int(update["applied"])
            if (
                tensor_state_hash(dict(actor.parent.state_dict()))
                != commitments["frozen_parent_hash"]
            ):
                raise ValueError("frozen whole parent changed during actual PPO update")
            row = dict(
                epoch=epoch,
                loss=loss_value,
                actor_gradient_norm=actor_norm,
                critic_gradient_norm=critic_norm,
                update=update,
                accepted_steps=accepted,
            )
            history.append(row)
            write_once(a.output / f"epoch{epoch}.json", row)
            print(row, flush=True)
        if not accepted:
            raise ValueError("all actor steps rejected; cannot claim a trained policy")
        state = dict(
            schema=SCHEMA,
            kind=KIND,
            config=cfg,
            commitments=commitments,
            initial_head=initial_head,
            head={k: v.detach().cpu().clone() for k, v in actor.head.state_dict().items()},
            initial_critic=initial_critic,
            critic={k: v.detach().cpu().clone() for k, v in critic.state_dict().items()},
            optimizer=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),
            actor_steps=accepted,
            critic_steps=accepted,
        )
        # The behavior is reconstructed from exact immutable initial head+parent.
        import copy

        behavior_model = copy.deepcopy(actor).eval().requires_grad_(False)
        behavior_model.head.load_state_dict(initial_head, strict=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(cfg["seed"])
            initial_value = MotorCritic()
        validate_checkpoint(
            state,
            behavior_model,
            initial_value,
            config=cfg,
            commitments=commitments,
            steps=accepted,
        )
        cp = a.output / "actor_critic.pt"
        torch.save(state, cp)
        decoded = torch.load(cp, map_location="cpu", weights_only=True)
        validate_checkpoint(
            decoded,
            behavior_model,
            initial_value,
            config=cfg,
            commitments=commitments,
            steps=accepted,
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=STATUS,
                checkpoint_kind=KIND,
                **declared,
                declaration_hash=digest(a.output / "declaration.json"),
                checkpoint_path=str(cp),
                checkpoint_hash=digest(cp),
                history=history,
                new_actor_steps=accepted,
                new_critic_steps=accepted,
                actor_rows_per_epoch=total_rows,
                experience_worlds=24,
                terminal_rewards=[e["terminal_reward"] for e in batch],
                parent_weights_unchanged=True,
                on_policy_RL_update=True,
                physical_steps_executed=0,
                physical_retention_proven=False,
                continuous_learning_proven=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                accepted_actor_steps=accepted,
                accepted_critic_steps=accepted,
                history=history,
                physical_steps_executed=0,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
