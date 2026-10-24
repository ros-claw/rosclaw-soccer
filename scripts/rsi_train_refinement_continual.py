"""Generation1 PPO on NEW own-policy trajectories with trained critic/Adam carry.

No native physics here. Final learned weights remain unpromoted until separate
whole native exams, retention and disjoint Fresh. This is not end-to-end torque RL.
"""

import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("prepared-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("prepared-producer-head", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    import math

    import torch

    from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
    from rosclaw_soccer.rsi.paid_recurrent_ppo import backtracked_adam_update
    from rosclaw_soccer.rsi.refinement_actor_critic import (
        refinement_conditional_kl,
        refinement_own_history_loss,
    )
    from rosclaw_soccer.rsi.refinement_continual_batch import continual_config
    from rosclaw_soccer.rsi.refinement_continual_checkpoint import (
        KIND,
        SCHEMA,
        STATUS,
        rebased_adam,
        same_nested,
        validate_continual_checkpoint,
    )
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import (
        file_hash,
        load_onpolicy_refinement,
    )
    from rosclaw_soccer.rsi.refinement_onpolicy_learning import tensor_state_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins
    from scripts.rsi_stroke_search_common import open_motor

    clean(a.expected_head)
    root = a.prepared_root
    if a.output.exists() or root.is_symlink() or (root / "failed.json").exists():
        raise ValueError("unused output and complete ordinary predeclared preparation required")
    rp, dp, bp = root / "complete.json", root / "declaration.json", root / "prepared_batch.pt"
    ph, dh, bh = file_hash(rp), file_hash(dp), file_hash(bp, maximum=32 * 1024 * 1024)
    r, d = [json.loads(x.read_text()) for x in (rp, dp)]
    cfg = continual_config(r["config"]["state_id"])
    if (
        r["status"] != "CONTINUAL_NEW_LEARNED_ARM24_AND_TRAINED_CRITIC_PREPARED_NOT_UPDATED"
        or r["source_head"] != a.prepared_producer_head
        or d["source_head"] != a.prepared_producer_head
        or r["config"] != cfg
        or d["config"] != cfg
        or r["declaration_hash"] != dh
        or r["prepared_batch_hash"] != bh
        or r["input_file_hashes"] != d["input_file_hashes"]
        or r["prior_training_hash"] != d["prior_training_hash"]
        or r["selected_indices"] != list(range(1, 48, 2))
        or r["old_arm_rows_used"] != 0
        or r["actual_new_rollout_worlds"] != 24
        or r["new_optimizer_steps"] != 0
        or r["prior_adam_steps"] != 4
        or r["optimizer_steps"] != 4
        or any(
            r[k] is not False
            for k in (
                "goal_complete",
                "promotion_authorized",
                "hardware_authorized",
                "continuous_learning_proven",
            )
        )
    ):
        raise ValueError("exact new learned-arm batch and declared continuation state required")
    pins = dict(r["input_file_hashes"])
    pins.update({str(rp): ph, str(dp): dh, str(bp): bh})
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    accepted, history = 0, []
    try:
        prior_path, prior = receipt(pins, r["prior_training_hash"])
        _, bc = receipt(pins, prior["behavior_training_hash"])
        _, bank = receipt(pins, bc["bank_hash"])
        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, base = open_motor(
            pins,
            prior["behavior_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-refinement-v1",
        )
        behavior, _ = load_onpolicy_refinement(
            base,
            behavior_training_hash=prior["behavior_training_hash"],
            root=prior_path.parent,
            expected_receipt_hash=r["prior_training_hash"],
        )
        parent_cp = torch.load(
            Path(prior["checkpoint_path"]), map_location="cpu", weights_only=True
        )
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(prior["config"]["seed"])
            prior_critic = MotorCritic()
        prior_critic.load_state_dict(parent_cp["critic"], strict=True)
        prior_critic.eval().requires_grad_(False)
        prepared = torch.load(bp, map_location="cpu", weights_only=True)
        initial_optimizer = rebased_adam(parent_cp["optimizer"], cfg)
        if (
            prepared["config"] != cfg
            or not same_nested(prepared["initial_head"], dict(behavior.head.state_dict()))
            or not same_nested(prepared["initial_critic"], dict(prior_critic.state_dict()))
            or not same_nested(prepared["prior_optimizer"], parent_cp["optimizer"])
            or not same_nested(prepared["rebased_optimizer"], initial_optimizer)
        ):
            raise ValueError("actual prior head/critic/Adam state must carry into generation1")
        batch = prepared["episodes"]
        if len(batch) != 24 or [e["index"] for e in batch] != r["selected_indices"]:
            raise ValueError("all24 independently audited new-policy trajectories required")
        std = torch.full((29,), math.log(0.02), dtype=torch.float32)
        with torch.no_grad():
            for e in batch:
                mean, _, ev = behavior.forward_with_evidence(
                    e["observations"][:, None], behavior.initial_state()
                )
                value = prior_critic(ev["critic_features"][:, 0])
                if not torch.allclose(
                    mean[:, 0], e["old_means"], atol=2e-5, rtol=1e-6
                ) or not torch.allclose(value, e["old_values"], atol=1e-6, rtol=1e-6):
                    raise ValueError(
                        "new means and trained critic values must reproduce independently"
                    )
                refinement_own_history_loss(behavior, prior_critic, e, std, rho=0.9)
        actor = copy.deepcopy(behavior).train()
        actor.head.requires_grad_(True)
        critic = copy.deepcopy(prior_critic).train().requires_grad_(True)
        torch.use_deterministic_algorithms(True)
        torch.set_rng_state(parent_cp["rng_state"])
        hp, vp = list(actor.head.parameters()), list(critic.parameters())
        optimizer = torch.optim.Adam(
            [dict(params=hp, lr=cfg["actor_lr"]), dict(params=vp, lr=cfg["critic_lr"])]
        )
        optimizer.load_state_dict(copy.deepcopy(initial_optimizer))
        commitments = dict(
            prior_training_hash=r["prior_training_hash"],
            prepared_hash=ph,
            frozen_parent_hash=tensor_state_hash(dict(behavior.parent.state_dict())),
        )
        declared = dict(
            config=cfg,
            source_head=a.expected_head,
            input_file_hashes=pins,
            **commitments,
            goal_complete=False,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        write_once(a.output / "declaration.json", declared)
        rows = sum(len(e["observations"]) for e in batch)

        def kl():
            return (
                sum(
                    len(e["observations"]) * refinement_conditional_kl(actor, e, std, rho=0.9)
                    for e in batch
                )
                / rows
            )

        for epoch in range(1, 5):
            optimizer.zero_grad(set_to_none=True)
            loss_value = 0.0
            for e in batch:
                loss, _ = refinement_own_history_loss(actor, critic, e, std, rho=0.9)
                w = len(e["observations"]) / rows
                (loss * w).backward()
                loss_value += float(loss.detach()) * w
            an, vn = [float(torch.nn.utils.clip_grad_norm_(x, 1.0)) for x in (hp, vp)]
            if not all(math.isfinite(x) for x in (loss_value, an, vn)):
                raise ValueError("finite continued actor/critic gradients required")
            update = backtracked_adam_update(
                optimizer, [*hp, *vp], kl, maximum_kl=0.015, maximum_attempts=8
            )
            accepted += int(update["applied"])
            if (
                tensor_state_hash(dict(actor.parent.state_dict()))
                != commitments["frozen_parent_hash"]
            ):
                raise ValueError("whole frozen parent changed")
            row = dict(
                epoch=epoch,
                loss=loss_value,
                actor_gradient_norm=an,
                critic_gradient_norm=vn,
                update=update,
                accepted_steps=accepted,
                cumulative_steps=4 + accepted,
            )
            history.append(row)
            write_once(a.output / f"epoch{epoch}.json", row)
            print(row, flush=True)
        if not accepted:
            raise ValueError("all generation1 steps rejected; no trained successor")
        state = dict(
            schema=SCHEMA,
            kind=KIND,
            config=cfg,
            commitments=commitments,
            initial_head=dict(behavior.head.state_dict()),
            initial_critic=dict(prior_critic.state_dict()),
            initial_optimizer=initial_optimizer,
            head=dict(actor.head.state_dict()),
            critic=dict(critic.state_dict()),
            optimizer=optimizer.state_dict(),
            rng_state=torch.get_rng_state(),
            new_actor_steps=accepted,
            new_critic_steps=accepted,
            actor_steps=4 + accepted,
            critic_steps=4 + accepted,
        )
        validate_continual_checkpoint(
            state,
            behavior,
            prior_critic,
            parent_cp["optimizer"],
            config=cfg,
            commitments=commitments,
            steps=accepted,
        )
        cp = a.output / "actor_critic.pt"
        torch.save(state, cp)
        validate_continual_checkpoint(
            torch.load(cp, map_location="cpu", weights_only=True),
            behavior,
            prior_critic,
            parent_cp["optimizer"],
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
                checkpoint_path=str(cp),
                checkpoint_hash=file_hash(cp),
                declaration_hash=file_hash(a.output / "declaration.json"),
                history=history,
                new_actor_steps=accepted,
                new_critic_steps=accepted,
                cumulative_actor_steps=4 + accepted,
                cumulative_critic_steps=4 + accepted,
                actor_rows_per_epoch=rows,
                new_policy_rollout_worlds=24,
                old_arm_rows_used=0,
                trained_prior_critic_used=True,
                prior_adam_moments_carried=True,
                learning_rate_rebased=True,
                physical_steps_executed=0,
                two_generation_parameter_continuity_verified=True,
                continuous_learning_proven=False,
                physical_retention_proven=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                accepted_actor_steps=accepted,
                accepted_critic_steps=accepted,
                cumulative_actor_steps=4 + accepted,
                history=history,
                physical_steps_executed=0,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
