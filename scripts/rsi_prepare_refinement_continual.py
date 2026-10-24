"""Prepare genuinely new on-policy actor/critic data from a learned-policy arm.

This executes NN queries only, not native physics or optimizer updates.
Critic values come from the prior TRAINED critic, not a new zero critic.
Both the unused old control arm and whole own-law audit remain preserved.
"""

import argparse
import copy
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in (
        "experience-root",
        "own-root",
        "training-root",
        "new-source-root",
        "new-own-root",
        "output",
    ):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "training-hash", "new-own-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    a = p.parse_args()
    import math

    import torch

    from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic
    from rosclaw_soccer.rsi.refinement_actor_critic import refinement_own_history_loss
    from rosclaw_soccer.rsi.refinement_continual_batch import continual_config, select_learned_arm
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from rosclaw_soccer.rsi.refinement_onpolicy_learning import KIND, tensor_state_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_learned_refinement_common import admit, open_models
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_refinement_onpolicy_data import episodes

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only continual batch output required")
    admitted = admit(a)
    pins = admitted[1]
    prior = json.loads((a.training_root / "complete.json").read_text())
    if (
        prior["checkpoint_kind"] != KIND
        or prior["new_actor_steps"] != 4
        or prior["new_critic_steps"] != 4
    ):
        raise ValueError(
            "exact trained generation0 actual new-rollout behavior required, not peer branch"
        )
    reports = []
    for root in (a.new_source_root, a.new_own_root):
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("ordinary complete new on-policy experience required")
        rp, dp = root / "complete.json", root / "declaration.json"
        rh, dh = file_hash(rp), file_hash(dp)
        r, d = [json.loads(x.read_text()) for x in (rp, dp)]
        if (
            r["declaration_hash"] != dh
            or r["input_file_hashes"] != d["input_file_hashes"]
            or any(r[k] != d[k] for k in ("source_head", "config", "training_hash"))
        ):
            raise ValueError("exact prior-policy collection declaration required")
        for k, v in r["input_file_hashes"].items():
            if k in pins and pins[k] != v:
                raise ValueError("conflicting new experience ancestry")
            pins[k] = v
        pins.update({str(rp): rh, str(dp): dh})
        reports.append(r)
    if file_hash(a.new_own_root / "complete.json") != a.new_own_hash:
        raise ValueError("exact independently audited new-policy batch required")
    selected = select_learned_arm(
        *reports,
        source_hash=file_hash(a.new_source_root / "complete.json"),
        training_hash=a.training_hash,
    )
    if reports[0]["config"]["state_id"] != a.state_id:
        raise ValueError("same exact behavior/state required")
    for r in reports[0]["records"]:
        if set(r["artifacts"]) != {"initial", "physical", "A", "B", "sampling"}:
            raise ValueError("all paired arm source primitives required")
        for art in r["artifacts"].values():
            path = a.new_source_root / art["path"]
            if (
                path.resolve().parent != (a.new_source_root / f"trial{r['index']}").resolve()
                or file_hash(path) != art["hash"]
            ):
                raise ValueError("exact confined actual new on-policy primitives required")
            pins[str(path)] = art["hash"]
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    cfg = continual_config(a.state_id)
    write_once(
        a.output / "declaration.json",
        dict(
            config=cfg,
            source_head=a.expected_head,
            input_file_hashes=pins,
            prior_training_hash=a.training_hash,
            new_source_hash=file_hash(a.new_source_root / "complete.json"),
            new_own_hash=a.new_own_hash,
            selected_indices=[r["index"] for r in selected],
            physical_steps_executed=0,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    try:
        setup, _, actor, courses, preflight = open_models(a, admitted)
        if courses != [reports[0]["records"][i]["course"] for i in (0, 16, 32)]:
            raise ValueError(
                "all new-policy physical initial courses must bind inherited preflight"
            )
        cp = torch.load(Path(prior["checkpoint_path"]), map_location="cpu", weights_only=True)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(prior["config"]["seed"])
            critic = MotorCritic()
        critic.load_state_dict(cp["critic"], strict=True)
        critic.eval().requires_grad_(False)
        scale = actor.parent.memory.parent.get_buffer("action_scale").clone()
        batch = episodes(a.new_source_root, dict(records=selected), scale)
        log_std = torch.full((29,), math.log(0.02), dtype=torch.float32)
        with torch.no_grad():
            for ep in batch:
                means, _, ev = actor.forward_with_evidence(
                    ep["observations"][:, None], actor.initial_state()
                )
                if not torch.allclose(means[:, 0], ep["old_means"], atol=2e-5, rtol=1e-6):
                    raise ValueError(
                        "prior learned behavior does not reproduce actual new rollout means"
                    )
                ep["old_values"] = critic(ev["critic_features"][:, 0]).clone()
                ep["advantages"] = ep["returns"] - ep["old_values"]
        flat = torch.cat([ep["advantages"] for ep in batch])
        for ep in batch:
            ep["advantages"] = (ep["advantages"] - flat.mean()) / flat.std(
                unbiased=False
            ).clamp_min(1e-6)
            with torch.no_grad():
                refinement_own_history_loss(actor, critic, ep, log_std, rho=0.9)
        actor_before = tensor_state_hash(dict(actor.state_dict()))
        critic_before = tensor_state_hash(dict(critic.state_dict()))
        # Allocate an optimizer only to prove the carry path; NEVER .step().
        actor.head.requires_grad_(True)
        critic.requires_grad_(True)
        optimizer = torch.optim.Adam(
            [
                dict(params=list(actor.head.parameters()), lr=cfg["actor_lr"]),
                dict(params=list(critic.parameters()), lr=cfg["critic_lr"]),
            ]
        )
        optimizer.load_state_dict(copy.deepcopy(cp["optimizer"]))
        if any(float(v["step"]) != 4 for v in optimizer.state_dict()["state"].values()):
            raise ValueError("actual generation0 four-step Adam counters required")
        old_rates = [g["lr"] for g in optimizer.param_groups]
        for group, rate in zip(
            optimizer.param_groups, (cfg["actor_lr"], cfg["critic_lr"]), strict=True
        ):
            group["lr"] = rate
        if actor_before != tensor_state_hash(
            dict(actor.state_dict())
        ) or critic_before != tensor_state_hash(dict(critic.state_dict())):
            raise ValueError("preparation cannot alter learned weights")
        bp = a.output / "prepared_batch.pt"
        torch.save(
            dict(
                config=cfg,
                episodes=batch,
                initial_head=actor.head.state_dict(),
                initial_critic=critic.state_dict(),
                prior_optimizer=cp["optimizer"],
                rebased_optimizer=optimizer.state_dict(),
            ),
            bp,
        )
        # Fail closed on any unsafe/non-tensor serialization addition.
        torch.load(bp, map_location="cpu", weights_only=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="CONTINUAL_NEW_LEARNED_ARM24_AND_TRAINED_CRITIC_PREPARED_NOT_UPDATED",
                source_head=a.expected_head,
                config=cfg,
                input_file_hashes=pins,
                prior_training_hash=a.training_hash,
                new_source_hash=file_hash(a.new_source_root / "complete.json"),
                new_own_hash=a.new_own_hash,
                selected_indices=[r["index"] for r in selected],
                old_arm_rows_used=0,
                actual_new_rollout_worlds=24,
                actor_rows=sum(len(ep["observations"]) for ep in batch),
                old_values_nonzero_rows=sum(int(ep["old_values"].ne(0).sum()) for ep in batch),
                prior_actor_weights_hash=actor_before,
                prior_critic_weights_hash=critic_before,
                prior_adam_steps=4,
                optimizer_steps=4,
                new_optimizer_steps=0,
                prior_learning_rates=old_rates,
                rebased_learning_rates=[g["lr"] for g in optimizer.param_groups],
                prepared_batch_hash=file_hash(bp),
                declaration_hash=file_hash(a.output / "declaration.json"),
                physical_steps_executed=0,
                optimizer_updates=0,
                continuous_learning_proven=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                physical_steps_executed=0,
                optimizer_updates=0,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
