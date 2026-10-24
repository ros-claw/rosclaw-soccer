"""Read-only own-input/gradient diagnosis of a completed PPO generation.

Recurrent state is independently initialized, never restored from a trace.
Saved h/c are comparisons only. No optimizer or native physics step executes.
Recorded inputs do not become a new-policy physical trajectory by replaying NN.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("experience-root", "own-root", "training-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("own-hash", "training-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--state-id", type=int, choices=(25, 26), required=True)
    a = p.parse_args()
    import numpy as np
    import torch

    from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp
    from rosclaw_soccer.rsi.refinement_actor_critic import refinement_own_history_loss
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import (
        file_hash,
        load_onpolicy_refinement,
    )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_refinement_onpolicy_data import admit, episodes
    from scripts.rsi_stroke_search_common import open_motor

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only diagnostic output required")
    source, _, pins, scope, bank, _, _ = admit(
        a.experience_root, a.own_root, a.own_hash, a.state_id
    )
    tp = a.training_root / "complete.json"
    if file_hash(tp) != a.training_hash:
        raise ValueError("exact completed training required")
    training = json.loads(tp.read_text())
    if training["config"]["state_id"] != a.state_id:
        raise ValueError("same qualified model/state required")
    for k, v in training["input_file_hashes"].items():
        if k in pins and pins[k] != v:
            raise ValueError("conflicting diagnostic ancestry")
        pins[k] = v
    for path in (tp, a.training_root / "declaration.json", Path(training["checkpoint_path"])):
        pins[str(path)] = file_hash(path)
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            training_hash=a.training_hash,
            input_file_hashes=pins,
            scope="RECORDED_OWN_INPUT_NN_AND_GRADIENT_DIAGNOSIS_NOT_NEW_POLICY_PHYSICS",
            physical_steps_executed=0,
            optimizer_updates=0,
            goal_complete=False,
            hardware_authorized=False,
        ),
    )
    try:
        contexts = (
            {int(k): v for k, v in bank["contexts"].items()}
            if isinstance(bank["contexts"], dict)
            else dict(enumerate(bank["contexts"]))
        )
        _, old = open_motor(
            pins,
            scope["student_training_hash"],
            {i: contexts[i] for i in (1, 4, 20, 21)},
            checkpoint_kind="consolidated-refinement-v1",
        )
        learned, _ = load_onpolicy_refinement(
            old,
            behavior_training_hash=scope["student_training_hash"],
            root=a.training_root,
            expected_receipt_hash=a.training_hash,
        )
        scale = old.parent.memory.parent.get_buffer("action_scale").clone()
        batch = episodes(a.experience_root, source, scale)
        audit = []
        for r in source["records"]:
            with np.load(a.experience_root / r["artifacts"]["A"]["path"], allow_pickle=False) as z:
                obs = torch.from_numpy(z["NN_obs"][:, None].copy())
                with torch.no_grad():
                    om, os = old(obs, old.initial_state())
                    nm, ns, ev = learned.forward_with_evidence(obs, learned.initial_state())
                    _, ps, pe = learned.parent.forward_with_evidence(
                        obs, learned.parent.initial_state()
                    )
                    _, delta = learned.adapt_features(
                        pe["critic_features"], pe["final_mean_delta_radians"]
                    )
                    independent_mean = pe["reference_nominal"] + delta / scale

                def same(x, y, label):
                    if x.shape != y.shape or not torch.allclose(x, y, atol=2e-5, rtol=1e-6):
                        raise ValueError("own-input audit differs: " + label)

                same(om[:, 0], torch.from_numpy(z["NN_action"].copy()), "old source means")
                same(nm, independent_mean, "independent frozen-parent/head formula")
                same(
                    ev["reference_nominal"][:, 0],
                    torch.from_numpy(z["NN_reference_nominal"].copy()),
                    "reference",
                )
                for j, field in enumerate(("NN_h_after", "NN_c_after")):
                    expected = torch.from_numpy(z[field][-1].copy())
                    for state in (os, ns, ps):
                        same(state[j], expected, "final independently regenerated memory")
                changed = (nm[:, 0] - om[:, 0]) * scale
                if (ev["final_mean_delta_radians"].abs() > 0.200001).any():
                    raise ValueError("original action envelope changed")
                audit.append(
                    dict(
                        index=r["index"],
                        recorded_controls=len(obs),
                        mean_change_radians_rms=float(changed.square().mean().sqrt()),
                        mean_change_radians_max=float(changed.abs().max()),
                        own_memory_and_formula_match=True,
                    )
                )
        # Compute independent actor gradients of the two actual objective terms
        # at the OLD rollout policy. Critic features are detached in the trainer.
        old.head.requires_grad_(True)
        parameters = tuple(old.head.parameters())
        policy_gradient = [torch.zeros_like(x) for x in parameters]
        anchor_gradient = [torch.zeros_like(x) for x in parameters]
        log_std = torch.full((29,), float(np.log(0.02)), dtype=torch.float32)
        total = sum(len(ep["observations"]) for ep in batch)
        from rosclaw_soccer.rsi.dual_recurrent_actor_critic import MotorCritic

        critic = MotorCritic()
        metrics = []
        for ep in batch:
            # Original implementation performs all authority/density checks first.
            _, checked = refinement_own_history_loss(old, critic, ep, log_std, rho=0.9)
            mean, _, ev = old.forward_with_evidence(
                ep["observations"][:, None], old.initial_state()
            )
            mean = mean[:, 0]
            ratio = (
                history_motor_logp(
                    "box-truncated-ar1",
                    mean,
                    ep["actions"],
                    ep["reference_nominal"],
                    ep["action_scale"],
                    log_std,
                    rho=0.9,
                )
                - ep["old_logp"]
            ).exp()
            advantage = ep["advantages"]
            policy = -torch.minimum(ratio * advantage, ratio.clamp(0.9, 1.1) * advantage).mean()
            anchor = (
                0.02
                * 0.5
                * ((mean - ev["refinement_parent_mean"][:, 0].detach()) / 0.02)
                .square()
                .sum(-1)
                .mean()
            )
            weight = len(mean) / total
            pg = torch.autograd.grad(policy, parameters, retain_graph=True)
            ag = torch.autograd.grad(anchor, parameters)
            for acc, grads in ((policy_gradient, pg), (anchor_gradient, ag)):
                for x, g in zip(acc, grads, strict=True):
                    x.add_(g.detach(), alpha=weight)
            metrics.append(
                dict(
                    index=ep["index"],
                    initial_group=ep["index"] // 8,
                    reward=ep["terminal_reward"],
                    mean_advantage=float(advantage.mean()),
                    policy_loss=float(checked["policy_loss"]),
                    weighted_anchor_loss=float(anchor.detach()),
                )
            )
        pg, ag = [torch.cat([x.flatten() for x in g]) for g in (policy_gradient, anchor_gradient)]
        step = torch.cat(
            [
                (v.detach() - old.head.state_dict()[k]).flatten()
                for k, v in learned.head.state_dict().items()
            ]
        )
        gradients = dict(
            policy_norm=float(pg.norm()),
            anchor_norm=float(ag.norm()),
            cosine_policy_anchor=float(torch.nn.functional.cosine_similarity(pg[None], ag[None])),
            first_order_policy_loss_change=float(pg @ step),
            first_order_anchor_loss_change=float(ag @ step),
            learned_parameter_step_norm=float(step.norm()),
        )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="PPO_RECORDED_OWN_INPUT_AND_OBJECTIVE_GRADIENTS_DIAGNOSED",
                source_head=a.expected_head,
                training_hash=a.training_hash,
                input_file_hashes=pins,
                audit=audit,
                gradients=gradients,
                episode_metrics=metrics,
                recorded_input_streams=24,
                physical_steps_executed=0,
                optimizer_updates=0,
                new_policy_physical_success_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
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
                goal_complete=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
