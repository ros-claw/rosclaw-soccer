"""Paired critic normalization ablation; actor stays byte-identical to BC."""

import argparse
import io
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("bank", "training", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("bank-hash", "training-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--normalize", action="store_true")
    a = p.parse_args()
    if a.seed not in (926041, 926042, 926043):
        p.error("three predeclared paired fitting seeds required")
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins
    from scripts.rsi_proposal_feedback_data import admit_feedback, critic_arrays

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused calibration output required")
    pins = {}
    old, checkpoint_path, policy_path, files = admit_feedback(
        a.bank, a.bank_hash, a.training, a.training_hash, pins
    )
    core = pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import json

    import numpy as np
    import torch
    from rosclaw.feedback.contracts import canonical_hash
    from rosclaw.simforge.g1_neural_torque import G1TorqueSafetyConfig
    from rosclaw.simforge.g1_neural_torque_learning import (
        G1ContinualTorqueActorCritic,
        G1NeuralTorqueLearnerConfig,
    )

    torch.set_num_threads(1)
    values = dict(old["config"])
    values["gamma"] = 0.9995
    config = G1NeuralTorqueLearnerConfig(**values)
    learner = G1ContinualTorqueActorCritic(config, safety=G1TorqueSafetyConfig(**old["safety"]))
    payload = torch.load(
        io.BytesIO(checkpoint_path.read_bytes()), map_location="cpu", weights_only=True
    )
    if payload["config_hash"] != canonical_hash(old["config"]):
        raise ValueError("exact original BC checkpoint config required")
    payload["config_hash"] = canonical_hash(asdict(config))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    learner.restore_checkpoint(buffer.getvalue())

    def actor_binary():
        return learner.artifact_bytes(
            body_hash=old["body_hash"],
            parent_policy_hash=old["parent_policy_hash"],
            dataset_hash=old["dataset_hash"],
        )

    baseline = policy_path.read_bytes()
    if actor_binary() != baseline:
        raise ValueError("restoration changed the BC actor")
    groups = critic_arrays(files)
    x, actions, targets = groups["train"]
    critics = [learner.reward_critic, learner.fall_critic, learner.constraint_critic]

    def prediction(split, indices):
        obs, act, _ = groups[split]
        states = learner._observation_tensor(obs[indices])
        ratio = learner._tensor(act[indices]) / learner.actor.action_limits
        return torch.stack(
            [torch.stack(critic(states, ratio), dim=-1) for critic in critics], dim=1
        ).squeeze(2)

    with torch.no_grad():
        before = prediction("validation", np.arange(min(256, len(groups["validation"][0])))).clone()
    statistics = (
        learner.configure_frozen_value_scales(targets)
        if a.normalize
        else {label: {"center": 0.0, "scale": 1.0} for label in ("reward", "fall", "constraint")}
    )
    scales = learner._tensor(
        np.asarray([statistics[label]["scale"] for label in ("reward", "fall", "constraint")])
    )
    with torch.no_grad():
        after = prediction("validation", np.arange(min(256, len(groups["validation"][0]))))
    preservation_error = float((before - after).abs().max())
    if preservation_error > 1e-3:
        raise ValueError("initial raw-unit prediction preservation exceeded floating tolerance")
    constants = targets.mean(axis=0)
    constant_mse = {
        split: np.mean((data[2] - constants) ** 2, axis=0).tolist()
        for split, data in groups.items()
    }

    def evaluate(split):
        total = np.zeros(3)
        count = 0
        with torch.no_grad():
            for start in range(0, len(groups[split][0]), 256):
                indices = np.arange(start, min(start + 256, len(groups[split][0])))
                residual = (
                    prediction(split, indices)
                    - learner._tensor(groups[split][2][indices])[:, :, None]
                )
                total += residual.square().sum(dim=(0, 2)).cpu().numpy()
                count += 2 * len(indices)
        return (total / count).tolist()

    a.output.mkdir(parents=True, exist_ok=False)
    declaration = dict(
        protocol="paired_frozen_proposal_critic_scale.v1",
        source_head=a.expected_head,
        core_root=str(core),
        core_commit=a.core_expected_head,
        input_file_hashes=pins,
        bank_hash=a.bank_hash,
        training_hash=a.training_hash,
        seed=a.seed,
        normalize=a.normalize,
        statistics=statistics,
        constant_training_mean_baseline_mse=constant_mse,
        initial_value_preservation_error=preservation_error,
        steps=128,
        actor_optimizer_updates=0,
        validation_used_for_statistics=False,
        observations_are_truncated16step_context=True,
        train_rows=len(x),
        validation_rows=len(groups["validation"][0]),
        training_geometry_indices=[0, 3],
        validation_geometry_indices=[1, 2],
        training_body_failures=sum(
            row["ordinary_body_failure_retained"] for row, *_ in files if row["index"] in (0, 3)
        ),
        validation_body_failures=sum(
            row["ordinary_body_failure_retained"] for row, *_ in files if row["index"] in (1, 2)
        ),
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration)
    steps = []
    hook = learner.critic_optimizer.register_step_post_hook(
        lambda *args, **kwargs: steps.append(len(steps) + 1)
    )
    metrics = []
    try:
        metrics.append(dict(step=0, training=evaluate("train"), validation=evaluate("validation")))
        rng = np.random.default_rng(a.seed)
        for step in range(128):
            indices = rng.choice(len(x), 256, replace=False)
            residual = (
                prediction("train", indices) - learner._tensor(targets[indices])[:, :, None]
            ) / scales[None, :, None]
            loss = residual.square().mean()
            if not bool(torch.isfinite(loss)):
                raise ValueError("nonfinite scaled critic objective")
            learner.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for critic in critics for parameter in critic.parameters()],
                10.0,
                error_if_nonfinite=True,
            )
            learner.critic_optimizer.step()
            if (step + 1) % 32 == 0:
                metrics.append(
                    dict(
                        step=step + 1, training=evaluate("train"), validation=evaluate("validation")
                    )
                )
                print(json.dumps(metrics[-1]), flush=True)
        if len(steps) != 128 or actor_binary() != baseline or learner.update_index != 0:
            raise ValueError("actual critic budget or frozen actor identity violated")
        for target, current in zip(
            (learner.reward_target, learner.fall_target, learner.constraint_target),
            critics,
            strict=True,
        ):
            target.load_state_dict(current.state_dict())
        path = a.output / "critic-checkpoint.pt"
        with path.open("xb") as stream:
            stream.write(learner.checkpoint_bytes())
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="PROPOSAL_CRITIC_CALIBRATION_EXPERIMENT_ACTOR_FROZEN",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration=declaration,
                metrics=metrics,
                actual_critic_Adam_steps=len(steps),
                actor_optimizer_updates=0,
                alpha_optimizer_updates=0,
                SAC_update_index=0,
                actor_binary_unchanged=True,
                checkpoint=dict(path=path.name, hash=digest(path)),
                policy_hash=old["policy"]["hash"],
                new_physics_steps=0,
                physical_gain_verified=False,
                calibration_promotion_authorized=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), actual_critic_steps=len(steps), actor_optimizer_updates=0),
        )
        raise
    finally:
        hook.remove()


if __name__ == "__main__":
    main()
