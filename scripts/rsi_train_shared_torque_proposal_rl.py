"""SIM-only proposal-Q pilot: MC critic warmup then two fresh SAC updates.

Reuse the real BC checkpoint, critics, Adam, retention anchors and RNG. No
physical or promotional success is inferred from optimizer progress. Never
train on bypassed-inference placeholders or treat BC anchors as Q transitions.
"""

import argparse
import io
import json
from dataclasses import asdict, replace
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "rollout", "own", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "training-hash",
        "rollout-hash",
        "own-hash",
        "core-expected-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused training output required")
    reports, pins = {}, {}
    for name in ("training", "rollout", "own"):
        path = getattr(a, name) / "complete.json"
        expected = getattr(a, name + "_hash")
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact immutable completed training/physical/replay lineage required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting source provenance")
            pins[key] = value
        pins[str(path)] = expected
        reports[name] = report
    old, rollout, own = (reports[name] for name in ("training", "rollout", "own"))
    declaration = json.loads((a.rollout / "declaration.json").read_text())
    if (
        old["SAC_update_index"] != 0
        or old["critic_optimizer_updates"] != 0
        or old["actual_BC_actor_optimizer_updates"] != 390
        or old["training_indices"] != [0, 3]
        or old["validation_indices"] != [1, 2]
        or not own["all_exact"]
        or own["source_hash"] != a.rollout_hash
        or len(own["records"]) != 4
        or len(rollout["records"]) != 4
        or declaration["protocol"] != "robonaldo.native_guarded_proposal_trace.v1"
        or declaration["torque_intervention_fraction"] != 1.0
    ):
        raise ValueError(
            "exact retained390-step BC and independently replayed full proposal experiment required"
        )

    def pin_artifact(root, artifact):
        path = root / artifact["path"]
        if (
            path.is_symlink()
            or path.parent.resolve() != root.resolve()
            or digest(path) != artifact["hash"]
        ):
            raise ValueError("confined exact immutable binary/array artifact required")
        pins[str(path)] = artifact["hash"]
        return path

    checkpoint_path = pin_artifact(a.training, old["checkpoint"])
    pin_artifact(a.training, old["policy"])
    files = []
    for i, row in enumerate(rollout["records"]):
        if (
            row["index"] != i
            or not row["healthy_full_horizon"]
            or row["native_steps"] != 6520
            or row["torque_intervention_fraction"] != 1.0
            or not own["records"][i]["all_exact"]
            or not all(own["records"][i]["guarded_proposal_exact"].values())
        ):
            raise ValueError(
                "all four complete actual native clocks and raw proposal identities required"
            )
        ep = row["torque_execution_artifacts"]["a/"]
        files.append(
            (
                pin_artifact(a.rollout, row["artifact"]),
                pin_artifact(a.rollout, ep),
                pin_artifact(a.rollout, ep["guarded_proposal_trace"]),
            )
        )
    core = pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    import torch
    from rosclaw.feedback.contracts import canonical_hash
    from rosclaw.simforge.g1_neural_torque import G1TeacherTorqueEpisode, G1TorqueSafetyConfig
    from rosclaw.simforge.g1_neural_torque_learning import (
        G1ContinualTorqueActorCritic,
        G1NeuralTorqueLearnerConfig,
        G1NeuralTorqueReplay,
        neural_torque_replay_hash,
        teacher_replay,
    )
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_kick_reward import finite_returns, native_kick_rewards
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    config_values = dict(old["config"])
    # Explicit timebase migration ONLY: same architecture/optimizer settings.
    # gamma=.995 at2ms has a .4s e-fold horizon, too short for the actual ball
    # flight. .9995 gives ~4s. Preserve every optimizer/RNG/retention tensor.
    config_values["gamma"] = 0.9995
    config = G1NeuralTorqueLearnerConfig(**config_values)
    learner = G1ContinualTorqueActorCritic(config, safety=G1TorqueSafetyConfig(**old["safety"]))
    payload = torch.load(
        io.BytesIO(checkpoint_path.read_bytes()), map_location="cpu", weights_only=True
    )
    if payload["config_hash"] != canonical_hash(old["config"]) or payload["update_index"] != 0:
        raise ValueError("exact original checkpoint config and no earlier SAC update required")
    payload["config_hash"] = canonical_hash(asdict(config))
    buffer = io.BytesIO()
    torch.save(payload, buffer)
    learner.restore_checkpoint(buffer.getvalue())
    baseline = learner.artifact_bytes(
        body_hash=old["body_hash"],
        parent_policy_hash=old["parent_policy_hash"],
        dataset_hash=old["dataset_hash"],
    )
    if baseline != (a.training / old["policy"]["path"]).read_bytes():
        raise ValueError("checkpoint restoration/config migration changed deployed BC actor")
    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("one exact physical source scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    if hash_json(layout.profile()) != old["body_hash"]:
        raise ValueError("replay must bind the exact native body")
    datasets, feedback = [], []
    hard = np.asarray(G1_HARD_TORQUE_LIMITS)
    for i, (raw_path, ep_path, trace_path) in enumerate(files):
        with np.load(raw_path) as raw, np.load(ep_path) as ep, np.load(trace_path) as trace:
            observations = ep["observations"].copy()
            valid = trace["actor_proposal_valid"]
            if (
                valid.dtype != np.bool_
                or valid.shape != (6520,)
                or not np.array_equal(trace["applied_actions"], raw["torque"][:, layout.ai])
                or np.any(trace["actor_proposals"][~valid])
                or not np.all(trace["used_parent"][~valid])
                or not np.all(trace["intervention_fractions"] == 1)
            ):
                raise ValueError(
                    "actual proposal mask, force identity and full intervention required"
                )
            signal = native_kick_rewards(
                ball_positions=np.vstack(
                    ([*declaration["starts"][i], 0.11], raw["qpos"][:, layout.bq : layout.bq + 3])
                ),
                pelvis_after=raw["qpos"][:, layout.rq : layout.rq + 7],
                joint_velocity_after=raw["qvel"][:, layout.vi],
                applied_torque=trace["applied_actions"],
                hard_limits=hard,
                foot_contact=np.max(raw["ball_forces"][:, 0, :2], axis=1) > 1,
                illegal_contact=raw["ball_forces"][:, 0, 2] > 1,
                used_parent=trace["used_parent"],
            )
            indices = np.unique(np.append(np.arange(15, 6519, 10), 6518))
            indices = indices[valid[indices] & valid[indices + 1]]
            if len(indices) < config.batch_size:
                raise ValueError("enough real consecutive inferred proposal transitions required")
            observations_batch = np.stack([observations[k - 15 : k + 1] for k in indices])
            next_batch = np.stack([observations[k - 14 : k + 2] for k in indices])
            replay = G1NeuralTorqueReplay(
                observations=observations_batch,
                next_observations=next_batch,
                actions=trace["actor_proposals"][indices].copy(),
                parent_actions=trace["parent_actions"][indices].copy(),
                rewards=signal["rewards"][indices, None],
                fall_costs=signal["fall_costs"][indices, None],
                constraint_costs=signal["constraint_costs"][indices, None],
                terminals=(indices == 6518).astype(np.float32)[:, None],
                partitions=np.zeros(len(indices), np.int8),
                policy_lags=np.zeros(len(indices), np.int64),
                action_semantics="ACTOR_PROPOSAL_THROUGH_GUARD",
            )
            targets = np.stack(
                [
                    finite_returns(signal[name], gamma=config.gamma)[indices]
                    for name in ("rewards", "fall_costs", "constraint_costs")
                ],
                axis=1,
            )
            datasets.append((replay, targets))
            feedback.append(
                dict(
                    index=i,
                    native_proposals=int(valid.sum()),
                    sampled_transitions=len(indices),
                    crossing_index=signal["crossing_index"],
                    crossing_error_m=signal["crossing_error_m"],
                    total_reward=float(signal["rewards"].sum()),
                    replay_hash=neural_torque_replay_hash(replay),
                )
            )
    train = G1NeuralTorqueReplay.combine(*(datasets[i][0] for i in (0, 3)))
    target_train = np.concatenate([datasets[i][1] for i in (0, 3)])
    validation = G1NeuralTorqueReplay.combine(*(datasets[i][0] for i in (1, 2)))
    target_val = np.concatenate([datasets[i][1] for i in (1, 2)])
    teacher_paths = [
        Path(path)
        for path, expected in pins.items()
        if path.endswith("/complete.json") and expected == old["teacher_hash"]
    ]
    if len(teacher_paths) != 1:
        raise ValueError("one exact retained historical teacher required")
    teacher = json.loads(teacher_paths[0].read_text())
    anchors = []
    for i in (0, 3):
        artifact = teacher["records"][i]["torque_teacher_artifacts"]["a/"]
        path = pin_artifact(teacher_paths[0].parent, artifact)
        with np.load(path) as data:
            anchors.append(
                G1TeacherTorqueEpisode(
                    data["observations"].copy(),
                    data["actions"].copy(),
                    data["parent_actions"].copy(),
                )
            )
    anchor = replace(
        teacher_replay(tuple(anchors), sequence_length=16),
        action_semantics="ACTOR_PROPOSAL_THROUGH_GUARD",
    )
    a.output.mkdir(parents=True, exist_ok=False)
    declaration_out = dict(
        protocol="shared_native_proposal_Q_pilot.v1",
        source_head=a.expected_head,
        core_source=str(core),
        core_commit=a.core_expected_head,
        input_file_hashes=pins,
        training_hash=a.training_hash,
        rollout_hash=a.rollout_hash,
        own_hash=a.own_hash,
        training_indices=[0, 3],
        validation_indices=[1, 2],
        native_dt_s=0.002,
        config=asdict(config),
        config_migration={"gamma": [old["config"]["gamma"], config.gamma]},
        MC_critic_warmup_steps=128,
        SAC_actor_steps=2,
        BC_anchor_labels_not_critic_transitions=True,
        sequence_burnin_limit="16 step truncated GRU context; not full runtime persistent hidden equivalence",
        feedback=feedback,
        hardware_authorized=False,
        promotion_authorized=False,
    )
    write_once(a.output / "declaration.json", declaration_out)

    def artifact(name, content):
        with (a.output / name).open("xb") as stream:
            stream.write(content)
        return dict(path=name, hash=digest(a.output / name))

    counts = dict(actor=0, critic=0, alpha=0)
    hooks = []
    for name in counts:

        def count_step(optimizer, args, kwargs, name=name):
            counts[name] += 1

        hooks.append(getattr(learner, name + "_optimizer").register_step_post_hook(count_step))
    critics = [learner.reward_critic, learner.fall_critic, learner.constraint_critic]

    def predictions(replay, indices):
        obs = learner._observation_tensor(replay.observations[indices])
        actions = learner._tensor(replay.actions[indices]) / learner.actor.action_limits
        return torch.stack(
            [torch.stack(critic(obs, actions), dim=-1) for critic in critics], dim=1
        ).squeeze(2)

    def evaluate(replay, targets):
        with torch.no_grad():
            losses = []
            for start in range(0, replay.count, 256):
                indices = np.arange(start, min(replay.count, start + 256))
                residual = (
                    predictions(replay, indices) - learner._tensor(targets[indices])[:, :, None]
                )
                losses.append((residual.square().sum(dim=(0, 2)).cpu().numpy(), len(indices) * 2))
            return (sum(x for x, n in losses) / sum(n for x, n in losses)).tolist()

    try:
        before = artifact("resumed-before-training.pt", learner.checkpoint_bytes())
        metrics = [
            dict(
                stage="before_MC",
                training=evaluate(train, target_train),
                validation=evaluate(validation, target_val),
            )
        ]
        rng = np.random.default_rng(926027)
        for step in range(128):
            indices = rng.choice(train.count, config.batch_size, replace=False)
            residual = (
                predictions(train, indices) - learner._tensor(target_train[indices])[:, :, None]
            )
            loss = residual.square().mean()
            if not torch.isfinite(loss):
                raise ValueError("nonfinite MC critic loss")
            learner.critic_optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [parameter for critic in critics for parameter in critic.parameters()], 10.0
            )
            learner.critic_optimizer.step()
            if (step + 1) % 32 == 0:
                metrics.append(
                    dict(
                        stage=f"MC_{step + 1}",
                        training=evaluate(train, target_train),
                        validation=evaluate(validation, target_val),
                    )
                )
                print(json.dumps(metrics[-1]), flush=True)
        if (
            learner.artifact_bytes(
                body_hash=old["body_hash"],
                parent_policy_hash=old["parent_policy_hash"],
                dataset_hash=old["dataset_hash"],
            )
            != baseline
        ):
            raise ValueError("critic warmup changed actor")
        for target, current in zip(
            (learner.reward_target, learner.fall_target, learner.constraint_target),
            critics,
            strict=True,
        ):
            target.load_state_dict(current.state_dict())
        updates = []
        for _ in range(2):
            recent = replace(
                train, policy_lags=np.full(train.count, learner.update_index, np.int64)
            )
            update = learner.update(G1NeuralTorqueReplay.combine(anchor, recent))
            if not update.finite:
                raise ValueError("nonfinite SAC update")
            updates.append(update.to_dict())
        if counts != dict(actor=2, critic=130, alpha=2) or learner.update_index != 2:
            raise ValueError("actual optimizer counters disagree with two fresh policy updates")
        check_pins(pins)
        checkpoint = artifact("trained-checkpoint.pt", learner.checkpoint_bytes())
        dataset_hash = hash_json(
            dict(
                feedback=feedback,
                training_hash=a.training_hash,
                rollout_hash=a.rollout_hash,
                native_reward="native_kick_reward.v1",
            )
        )
        policy = artifact(
            "trained-torque-policy.bin",
            learner.artifact_bytes(
                body_hash=old["body_hash"],
                parent_policy_hash=old["parent_policy_hash"],
                dataset_hash=dataset_hash,
            ),
        )
        write_once(
            a.output / "complete.json",
            dict(
                status="CORE_TORQUE_ACTOR_RL_TRAINED_NOT_PHYSICALLY_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                declaration=declaration_out,
                actual_BC_actor_optimizer_updates=390,
                actual_SAC_actor_optimizer_updates=counts["actor"],
                critic_optimizer_updates=counts["critic"],
                MC_critic_warmup_steps=128,
                SAC_update_index=learner.update_index,
                metrics=metrics,
                SAC_updates=updates,
                config=asdict(config),
                safety=old["safety"],
                body_hash=old["body_hash"],
                parent_policy_hash=old["parent_policy_hash"],
                dataset_hash=dataset_hash,
                initial_checkpoint=before,
                checkpoint=checkpoint,
                policy=policy,
                training_indices=[0, 3],
                validation_indices=[1, 2],
                physical_gain_verified=False,
                new_physics_steps=0,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), actual_optimizer_steps=counts, new_physics_steps=0),
        )
        raise
    finally:
        for hook in hooks:
            hook.remove()


if __name__ == "__main__":
    main()
