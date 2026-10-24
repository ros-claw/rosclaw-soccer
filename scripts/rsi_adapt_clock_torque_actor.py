"""Bounded supervised adaptation of the existing GRU/Adam to faster real teachers.

No actor reinitialization, observation-statistics refit, critic update or RL
claim. Retain historical behavior through the same original training anchors.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("training", "teacher", "identity", "legacy-teacher", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "training-hash",
        "teacher-hash",
        "identity-hash",
        "legacy-teacher-hash",
        "core-expected-head",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, required=True)
    a = p.parse_args()
    if a.seed not in (926051, 926052, 926053):
        p.error("three predeclared paired adaptation seeds required")
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused adaptation output required")
    pins, reports = {}, {}
    for name in ("training", "teacher", "identity", "legacy_teacher"):
        root = getattr(a, name)
        path = root / "complete.json"
        expected = getattr(a, name + "_hash")
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact completed immutable adaptation ancestry required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting adaptation ancestry")
            pins[key] = value
        pins[str(path)] = expected
        reports[name] = report
    old, teacher, identity, legacy = (
        reports[k] for k in ("training", "teacher", "identity", "legacy_teacher")
    )
    if (
        old["actual_BC_actor_optimizer_updates"] != 390
        or old["SAC_update_index"] != 0
        or old["critic_optimizer_updates"] != 0
        or not teacher["all_exact"]
        or not identity["all_exact"]
        or len(teacher["records"]) != 16
        or len(identity["records"]) != 4
        or len(legacy["records"]) != 4
    ):
        raise ValueError(
            "original390-step BC, exact identity4 and complete fast16 teacher required"
        )

    def pin(root, entry):
        path = root / entry["path"]
        if (
            path.parent.resolve() != root.resolve()
            or path.is_symlink()
            or digest(path) != entry["hash"]
        ):
            raise ValueError("exact confined ordinary artifact required")
        pins[str(path)] = entry["hash"]
        return path

    paths = {}
    for key, report, root in (
        ("teacher", teacher, a.teacher),
        ("identity", identity, a.identity),
        ("legacy", legacy, a.legacy_teacher),
    ):
        paths[key] = [
            pin(
                root,
                row["torque_teacher_artifacts"]["a/"]
                if key == "legacy"
                else row["native_clock_teacher"],
            )
            for row in report["records"]
        ]
    checkpoint = pin(a.training, old["checkpoint"])
    original_policy = pin(a.training, old["policy"])
    source_paths = [
        Path(k)
        for k, v in pins.items()
        if k.endswith("/complete.json") and v == teacher["source_hash"]
    ]
    if len(source_paths) != 1:
        raise ValueError("one exact fast teacher physical producer required")
    fast_source = json.loads(source_paths[0].read_text())
    fast_decl_path = source_paths[0].parent / "declaration.json"
    pins[str(fast_decl_path)] = digest(fast_decl_path)
    fast_decl = json.loads(fast_decl_path.read_text())
    if (
        fast_decl["reference_rate"] != 1.18
        or fast_decl["protocol"] != "robonaldo.reference_clock_authority.v1"
        or len(fast_source["records"]) != 16
        or not all(row["healthy_full_horizon"] for row in fast_source["records"])
    ):
        raise ValueError(
            "preselected full16 healthy fast parent required, no silent survivor filtering"
        )
    parent_hash = fast_decl["reference_controller_hash"]
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    source_pins(pins)
    check_pins(pins)
    import numpy as np
    import torch
    from rosclaw.simforge.g1_neural_torque import G1TeacherTorqueEpisode, G1TorqueSafetyConfig
    from rosclaw.simforge.g1_neural_torque_learning import (
        G1ContinualTorqueActorCritic,
        G1NeuralTorqueLearnerConfig,
        _teacher_sequences,
    )

    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    identity_exact = []
    for before, after in zip(paths["legacy"], paths["identity"], strict=True):
        with np.load(before) as b, np.load(after) as c:
            identity_exact.append(
                {
                    k: bool(np.array_equal(b[k], c[k]))
                    for k in ("observations", "actions", "parent_actions")
                }
            )
    if not all(all(row.values()) for row in identity_exact):
        raise ValueError("independent teacher collector changed original102-observation/action ABI")

    def episodes(files):
        result = []
        for path in files:
            with np.load(path) as data:
                episode = G1TeacherTorqueEpisode(
                    data["observations"].copy(),
                    data["actions"].copy(),
                    data["parent_actions"].copy(),
                )
            if episode.observations.shape != (6520, 102) or not np.array_equal(
                episode.actions, episode.parent_actions
            ):
                raise ValueError("all actual native102/29 pass-through samples required")
            result.append(episode)
        return result

    fast, historical = episodes(paths["teacher"]), episodes(paths["legacy"])
    train_ids, val_ids = (0, 3, 4, 7, 8, 11, 12, 15), (1, 2, 5, 6, 9, 10, 13, 14)
    x, y = _teacher_sequences(tuple(fast[i] for i in train_ids), sequence_length=16, stride=10)
    vx, vy = _teacher_sequences(tuple(fast[i] for i in val_ids), sequence_length=16, stride=10)
    hx, hy = _teacher_sequences(tuple(historical[i] for i in (0, 3)), sequence_length=16, stride=10)
    hvx, hvy = _teacher_sequences(
        tuple(historical[i] for i in (1, 2)), sequence_length=16, stride=10
    )
    learner = G1ContinualTorqueActorCritic(
        G1NeuralTorqueLearnerConfig(**old["config"]), safety=G1TorqueSafetyConfig(**old["safety"])
    )
    learner.restore_checkpoint(checkpoint.read_bytes())
    if (
        learner.artifact_bytes(
            body_hash=old["body_hash"],
            parent_policy_hash=old["parent_policy_hash"],
            dataset_hash=old["dataset_hash"],
        )
        != original_policy.read_bytes()
    ):
        raise ValueError("resumption changed original actor before adaptation")
    for row in teacher["records"]:
        if row["native_clock_teacher"]["body_hash"] != old["body_hash"]:
            raise ValueError("actual fast teacher body differs from original BC")
    norm = (learner.observation_mean.copy(), learner.observation_std.copy())
    initial = learner.actor_snapshot()
    limits = learner.actor.action_limits

    def mse(xx, yy):
        total = 0.0
        with torch.no_grad():
            for i in range(0, len(xx), 256):
                pred = learner.actor.deterministic(learner._observation_tensor(xx[i : i + 256]))
                target = learner._tensor(yy[i : i + 256]).clamp(-limits, limits)
                total += float(((pred - target) / limits).square().sum())
        return total / (len(xx) * 29)

    before = dict(fast_validation=mse(vx, vy), historical_validation=mse(hvx, hvy))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="retained_fast_clock_native_torque_BC.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=a.seed,
            steps=128,
            teacher_hash=a.teacher_hash,
            training_indices=train_ids,
            validation_indices=val_ids,
            sequence_length=16,
            stride=10,
            observation_statistics_refitted=False,
            old_Adam_preserved=True,
            parent_controller_migration={"old": old["parent_policy_hash"], "new": parent_hash},
            reference_rate=1.18,
            historical_label_weight=0.5,
            historical_distillation_weight=0.5,
            critic_updates=0,
            SAC_updates=0,
            identity_102_29_exact=identity_exact,
            physics_success_not_inferred_from_BC=True,
            hardware_authorized=False,
        ),
    )
    rng = np.random.default_rng(a.seed)
    counts = dict(actor=0, critic=0, alpha=0)
    hooks = []
    for name in counts:

        def step_hook(opt, args, kwargs, name=name):
            counts[name] += 1

        hooks.append(getattr(learner, name + "_optimizer").register_step_post_hook(step_hook))
    losses = []
    for step in range(128):
        i, j = rng.choice(len(x), 256, replace=False), rng.choice(len(hx), 256, replace=False)
        obs, anchor = learner._observation_tensor(x[i]), learner._observation_tensor(hx[j])
        pred, apred = learner.actor.deterministic(obs), learner.actor.deterministic(anchor)
        target, atarget = (
            learner._tensor(y[i]).clamp(-limits, limits),
            learner._tensor(hy[j]).clamp(-limits, limits),
        )
        with torch.no_grad():
            retained = learner.parent_actor.deterministic(anchor)
        loss = (
            ((pred - target) / limits).square().mean()
            + 0.5 * ((apred - atarget) / limits).square().mean()
            + 0.5 * ((apred - retained) / limits).square().mean()
        )
        if not torch.isfinite(loss):
            raise ValueError("nonfinite retained BC adaptation")
        learner.actor_optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(learner.actor.parameters(), 10.0)
        learner.actor_optimizer.step()
        losses.append(float(loss.detach()))
        if (step + 1) % 32 == 0:
            print(
                json.dumps(dict(actual_actor_Adam_steps=counts["actor"], loss=losses[-1])),
                flush=True,
            )
    for hook in hooks:
        hook.remove()
    if counts != dict(actor=128, critic=0, alpha=0) or not all(
        np.array_equal(a, b)
        for a, b in zip(norm, (learner.observation_mean, learner.observation_std), strict=True)
    ):
        raise ValueError("adaptation changed frozen statistics or optimizer budget")
    after = dict(fast_validation=mse(vx, vy), historical_validation=mse(hvx, hvy))
    changed = {
        k: float(np.max(np.abs(v - initial[k]))) for k, v in learner.actor_snapshot().items()
    }
    if not any(changed.values()):
        raise ValueError("actor adaptation had no actual weight changes")
    dataset_hash = hash_json(
        dict(
            teacher_hash=a.teacher_hash,
            training_indices=train_ids,
            historical_teacher_hash=a.legacy_teacher_hash,
            historical_indices=[0, 3],
        )
    )

    def artifact(name, content):
        with (a.output / name).open("xb") as f:
            f.write(content)
        return dict(path=name, hash=digest(a.output / name))

    saved_checkpoint = artifact("adapted-checkpoint.pt", learner.checkpoint_bytes())
    saved_policy = artifact(
        "adapted-torque-policy.bin",
        learner.artifact_bytes(
            body_hash=old["body_hash"], parent_policy_hash=parent_hash, dataset_hash=dataset_hash
        ),
    )
    check_pins(pins)
    write_once(
        a.output / "complete.json",
        dict(
            status="CORE_TORQUE_ACTOR_BC_TRAINED_NOT_PHYSICALLY_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            actual_BC_actor_optimizer_updates=518,
            new_BC_actor_updates=128,
            actual_SAC_actor_optimizer_updates=0,
            critic_optimizer_updates=0,
            SAC_update_index=0,
            config=old["config"],
            safety=old["safety"],
            body_hash=old["body_hash"],
            parent_policy_hash=parent_hash,
            dataset_hash=dataset_hash,
            training_hash=a.training_hash,
            teacher_hash=a.teacher_hash,
            seed=a.seed,
            training_indices=train_ids,
            validation_indices=val_ids,
            policy=saved_policy,
            checkpoint=saved_checkpoint,
            before=before,
            after=after,
            actor_parameter_max_change=changed,
            actual_optimizer_counts=counts,
            normalization_unchanged=True,
            identity_102_29_exact=identity_exact,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
