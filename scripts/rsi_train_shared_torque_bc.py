"""Train the existing Core recurrent29-torque actor on native teacher data.

First-pass BC of body control, not precision success, online SAC or promotion.
Two fixed training starts and two validation starts; never fit normalization
on validation rows. All real substeps and teacher/parent actions are admitted.
"""

import argparse
import json
import math
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--teacher", type=Path, required=True)
    p.add_argument("--teacher-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.teacher / "complete.json") != a.teacher_hash:
        raise ValueError("unused output and exact complete native teacher required")
    report = json.loads((a.teacher / "complete.json").read_text())
    declaration = json.loads((a.teacher / "declaration.json").read_text())
    if (
        declaration["protocol"] != "robonaldo.native_torque_teacher.v1"
        or len(report["records"]) != 4
        or declaration["worlds"] != 4
        or report["optimizer_updates"] != 0
        or not all(
            row["arm"] == "single"
            and row["healthy_full_horizon"]
            and row["native_steps"] == 6520
            and row["zero_intervention_full_identity"]
            and all(row["zero_intervention_full_identity"].values())
            for row in report["records"]
        )
    ):
        raise ValueError(
            "all four physically healthy, pass-through-identical full teacher cases required"
        )
    pins = dict(report["input_file_hashes"])
    pins[str(a.teacher / "complete.json")] = a.teacher_hash
    pins[str(a.teacher / "declaration.json")] = digest(a.teacher / "declaration.json")
    for row in report["records"]:
        for artifact in (row["artifact"], row["torque_teacher_artifacts"]["a/"]):
            path = a.teacher / artifact["path"]
            if (
                path.parent.resolve() != a.teacher.resolve()
                or path.is_symlink()
                or digest(path) != artifact["hash"]
            ):
                raise ValueError("exact confined native raw and teacher arrays required")
            pins[str(path)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.g1_neural_torque import G1TeacherTorqueEpisode, G1TorqueSafetyConfig
    from rosclaw.simforge.g1_neural_torque_learning import (
        G1ContinualTorqueActorCritic,
        G1NeuralTorqueLearnerConfig,
        teacher_dataset_hash,
    )
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique immutable canonical teacher scene required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    body_hash = hash_json(layout.profile())
    episodes = []
    for index, row in enumerate(report["records"]):
        teacher = row["torque_teacher_artifacts"]["a/"]
        if (
            row["index"] != index
            or teacher["body_hash"] != body_hash
            or teacher["commands"] != 6520
            or teacher["applied"] != 6520
        ):
            raise ValueError("one exact body and ordered four complete native episodes required")
        with (
            np.load(a.teacher / row["artifact"]["path"]) as raw,
            np.load(a.teacher / teacher["path"]) as data,
        ):
            episode = G1TeacherTorqueEpisode(
                data["observations"].copy(), data["actions"].copy(), data["parent_actions"].copy()
            )
            obs = episode.observations
            if (
                obs.shape != (6520, 102)
                or raw["qpos"].shape != (6520, model.nq)
                or raw["qvel"].shape != (6520, model.nv)
                or np.any(raw["applied_torque_change"])
                or not np.array_equal(raw["parent_torque"], raw["torque"])
                or not np.array_equal(
                    episode.actions, raw["torque"][:, layout.ai].astype(np.float32)
                )
                or not np.array_equal(episode.actions, episode.parent_actions)
                or not np.array_equal(obs[1:, :29], raw["qpos"][:-1, layout.qi].astype(np.float32))
                or not np.array_equal(
                    obs[1:, 29:58], raw["qvel"][:-1, layout.vi].astype(np.float32)
                )
                or not np.array_equal(obs[0, -29:], np.zeros(29, np.float32))
                or not np.array_equal(
                    obs[1:, -29:],
                    (raw["torque"][:-1, layout.ai] / np.array(G1_HARD_TORQUE_LIMITS)).astype(
                        np.float32
                    ),
                )
            ):
                raise ValueError(
                    "teacher features/actions/history do not match actual before-step native states"
                )
            relative = (
                raw["qpos"][:-1, layout.bq : layout.bq + 3]
                - raw["qpos"][:-1, layout.rq : layout.rq + 3]
            )
            if (
                not np.array_equal(obs[1:, 61:64], relative.astype(np.float32))
                or not np.array_equal(
                    obs[1:, 64:67], raw["qvel"][:-1, layout.bv : layout.bv + 3].astype(np.float32)
                )
                or not np.allclose(obs[:, 67:69], np.array([0.0, 0.11], np.float32), atol=0, rtol=0)
                or not np.all(np.isin(obs[:, 71:73], [0.0, 1.0]))
            ):
                raise ValueError("causal ball/task/contact feature columns differ")
            phase = np.arange(6520) / 6520.0 * (2.0 * math.pi)
            if not np.allclose(
                obs[:, 69:71],
                np.stack((np.sin(phase), np.cos(phase)), axis=1).astype(np.float32),
                atol=1e-7,
                rtol=0,
            ):
                raise ValueError("teacher task clock differs from declared6520 native steps")
            episodes.append(episode)
    dataset_hash = teacher_dataset_hash(tuple(episodes))
    config = G1NeuralTorqueLearnerConfig(
        hidden_dim=96, sequence_length=16, batch_size=256, device="cpu", seed=926026
    )
    safety = G1TorqueSafetyConfig(torque_guard_scale=0.85)
    learner = G1ContinualTorqueActorCritic(config, safety=safety)
    initial = {
        name: parameter.detach().clone() for name, parameter in learner.actor.named_parameters()
    }
    updates = []

    def post_step(optimizer, args, kwargs):
        updates.append(len(updates) + 1)
        if len(updates) % 20 == 0:
            print(json.dumps(dict(actual_BC_Adam_updates=len(updates))), flush=True)

    hook = learner.actor_optimizer.register_step_post_hook(post_step)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="shared_native_torque_bc.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            teacher_hash=a.teacher_hash,
            body_hash=body_hash,
            parent_policy_hash=a.teacher_hash,
            dataset_hash=dataset_hash,
            config=asdict(config),
            safety=asdict(safety),
            epochs=30,
            stride=4,
            training_indices=[0, 3],
            validation_indices=[1, 2],
            all_native_samples=26080,
            training_native_samples=13040,
            validation_native_samples=13040,
            validation_is_not_private_Fresh=True,
            teacher_not_precision_qualified=True,
            new_physics_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )

    def artifact(name, payload):
        with (a.output / name).open("xb") as stream:
            stream.write(payload)
        return dict(path=name, hash=digest(a.output / name))

    try:
        before = artifact("initial-checkpoint.pt", learner.checkpoint_bytes())
        metrics = learner.pretrain_behavior(
            (episodes[0], episodes[3]), validation=(episodes[1], episodes[2]), epochs=30, stride=4
        )
        hook.remove()
        if not all(row.finite for row in metrics):
            raise ValueError("nonfinite BC training metrics")
        state_steps = {
            name: int(learner.actor_optimizer.state[parameter]["step"].item())
            for name, parameter in learner.actor.named_parameters()
        }
        if not updates or any(step != len(updates) for step in state_steps.values()):
            raise ValueError("actual actor Adam state differs from counted updates")
        changed = {
            name: float(torch.max(torch.abs(parameter.detach() - initial[name])).item())
            for name, parameter in learner.actor.named_parameters()
        }
        if not any(changed.values()):
            raise ValueError("BC did not change the actor")
        checkpoint = artifact("trained-checkpoint.pt", learner.checkpoint_bytes())
        policy = artifact(
            "trained-torque-policy.bin",
            learner.artifact_bytes(
                body_hash=body_hash, parent_policy_hash=a.teacher_hash, dataset_hash=dataset_hash
            ),
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="CORE_TORQUE_ACTOR_BC_TRAINED_NOT_PHYSICALLY_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                teacher_hash=a.teacher_hash,
                body_hash=body_hash,
                parent_policy_hash=a.teacher_hash,
                dataset_hash=dataset_hash,
                config=asdict(config),
                safety=asdict(safety),
                metrics=[row.to_dict() for row in metrics],
                actual_BC_actor_optimizer_updates=len(updates),
                actor_Adam_steps=state_steps,
                maximum_parameter_changes=changed,
                SAC_update_index=learner.update_index,
                critic_optimizer_updates=0,
                initial_checkpoint=before,
                checkpoint=checkpoint,
                policy=policy,
                training_indices=[0, 3],
                validation_indices=[1, 2],
                teacher_not_precision_qualified=True,
                new_physics_steps=0,
                physical_gain_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        hook.remove()
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), actual_BC_updates=len(updates), new_physics_steps=0),
        )
        raise


if __name__ == "__main__":
    main()
