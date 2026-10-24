"""Three fixed-seed BC candidates from independently proven receiving states.

Not PPO, online RL, direct torque, Fresh, promotion, or a student physical
execution. Teacher-state command features are reconstructed WITHOUT teacher
commands as inputs. Failed teacher worlds remain in the bank, not BC targets.
"""

import argparse
import json
from pathlib import Path

SEEDS = (732023, 732024, 732025)
STEPS = 400


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "own-root", "review-root", "scene", "velocity-config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("review-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, phase_delta
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.receiving_command_actor import ReceivingCommandActor, command_inputs
    from rosclaw_soccer.rsi.receiving_command_learning import fit_command_bc
    from rosclaw_soccer.rsi.refinement_onpolicy_checkpoint import file_hash
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused once-only BC bank/training output required")
    for root in (a.source_root, a.own_root, a.review_root):
        if root.is_symlink() or (root / "failed.json").exists():
            raise ValueError("closed ordinary complete native/role evidence required")
    rp = a.review_root / "complete.json"
    if file_hash(rp) != a.review_hash:
        raise ValueError("exact predeclared all96 paired review required")
    review = json.loads(rp.read_text())
    source, own = [
        json.loads((root / "complete.json").read_text()) for root in (a.source_root, a.own_root)
    ]
    effect = review["overall"]
    if (
        review["status"] != "PROJECTED_RECEIVER_ROLE_PAIRED96_REVIEWED_NOT_LEARNING_OR_PROMOTION"
        or review["source_hash"] != file_hash(a.source_root / "complete.json")
        or review["own_hash"] != file_hash(a.own_root / "complete.json")
        or own["source_hash"] != review["source_hash"]
        or own["independent_whole_world_replay"] is not True
        or own["sampling_law_audit_passed"] is not True
        or own["records"]
        != [{k: v for k, v in r.items() if k != "artifacts"} for r in source["records"]]
        or any(
            effect[k]
            for k in (
                "safety_losses",
                "controlled_regressions",
                "functional_regressions",
                "strict_regressions",
            )
        )
        or effect["functional_new"] <= effect["functional_old"]
        or len(source["records"]) != 96
    ):
        raise ValueError("closed all96 improvement with ZERO old-role/safety regressions required")
    pins = dict(review["input_file_hashes"])
    pins[str(rp)] = a.review_hash
    for path in (a.scene, a.velocity_config):
        if source["input_file_hashes"].get(str(path)) != file_hash(path):
            raise ValueError("original unchanged body and command codec required")
    source_pins(pins)
    check_pins(pins)
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    params = VelocityPolicyParameters.load(
        a.velocity_config, expected_hash=pins[str(a.velocity_config)]
    )
    model = mujoco.MjModel.from_xml_path(str(a.scene))
    layout = NativeVelocityLayout(
        model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
    )
    records = source["records"]
    forward = (
        np.array(records[64]["course"]["ball_planar_offset_world"])
        - np.array(records[32]["course"]["ball_planar_offset_world"])
    ) / 0.01
    if abs(np.linalg.norm(forward) - 1) > 1e-10:
        raise ValueError("unchanged declared symmetric physical initial courses required")
    lateral = np.array([-forward[1], forward[0]])
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            review_hash=a.review_hash,
            seeds=list(SEEDS),
            steps_per_seed=STEPS,
            minibatch=256,
            learning_rate=0.0003,
            optimizer="Adam",
            algorithm="OFFLINE_BC_COMMAND_ROLE_NOT_PPO",
            validation_noise_groups=[i for i in range(32) if i % 4 == 3],
            split_shared_across_three_initials=True,
            train_only_normalization=True,
            source_commands_are_labels_never_inputs=True,
            new_physics_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    candidates, total_steps = [], 0
    try:
        inputs, commands, worlds, eligible = [], [], [], []
        for index, row in enumerate(records):
            raw = {}
            for key in ("initial", "physical", "B"):
                art = row["artifacts"][key]
                path = a.source_root / art["path"]
                if pins.get(str(path)) != art["hash"]:
                    raise ValueError("all teacher-state primitives must be pinned in own review")
                with np.load(path, allow_pickle=False) as z:
                    raw[key] = {k: z[k].copy() for k in z.files}
            initial, physics, b = raw["initial"], raw["physical"], raw["B"]
            n = row["controls"]
            qs = np.concatenate((initial["qpos"][None], physics["qpos"][:-1, -1]), axis=0)
            vs = np.concatenate((initial["qvel"][None], physics["qvel"][:-1, -1]), axis=0)
            if (
                qs.shape != (n, model.nq)
                or vs.shape != (n, model.nv)
                or b["obs"].shape != (n, 98)
                or b["command"].shape != (n, 3)
            ):
                raise ValueError("complete actual pre-action teacher body histories required")
            feet = np.r_[
                False,
                np.maximum.accumulate(
                    np.any(physics["ball_forces"][:-1, :, 1, :2] > 1, axis=(1, 2))
                ),
            ]
            other = np.r_[
                False,
                np.maximum.accumulate(np.any(physics["ball_forces"][:-1, :, 1, 2] > 1, axis=1)),
            ]
            anchor = np.array(row["course"]["target"], dtype=np.float64)
            anchor[:2] += 0.25 * forward + 0.12 * lateral
            home = initial["qpos"][layout.rq : layout.rq + 2].copy()
            phase, launched = phase_delta(), False
            outcome = row["outcome"]
            success = bool(
                outcome["B_legal_functional_receive_before_deadline"]
                and outcome["whole_body_healthy"]
                and not outcome["B_excluded_contact"]
            )
            for t in range(n):
                q = np.r_[
                    qs[t, layout.rq : layout.rq + 7],
                    qs[t, layout.qi],
                    qs[t, layout.bq : layout.bq + 7],
                ]
                v = np.r_[
                    vs[t, layout.rv : layout.rv + 6],
                    vs[t, layout.vi],
                    vs[t, layout.bv : layout.bv + 6],
                ]
                features, _, phase, launched = command_inputs(
                    q,
                    v,
                    b["obs"][t, :3].astype(np.float64),
                    params=params,
                    last_action=b["obs"][t, 69:].copy(),
                    phase=phase,
                    home_xy=home,
                    pass_direction=forward,
                    anchor_world=anchor,
                    launch_seen=launched,
                    foot_contact_seen=bool(feet[t]),
                    other_body_contact_seen=bool(other[t]),
                )
                if phase != b["phase"][t]:
                    raise ValueError("original own gait phase must be reconstructed exactly")
                inputs.append(features)
            commands.extend(b["command"])
            worlds.extend([index] * n)
            eligible.extend([success] * n)
        x, y = torch.from_numpy(np.stack(inputs)), torch.from_numpy(np.stack(commands))
        world = torch.tensor(worlds, dtype=torch.int64)
        eligible_mask = torch.tensor(eligible, dtype=torch.bool)
        validation_group = world.remainder(32).remainder(4) == 3
        training_mask = eligible_mask & ~validation_group
        validation_mask = eligible_mask & validation_group
        if (
            not torch.isfinite(y).all()
            or y.dtype != torch.float32
            or ((y < torch.tensor([-0.5, -0.5, -1])) | (y > torch.tensor([1.0, 0.5, 1.0]))).any()
            or any(not (training_mask & (world // 32 == i)).any() for i in range(3))
            or any(not (validation_mask & (world // 32 == i)).any() for i in range(3))
        ):
            raise ValueError(
                "original command bounds and success support in all three initial clusters required"
            )
        bank_path = a.output / "teacher-command-bank.npz"
        np.savez_compressed(
            bank_path,
            observations=x.numpy(),
            commands=y.numpy(),
            world=world.numpy(),
            eligible=eligible_mask.numpy(),
            training=training_mask.numpy(),
            validation=validation_mask.numpy(),
        )
        bank_hash = file_hash(bank_path)
        train_ids = torch.nonzero(training_mask, as_tuple=True)[0]
        val_ids = torch.nonzero(validation_mask, as_tuple=True)[0]
        for seed in SEEDS:

            def observe_step(step, loss, norm, current_seed=seed):
                nonlocal total_steps
                total_steps += 1
                if step % 50 == 0:
                    print(
                        json.dumps(
                            dict(
                                seed=current_seed,
                                optimizer_steps=step,
                                loss=loss,
                                gradient_norm=norm,
                            )
                        ),
                        flush=True,
                    )

            fitted = fit_command_bc(
                x, y, training_mask, validation_mask, seed=seed, steps=STEPS, on_step=observe_step
            )
            actor = fitted["actor"]
            cp = a.output / f"command-actor-seed{seed}.pt"
            torch.save(
                dict(
                    schema="soccer.receiving_command_bc.v1",
                    actor=actor.state_dict(),
                    optimizer=fitted["optimizer"],
                    batch_rng=fitted["batch_rng"],
                    seed=seed,
                    optimizer_updates=STEPS,
                    bank_hash=bank_hash,
                    review_hash=a.review_hash,
                    policy_domain=actor.policy_domain,
                ),
                cp,
            )
            restored = torch.load(cp, map_location="cpu", weights_only=True)
            replica = ReceivingCommandActor(actor.observation_mean, actor.observation_scale)
            replica.load_state_dict(restored["actor"], strict=True)
            with torch.no_grad():
                if not torch.equal(actor(x[:32]), replica(x[:32])):
                    raise ValueError("safe checkpoint reload changed neural command outputs")
            result = dict(
                seed=seed,
                optimizer_updates=STEPS,
                changed_parameter_tensors=fitted["changed_parameter_tensors"],
                before=fitted["before"],
                after=fitted["after"],
                checkpoint_path=str(cp),
                checkpoint_hash=file_hash(cp),
                own_native_student_execution=False,
                teacher_state_BC_not_onpolicy=True,
            )
            candidates.append(result)
            write_once(a.output / f"seed{seed}.json", result)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="RECEIVING_COMMAND_BC_THREE_SEEDS_TRAINED_PENDING_OWN_NATIVE",
                source_head=a.expected_head,
                input_file_hashes=pins,
                review_hash=a.review_hash,
                bank_hash=bank_hash,
                rows=len(x),
                worlds=96,
                training_rows=len(train_ids),
                validation_rows=len(val_ids),
                failed_world_rows_retained_not_BC=int((~eligible_mask).sum()),
                normalization_fitted_on_training_rows_only=True,
                validation_noise_groups_not_Fresh=True,
                candidates=candidates,
                optimizer_updates=total_steps,
                new_physical_steps=0,
                critic_updates=0,
                online_RL_proven=False,
                neural_physical_growth_proven=False,
                direct_torque_policy=False,
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
                completed_candidates=candidates,
                actual_optimizer_steps=total_steps,
                new_physical_steps=0,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
