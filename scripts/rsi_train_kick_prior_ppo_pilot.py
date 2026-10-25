"""One own-policy head-PPO proposal, four critic steps; not a promoted motor skill."""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("rollouts", "bank", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("rollouts-hash", "bank-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    rp, bp = a.rollouts / "complete.json", a.bank / "complete.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, bp))
        or (digest(rp) != a.rollouts_hash or digest(bp) != a.bank_hash)
    ):
        raise ValueError("new output and exact own-rollout/causal-anchor banks required")
    rollout, bank = (json.loads(p.read_text()) for p in (rp, bp))
    if rollout[
        "status"
    ] != "EIGHT_OWN_POLICY_KICK_TRAJECTORIES_COMMITTED_AND_RAW_LIKELIHOODS_REPLAYED" or (
        rollout["actor_optimizer_updates"] != 0
        or rollout["worlds"] != 8
        or rollout["promotion_authorized"] is not False
        or [r["course_index"] for r in rollout["results"]] != [0, 1, 10, 11, 20, 21, 30, 31]
        or bank["status"] != "CAUSAL_KICK_ENTRY_BANK_RECONSTRUCTED_FIRST_ACTIONS_EXACT_NOT_TRAINED"
    ):
        raise ValueError(
            "complete original own-policy experience, not teacher demonstrations required"
        )
    pins = dict(rollout["input_file_hashes"])
    for name, h in bank["input_file_hashes"].items():
        if name in pins and pins[name] != h:
            raise ValueError("conflicting original experience/anchor identity")
        pins[name] = h
    pins.update({str(rp): a.rollouts_hash, str(bp): a.bank_hash})
    anchor_path = a.bank / bank["artifact"]["path"]
    if anchor_path.parent.resolve() != a.bank.resolve() or anchor_path.is_symlink():
        raise ValueError("confined actual causal anchors required")
    pins[str(anchor_path)] = bank["artifact"]["hash"]
    for row in rollout["results"]:
        for key in ("artifact", "rng_checkpoint"):
            artifact = row[key]
            if artifact is None:
                continue
            path = a.rollouts / artifact["path"]
            if path.is_symlink() or path.parent.resolve() != a.rollouts.resolve():
                raise ValueError("confined raw own experience and generator states required")
            pins[str(path)] = artifact["hash"]
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.actor_trust_region import bounded_adam_proposal
    from rosclaw_soccer.rsi.kick_prior_actor import KICK_PRIOR_HASH, KickPriorActor
    from rosclaw_soccer.rsi.kick_prior_ppo import (
        KickValueHead,
        freeze_kick_features,
        own_ppo_objective,
        value_features,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.sim.contracts import hash_json

    torch.set_num_threads(1)
    torch.manual_seed(926201)
    models = [Path(k) for k in pins if k.endswith("/policy/robonaldo/model/policy-obs-aic.onnx")]
    if len(models) != 1 or bank["policy_model_hash"] != KICK_PRIOR_HASH:
        raise ValueError("exact current kick prior required")
    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    if len(scenes) != 1:
        raise ValueError("unique exact compiled research body required")
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    if hash_json(layout.profile()) != bank["body_hash"]:
        raise ValueError("actual body differs from the neural policy data contract")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g1.own_kick_head_ppo_pilot.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            seed=926201,
            actor_proposals=1,
            critic_steps=4,
            actor_lr=1e-4,
            critic_lr=0.001,
            terminal_discount=0.99,
            reward="existing H300 training_reward; if fullH360 unhealthy min(reward,-10)",
            original_first_three_linear_layers_frozen=True,
            trainable_motor_parameters=3741,
            ppo_ratio_clip=[0.8, 1.2],
            raw_std=0.015,
            ar_rho=0.9,
            actor_trust="existing bounded Adam, max(own conditional KL, original-anchor KL)<=.005",
            anchor_kl_loss_weight=0.1,
            anchor_rows=1052,
            advantages=(
                "pre-update value baseline, group(equal episode)-weighted standardized MC return"
            ),
            separate_value_gradients=True,
            no_best_seed_selection=True,
            provisional_motor_candidate_only=True,
            physical_football_gain_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    actor = KickPriorActor.from_onnx(models[0], expected_hash=KICK_PRIOR_HASH)
    plastic = freeze_kick_features(actor)
    if len(plastic) != 2 or sum(p.numel() for p in plastic) != 3741:
        raise ValueError("only original output weight/bias may learn")
    critic = KickValueHead()
    actor_optimizer = torch.optim.Adam(plastic, lr=1e-4)
    value_optimizer = torch.optim.Adam(critic.parameters(), lr=0.001)
    initial = {k: v.detach().clone() for k, v in actor.state_dict().items()}
    with np.load(anchor_path) as saved:
        anchors = torch.from_numpy(saved["features"][:, :547].copy())
    if anchors.shape != (1052, 547):
        raise ValueError("all original causal anchor rows required")
    with torch.no_grad():
        anchor_means = actor(anchors).detach()
    episodes = []
    source_returns = []
    for row in rollout["results"]:
        with np.load(a.rollouts / row["artifact"]["path"]) as saved:
            arrays = dict(saved)
        if (
            row["ball_qpos_address"] != layout.bq
            or row["robot_qpos_address"] != layout.rq
            or (
                np.any(
                    np.abs(arrays["torque"][:, layout.ai])
                    > 0.85 * np.array(G1_HARD_TORQUE_LIMITS) + 1e-12
                )
            )
        ):
            raise ValueError("actual body addresses or applied hard effort caps differ")
        outcome = relabel_arrays(
            arrays,
            controls=row["controls"],
            ball_address=row["ball_qpos_address"],
            root_address=row["robot_qpos_address"],
        )
        pose = arrays["qpos"][:, row["robot_qpos_address"] : row["robot_qpos_address"] + 7]
        healthy = row["controls"] == 360 and not bool(
            np.any((pose[:, 2] < 0.65) | (1 - 2 * (pose[:, 4] ** 2 + pose[:, 5] ** 2) < 0.9))
        )
        if (
            hash_json(outcome) != hash_json(row["outcome"])
            or healthy != row["healthy_full_horizon"]
        ):
            raise ValueError("actual complete physical outcome differs")
        n = row["own_committed_neural_samples"]
        if n == 0:
            source_returns.append(dict(course_index=row["course_index"], no_motor_entry=True))
            continue
        frames = arrays["own_control_frames"]
        if not np.array_equal(frames, np.arange(row["entry_control"], row["controls"])) or (
            len(frames) != n
            or not np.array_equal(arrays["own_executed_pd_target"], arrays["joint_targets"][frames])
        ):
            raise ValueError("complete actual motor commitments required")
        obs = torch.from_numpy(arrays["own_observations"].copy())
        old = torch.from_numpy(arrays["own_old_means"].copy())
        with torch.no_grad():
            if float((actor(obs) - old).abs().max()) > 1e-5:
                raise ValueError("experience is not from this initial actor")
        features = value_features(actor, obs, torch.from_numpy(frames))
        reward = float(outcome["training_reward"])
        if not healthy:
            reward = min(reward, -10.0)
        returns = torch.tensor(
            [reward * 0.99 ** (n - 1 - i) for i in range(n)], dtype=torch.float32
        )
        with torch.no_grad():
            old_values = critic(features).detach()
        episodes.append(
            dict(
                observations=obs,
                old_means=old,
                raw_actions=torch.from_numpy(arrays["own_raw_actions"].copy()),
                old_logp=torch.from_numpy(arrays["own_raw_logp"].copy()),
                features=features,
                returns=returns,
                advantages=(returns - old_values).detach(),
                actual_pd_commitment_verified=True,
                external_teacher=False,
            )
        )
        source_returns.append(
            dict(course_index=row["course_index"], terminal_reward=reward, motor_samples=n)
        )
    if len(episodes) != 8:
        raise ValueError("pilot requires all eight actually entered own-policy trajectories")
    mean = torch.stack([e["advantages"].mean() for e in episodes]).mean()
    variance = torch.stack([(e["advantages"] - mean).square().mean() for e in episodes]).mean()
    for e in episodes:
        e["advantages"] = ((e["advantages"] - mean) / (variance.sqrt() + 1e-8)).detach()
    value_losses = []
    for _ in range(4):
        value_optimizer.zero_grad(set_to_none=True)
        loss = torch.stack(
            [(critic(e["features"]) - e["returns"]).square().mean() for e in episodes]
        ).mean()
        if not torch.isfinite(loss):
            raise ValueError("nonfinite value loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(critic.parameters(), 1.0, error_if_nonfinite=True)
        value_optimizer.step()
        value_losses.append(float(loss.detach()))
    if any(p.grad is not None for p in actor.parameters()):
        raise ValueError("value training changed motor gradients")
    log_std = torch.full((29,), math.log(0.015), dtype=torch.float32)
    decision = bounded_adam_proposal(
        plastic,
        actor_optimizer,
        lambda: own_ppo_objective(actor, episodes, anchors, anchor_means, log_std),
    )
    changed = {k: int((v - initial[k]).count_nonzero()) for k, v in actor.state_dict().items()}
    if any(value for key, value in changed.items() if not key.startswith("actor.6.")):
        raise ValueError("original motor features changed")
    accepted = decision["accepted"]
    if accepted != bool(sum(changed.values())):
        raise ValueError("accepted actor proposal did not match actual weight changes")
    actor_steps = [int(s["step"]) for s in actor_optimizer.state.values()]
    if (accepted and actor_steps != [1, 1]) or (not accepted and actor_steps):
        raise ValueError("actual actor Adam moments must match accepted/reverted step accounting")
    if (
        any(int(s["step"]) != 4 for s in value_optimizer.state.values())
        or len(value_optimizer.state) != 4
    ):
        raise ValueError("four actual value updates for all four parameter tensors required")
    checkpoint = a.output / "provisional-kick-head-ppo.pt"
    torch.save(
        dict(
            schema="soccer.g1.kick_head_ppo_pilot_checkpoint.v1",
            actor=actor.state_dict(),
            critic=critic.state_dict(),
            actor_optimizer=actor_optimizer.state_dict(),
            critic_optimizer=value_optimizer.state_dict(),
            torch_rng=torch.get_rng_state(),
            source_model_hash=KICK_PRIOR_HASH,
            rollouts_hash=a.rollouts_hash,
            bank_hash=a.bank_hash,
            body_hash=bank["body_hash"],
            actual_actor_optimizer_updates=int(accepted),
            actual_value_optimizer_updates=4,
            actor_update_attempts=1,
            seed=926201,
            simulation_only=True,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
        checkpoint,
    )
    restored = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if any(not torch.equal(v, restored["actor"][k]) for k, v in actor.state_dict().items()):
        raise ValueError("safe checkpoint replay differs")
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="ONE_OWN_POLICY_KICK_PPO_PROPOSAL_AND_FOUR_VALUE_UPDATES_NOT_PHYSICALLY_QUALIFIED",
            source_head=a.expected_head,
            input_file_hashes=pins,
            actor_decision=decision,
            actual_actor_optimizer_updates=int(accepted),
            actor_update_attempts=1,
            actual_value_optimizer_updates=4,
            changed_actor_parameter_elements=changed,
            source_returns=source_returns,
            source_motor_samples=sum(len(e["observations"]) for e in episodes),
            value_losses=value_losses,
            original_motor_features_unchanged=True,
            checkpoint=dict(path=checkpoint.name, hash=digest(checkpoint)),
            physical_steps_added=0,
            physical_football_gain_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
