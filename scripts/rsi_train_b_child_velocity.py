"""Continue B's SIM actor/critic from independently replayed own experiences."""

import argparse
import json
import math
from collections import Counter
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters, target29
from rosclaw_soccer.rsi.precontact_feedback import full_history_precontact_feedback
from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor, precontact_gate
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic, episode_balanced_value_loss
from rosclaw_soccer.rsi.receiver_bounded_update import bounded_actor_update
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean
from scripts.rsi_review_relay_balance_counterfactual import bind, report


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--group", nargs=2, action="append", required=True, metavar=("SOURCE", "AUDIT"))
    p.add_argument("--audit-hash", action="append", required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--parent-hash", required=True)
    p.add_argument("--anchor-data", type=Path, required=True)
    p.add_argument("--anchor-data-hash", required=True)
    p.add_argument("--safe-continuation", action="store_true")
    p.add_argument("--receiver-potential", action="store_true")
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.receiver_potential and not a.safe_continuation:
        p.error("receiver potential requires unchanged safe-continuation base reward")
    clean(a.expected_head)
    if len(a.group) != 2 or len(a.audit_hash) != 2:
        raise ValueError("both predeclared training blocks and replay hashes required")
    if a.output.exists() or a.output.resolve().is_relative_to(Path.cwd()):
        raise ValueError("unused external output required")
    anchor_receipt = a.anchor_data / "complete.json"
    if any(
        path.is_symlink() or digest(path) != h
        for path, h in ((a.parent, a.parent_hash), (anchor_receipt, a.anchor_data_hash))
    ):
        raise ValueError("exact parent and old anchor data required")
    anchor_info = json.loads(anchor_receipt.read_text())
    identity = anchor_info["model_identity"]
    model, mh, config, ch = identity
    params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
    scale = np.asarray(params.action_scale, np.float32)
    if anchor_info.get("sampled_controls") != 6331 or anchor_info.get("episodes") != 24:
        raise ValueError("complete previous numerical anchor set required")
    anchor_x, anchor_r, pins = (
        [],
        [],
        [(a.parent, a.parent_hash), (anchor_receipt, a.anchor_data_hash)],
    )
    for rec in anchor_info["records"]:
        path = a.anchor_data / rec["path"]
        if (
            path.is_symlink()
            or path.resolve().parent != a.anchor_data.resolve()
            or digest(path) != rec["hash"]
        ):
            raise ValueError("exact prior anchor rows required")
        with np.load(path, allow_pickle=False) as z:
            anchor_x.append(z["inputs"].copy())
            anchor_r.append(z["reference"].copy())
        pins.append((path, rec["hash"]))
    anchor_x, anchor_r = (
        torch.from_numpy(np.concatenate(anchor_x)),
        torch.from_numpy(np.concatenate(anchor_r)),
    )
    if len(anchor_x) != 6331:
        raise ValueError("full numerical anchor history required")
    episodes, event_counts, seen, rows = [], Counter(), set(), []
    for group, ah in zip(a.group, a.audit_hash, strict=True):
        root, audit = map(Path, group)
        source, _, sh = report(
            root,
            audit,
            ah,
            "NEURAL_RELAY_CHILD_VELOCITY_CANDIDATE_SAMPLING32_PENDING_OWN_REPLAY"
            if a.safe_continuation
            else "NEURAL_RELAY_CHILD_VELOCITY_SAMPLING32_PENDING_OWN_REPLAY",
            worlds=32,
        )
        blocks = tuple(sorted({r["course"]["context"] for r in source["records"]}))
        if blocks not in ((2, 3, 5, 7), (8, 9, 11, 13)) or blocks in seen:
            raise ValueError("complete unique training blocks 0 and 1 required")
        seen.add(blocks)
        input_pins = source["input_file_hashes"]
        if (
            input_pins.get(str(a.parent)) != a.parent_hash
            or input_pins.get(model) != mh
            or input_pins.get(config) != ch
        ):
            raise ValueError("same actual behavior parent and motor identity required")
        if a.safe_continuation:
            import mujoco

            from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout

            scene = root / "scene.xml"
            scene_hash = digest(scene)
            if scene.is_symlink():
                raise ValueError("exact native physics scene required")
            native_model = mujoco.MjModel.from_xml_path(str(scene))
            layout = NativeVelocityLayout(
                native_model, robot_prefix="b/", ball_body_name="soccer_ball", player_count=2
            )
            ball_v = layout.bv
            pins.append((scene, scene_hash))
        for row in source["records"]:
            seed = row["course"]["arm"]["exploration_seed"]
            if seed is None:
                continue
            if seed not in (92117, 92143, 92171):
                raise ValueError("all fixed stochastic seeds required")
            vp, pp = bind(root, row, "receiver_velocity"), bind(root, row, "physics")
            if vp is None or pp is None:
                raise ValueError("full own neural and physical trajectory required")
            with np.load(vp, allow_pickle=False) as v, np.load(pp, allow_pickle=False) as raw:
                n = len(v["control"])
                if not np.array_equal(v["control"], np.arange(n)) or not 1 <= n <= 300:
                    raise ValueError("exact complete sampled own control sequence required")
                if a.safe_continuation and not v["candidate_actor_selected"].all():
                    raise ValueError("all behavior actions must come from the exact candidate")
                for index in range(n):
                    if not np.array_equal(
                        target29(v["SAM_executed_action"][index], params), raw["target_b"][index]
                    ):
                        raise ValueError("sampled B action must equal actual native target")
                if a.safe_continuation:
                    from rosclaw_soccer.rsi.receiver_safe_continuation_credit import (
                        receiver_safe_continuation_credit,
                    )

                    h = row["handoff"]
                    direction = (
                        np.asarray(h["return_goal"][:2]) - np.asarray(h["kick_origin"][:2])
                        if h is not None
                        else np.array([1.0, 0.0])
                    )
                    direction /= np.linalg.norm(direction)
                    reward, events = receiver_safe_continuation_credit(
                        raw["ball_forces"],
                        raw["qvel"],
                        raw["body_failed"],
                        termination=row["termination"],
                        controls=n,
                        handoff_control=h["control"] if h is not None else None,
                        ball_velocity_index=ball_v,
                        return_direction=direction,
                        excluded_contact_observed=bool(row["result"]["excluded_contact_bodies"]),
                    )
                else:
                    reward, events = full_history_precontact_feedback(
                        raw["ball_forces"][:, :, 1, :], raw["body_failed"], controls=n
                    )
                returns, running = np.empty(n, np.float32), 0.0
                for index in range(n - 1, -1, -1):
                    running = float(reward[index]) + 0.99 * running
                    returns[index] = running
                if (
                    a.safe_continuation
                    and (
                        row["termination"] == "BODY_FAILURE"
                        or row["result"]["excluded_contact_bodies"]
                    )
                    and (returns > 1e-7).any()
                ):
                    raise ValueError("observed failed episodes cannot receive positive returns")
                if a.receiver_potential:
                    from rosclaw_soccer.rsi.receiver_potential_shaping import (
                        receiver_potential_shaping,
                    )

                    ip = bind(root, row, "initial")
                    if ip is None:
                        raise ValueError("actual initial shared state required")
                    with np.load(ip, allow_pickle=False) as initial:
                        pre_q = np.concatenate((initial["qpos"][None], raw["qpos"][: n - 1, -1]))
                        pre_v = np.concatenate((initial["qvel"][None], raw["qvel"][: n - 1, -1]))
                    shape = receiver_potential_shaping(
                        pre_q,
                        pre_v,
                        receiver_q=layout.rq,
                        receiver_v=layout.rv,
                        ball_q=layout.bq,
                        ball_v=layout.bv,
                    )
                    reward = reward + shape
                    returns, running = np.empty(n, np.float32), 0.0
                    for index in range(n - 1, -1, -1):
                        running = float(reward[index]) + 0.99 * running
                        returns[index] = running
                    pins.append((ip, digest(ip)))

                def t(x):
                    return torch.from_numpy(np.asarray(x, np.float32).copy())

                e = dict(
                    observations=t(np.concatenate((v["obs"], v["context14"]), axis=1)),
                    actions=t(v["SAM_executed_action"]),
                    old_means=t(v["action"]),
                    reference=t(v["reference_action"]),
                    old_logp=t(v["SAM_logp"]),
                    returns=t(returns),
                    action_scale=t(scale),
                    sampling_law="box-truncated-ar1",
                    all_actor_actions_actually_executed=True,
                    density_audited=True,
                    full_task_outcome_observed=True,
                    assisted=False,
                    external_teacher=False,
                    authority_rejected=False,
                )
                episodes.append(e)
                for event in events["events"]:
                    if not a.safe_continuation or events["events"][event]["accepted"]:
                        event_counts[event] += 1
                rows.append(
                    dict(
                        context=row["course"]["context"],
                        side=row["course"]["lateral_m"],
                        seed=seed,
                        controls=n,
                        termination=row["termination"],
                        events=events,
                        initial_return=float(returns[0]),
                        full_600_safety_censored=row["termination"]
                        == "NO_RECEIVING_HANDOFF_WITHIN_ORIGINAL_VELOCITY_H300",
                        source_physics_hash=digest(pp),
                        source_velocity_hash=digest(vp),
                    )
                )
            pins.extend(((vp, digest(vp)), (pp, digest(pp))))
        pins.extend(((root / "complete.json", sh), (audit, ah)))
    if len(episodes) != 48 or seen != {(2, 3, 5, 7), (8, 9, 11, 13)}:
        raise ValueError("all 48 declared stochastic B episodes required")
    torch.manual_seed(92801)
    torch.set_num_threads(1)
    actor = PrecontactReceivingMotor(
        VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14),
        torch.from_numpy(scale.copy()),
    ).eval()
    checkpoint = torch.load(a.parent, map_location="cpu", weights_only=True)
    if (
        checkpoint.get("hardware_authorized") is not False
        or checkpoint.get("promotion_authorized") is not False
    ):
        raise ValueError("unpromoted SIM parent required")
    actor.plastic.load_state_dict(checkpoint["actor"], strict=True)
    critic = ReceiverCritic()
    critic.load_state_dict(checkpoint["critic"], strict=True)
    optimizer, losses = torch.optim.Adam(critic.parameters(), lr=0.001), []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = episode_balanced_value_loss(critic, episodes)
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()):
            raise ValueError("finite critic gradients required")
        optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    observations = torch.cat([e["observations"] for e in episodes])
    with torch.no_grad():
        advantage = torch.cat([e["returns"] for e in episodes]) - critic(observations)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
        gated_controls = int((precontact_gate(observations) > 1e-6).sum())
    start = 0
    for ep in episodes:
        end = start + len(ep["observations"])
        ep["advantages"] = advantage[start:end]
        start = end
    update = bounded_actor_update(
        actor,
        critic,
        episodes,
        torch.full((29,), math.log(0.02)),
        anchor_x,
        anchor_r,
        step_size=1e-3,
        max_anchor_delta_rad=0.01,
    )
    if any(digest(path) != h for path, h in (*pins, (Path(model), mh), (Path(config), ch))):
        raise ValueError("pinned training evidence changed during update")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    dest = a.output / "research-candidate.pt"
    torch.save(
        dict(
            actor=actor.plastic.state_dict(),
            critic=critic.state_dict(),
            parent_hash=a.parent_hash,
            anchor_data_hash=a.anchor_data_hash,
            model_identity=identity,
            audit_hashes=a.audit_hash,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
        dest,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status="B_CHILD_VELOCITY_OWN_HISTORY_CANDIDATE",
            source_head=a.expected_head,
            parent_hash=a.parent_hash,
            anchor_data_hash=a.anchor_data_hash,
            model_identity=identity,
            audit_hashes=a.audit_hash,
            checkpoint_hash=digest(dest),
            episodes=len(episodes),
            sampled_controls=len(observations),
            reward_contract=(
                "safe-continuation-v2-potential-v1"
                if a.receiver_potential
                else "safe-continuation-v2"
                if a.safe_continuation
                else "precontact-v1"
            ),
            gated_controls=gated_controls,
            event_counts=dict(event_counts),
            critic_updates=3,
            critic_losses=losses,
            actor_result=update,
            records=rows,
            clean_contact_proven=False,
            physical_retention_proven=False,
            independent_holdout=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )


if __name__ == "__main__":
    main()
