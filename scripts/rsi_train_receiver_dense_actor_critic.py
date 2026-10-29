"""Bounded offline SIM actor-critic on independently audited physical feedback."""

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.receiver_actor_critic import ReceiverCritic, episode_balanced_value_loss
from rosclaw_soccer.rsi.receiver_bounded_update import bounded_actor_update
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
from scripts.rsi_atomic_artifacts import write_once
from scripts.rsi_diagnose_pass_contact_timing import digest
from scripts.rsi_native_fresh_common import clean


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("dataset", "dense", "dense-audit", "anchor-admission", "anchor-dataset", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in (
        "dataset-hash",
        "dense-hash",
        "dense-audit-hash",
        "anchor-admission-hash",
        "anchor-dataset-hash",
        "expected-head",
    ):
        p.add_argument("--" + name, required=True)
    p.add_argument("--seed", type=int, choices=(92401, 92402, 92403), required=True)
    p.add_argument("--receiver-motion-only", action="store_true")
    p.add_argument("--functional-proxy", action="store_true")
    p.add_argument("--precontact-proxy", action="store_true")
    p.add_argument("--functional-failure-timing", type=Path)
    p.add_argument("--functional-failure-timing-hash")
    a = p.parse_args()
    if a.functional_proxy and a.receiver_motion_only:
        p.error("functional proxy and motion-only feedback are distinct protocols")
    if a.precontact_proxy and (not a.functional_proxy or not a.functional_failure_timing):
        p.error("precontact research requires functional mode and exact failure timing")
    if bool(a.functional_failure_timing) != bool(a.functional_failure_timing_hash) or (
        a.functional_failure_timing and not a.functional_proxy
    ):
        p.error("timed failure requires functional proxy and a pinned timing receipt")
    clean(a.expected_head)
    paths = (
        (a.dataset / "complete.json", a.dataset_hash),
        (a.dense / "complete.json", a.dense_hash),
        (a.dense_audit, a.dense_audit_hash),
        (a.anchor_admission, a.anchor_admission_hash),
        (a.anchor_dataset, a.anchor_dataset_hash),
    )
    if a.functional_failure_timing:
        paths += ((a.functional_failure_timing, a.functional_failure_timing_hash),)
    if (
        a.output.exists()
        or a.output.resolve().is_relative_to(Path.cwd())
        or any(p.is_symlink() or digest(p) != h for p, h in paths)
    ):
        raise ValueError("pinned audits and unused external output required")
    data, dense, audit, admission, old_data = [json.loads(p.read_text()) for p, _ in paths[:5]]
    timing_by_key = {}
    if a.functional_failure_timing:
        timing = json.loads(a.functional_failure_timing.read_text())
        timing_rows = timing.get("records", [])
        if (
            timing.get("status") != "RECEIVER_FAILURE_TIMELINE_REPLAY_BOUND_NOT_TRAINED"
            or timing.get("dataset_hash") != a.dataset_hash
            or len(timing_rows) != 72
            or timing.get("failure_count")
            != sum(r.get("failure_control") is not None for r in timing_rows)
            or any(
                timing.get(k) is not False
                for k in (
                    "actor_training",
                    "independent_fresh",
                    "promotion_authorized",
                    "hardware_authorized",
                    "goal_complete",
                )
            )
        ):
            raise ValueError("exact unpromoted failure timing receipt required")
        timing_by_key = {(r["context"], r["side"], r["seed"]): r for r in timing_rows}
        if len(timing_by_key) != 72:
            raise ValueError("unique complete timing rows required")
    anchors = a.anchor_admission.parent / "healthy-anchors.npz"
    if (
        data.get("status") != "RECEIVER_ONPOLICY_PHYSICAL_EPISODES_NOT_TRAINED"
        or data.get("reward_profile") != "center-strict-full-history-v1"
        or data.get("episodes") != 72
        or dense.get("status")
        != (
            "RECEIVER_MOTION_PHYSICAL_FEEDBACK_EXPORTED_NOT_TRAINED"
            if a.receiver_motion_only
            else "RECEIVER_DENSE_PHYSICAL_FEEDBACK_EXPORTED_NOT_TRAINED"
        )
        or dense.get("dataset_hash") != a.dataset_hash
        or audit.get("status")
        != (
            "RECEIVER_MOTION_FEEDBACK_INDEPENDENTLY_RECOMPUTED_NOT_TRAINED"
            if a.receiver_motion_only
            else "RECEIVER_DENSE_FEEDBACK_INDEPENDENTLY_RECOMPUTED_NOT_TRAINED"
        )
        or (
            a.receiver_motion_only
            and (
                dense.get("reward_profile") != "receiver-motion-kinematic-v2"
                or audit.get("reward_profile") != "receiver-motion-kinematic-v2"
            )
        )
        or audit.get("dense_hash") != a.dense_hash
        or audit.get("original_hash") != a.dataset_hash
        or audit.get("episodes") != 72
        or audit.get("controls") != data.get("sampled_controls")
        or audit.get("maximum_reward_error", 1) > 1e-6
        or admission.get("status") != "RECEIVER_RESEARCH_TRAINING_DATA_ADMITTED_NOT_TRAINED"
        or admission.get("dataset_hash") != a.anchor_dataset_hash
        or admission.get("healthy_anchor_controls") != 5201
        or old_data.get("model_identity") != data.get("model_identity")
        or anchors.is_symlink()
        or digest(anchors) != admission.get("healthy_anchor_hash")
    ):
        raise ValueError("exact audited failure data and matching old healthy anchors required")
    if any(
        x.get(k) is not False
        for x in (data, dense, audit, admission)
        for k in ("promotion_authorized", "hardware_authorized")
    ):
        raise ValueError("unpromoted simulation-only data required")
    torch.manual_seed(a.seed)
    torch.set_num_threads(2)
    model, mh, config, ch = data["model_identity"]
    params = VelocityPolicyParameters.load(Path(config), expected_hash=ch)
    prior = VelocityPriorActor.from_onnx(Path(model), expected_hash=mh, reference_features=14)
    scale = torch.tensor(params.action_scale, dtype=torch.float32)
    if a.precontact_proxy:
        from rosclaw_soccer.rsi.precontact_receiving_motor import PrecontactReceivingMotor

        actor = PrecontactReceivingMotor(prior, scale).eval()
    else:
        actor = AnchoredReceivingMotor(prior, scale).eval()
    critic = ReceiverCritic()
    with np.load(anchors, allow_pickle=False) as f:
        anchor_x, anchor_r = (
            torch.from_numpy(f["inputs"].copy()),
            torch.from_numpy(f["reference"].copy()),
        )
    episodes, frozen, proxy_windows = [], [], []
    for original, shaped in zip(data["records"], dense["records"], strict=True):
        op, sp = a.dataset / original["path"], a.dense / shaped["path"]
        if (
            shaped["original_episode_hash"] != original["hash"]
            or digest(op) != original["hash"]
            or digest(sp) != shaped["hash"]
            or (original["context"], original["side"], original["seed"])
            != (shaped["context"], shaped["side"], shaped["seed"])
        ):
            raise ValueError("exact aligned sampled trajectories required")
        frozen.extend(((op, original["hash"]), (sp, shaped["hash"])))
        with np.load(op, allow_pickle=False) as e, np.load(sp, allow_pickle=False) as d:
            n = original["controls"]
            if (
                len(e["inputs"]) != n
                or d["dense_rewards"].shape != (n,)
                or not np.array_equal(e["action_scale"], scale.numpy())
                or not np.array_equal(e["rewards"], d["strict_terminal_rewards"])
            ):
                raise ValueError("actual actions and strict rewards must remain aligned")
            if a.functional_proxy:
                from rosclaw_soccer.rsi.functional_receive_proxy import (
                    functional_receive_proxy,
                    timed_failure_penalty,
                )

                if a.precontact_proxy:
                    from rosclaw_soccer.rsi.precontact_feedback import precontact_feedback

                    reward, contact_event = precontact_feedback(d["ball_forces"])
                    window = None
                else:
                    reward, window = functional_receive_proxy(
                        d["receiver_xy"],
                        d["ball_xy"],
                        d["ball_forces"],
                        d["body_failed"],
                        eventual_failure=original["actual"]["body_failure"]
                        if not timing_by_key
                        else False,
                    )
                    contact_event = None
                timing_credit = None
                if timing_by_key:
                    row = timing_by_key[(original["context"], original["side"], original["seed"])]
                    failure_control = row["failure_control"]
                    if (
                        row["sampled_controls"] != n
                        or row["source_block"] != original["source_block"]
                        or row["source_trial"] != original["source_trial"]
                        or row["source_physics_hash"] != original["source_physics_hash"]
                        or (failure_control is not None) != original["actual"]["body_failure"]
                        or (
                            failure_control is not None
                            and (
                                type(failure_control) is not int
                                or failure_control >= row["physical_controls"]
                            )
                        )
                        or ("A" in row["failed_bodies"]) != original["actual"]["A_failed"]
                        or ("B" in row["failed_bodies"]) != original["actual"]["B_failed"]
                    ):
                        raise ValueError("exact full native failure timing required")
                    reward, timing_credit = timed_failure_penalty(
                        reward, failure_control=failure_control, gamma=0.99
                    )
                proxy_windows.append(
                    dict(
                        context=original["context"],
                        side=original["side"],
                        seed=original["seed"],
                        first_window_control=window,
                        contact_event=contact_event,
                        eventual_failure=original["actual"]["body_failure"],
                        timing_credit=timing_credit,
                    )
                )
            else:
                reward = e["rewards"].astype(np.float32) + np.float32(0.2) * d["dense_rewards"]
            ret = np.empty(n, np.float32)
            running = 0.0
            for i in range(n - 1, -1, -1):
                running = float(reward[i]) + 0.99 * running
                ret[i] = running

            def t(name):
                return torch.from_numpy(e[name].copy())

            episodes.append(
                dict(
                    observations=t("inputs"),
                    actions=t("sampled_action"),
                    old_means=t("old_mean"),
                    reference=t("reference"),
                    old_logp=t("old_logp"),
                    action_scale=scale,
                    returns=torch.from_numpy(ret),
                    sampling_law="box-truncated-ar1",
                    all_actor_actions_actually_executed=True,
                    density_audited=True,
                    full_task_outcome_observed=True,
                    assisted=False,
                    external_teacher=False,
                    authority_rejected=False,
                )
            )
    if (
        len(episodes) != 72
        or sum(len(e["observations"]) for e in episodes) != 18903
        or len(anchor_x) != 5201
    ):
        raise ValueError("complete audited episode and retention anchor coverage required")
    observations = torch.cat([e["observations"] for e in episodes])
    targets = torch.cat([e["returns"] for e in episodes])
    optimizer, losses = torch.optim.Adam(critic.parameters(), lr=0.001), []
    for _ in range(3):
        optimizer.zero_grad(set_to_none=True)
        loss = episode_balanced_value_loss(critic, episodes)
        if not torch.isfinite(loss):
            raise ValueError("nonfinite critic loss")
        loss.backward()
        if any(p.grad is None or not torch.isfinite(p.grad).all() for p in critic.parameters()):
            raise ValueError("nonfinite critic gradient")
        optimizer.step()
        losses.append(float(loss.detach()))
    critic.eval().requires_grad_(False)
    with torch.no_grad():
        advantage = targets - critic(observations)
        advantage = (advantage - advantage.mean()) / (advantage.std(unbiased=False) + 1e-6)
    start = 0
    for ep in episodes:
        end = start + len(ep["observations"])
        ep["advantages"] = advantage[start:end]
        start = end
    result = bounded_actor_update(
        actor,
        critic,
        episodes,
        torch.full((29,), math.log(0.02)),
        anchor_x,
        anchor_r,
        step_size=1e-4,
    )
    if any(
        digest(path) != h
        for path, h in (
            *paths,
            *frozen,
            (anchors, admission["healthy_anchor_hash"]),
            (Path(model), mh),
            (Path(config), ch),
        )
    ):
        raise ValueError("inputs changed during exploratory optimizer")
    clean(a.expected_head)
    a.output.mkdir(parents=True)
    checkpoint = a.output / "research-candidate.pt"
    torch.save(
        dict(
            actor=actor.plastic.state_dict(),
            critic=critic.state_dict(),
            seed=a.seed,
            dataset_hash=a.dataset_hash,
            dense_hash=a.dense_hash,
            dense_audit_hash=a.dense_audit_hash,
            feedback_profile="precontact-force-timed-v1"
            if a.precontact_proxy
            else "functional-receive-timed-proxy-v2"
            if timing_by_key
            else "functional-receive-proxy-v1"
            if a.functional_proxy
            else "receiver-motion-kinematic-v2"
            if a.receiver_motion_only
            else "combined-distance-v1",
            physical_retention_proven=False,
            promotion_authorized=False,
            hardware_authorized=False,
            failure_timing_hash=a.functional_failure_timing_hash,
        ),
        checkpoint,
    )
    write_once(
        a.output / "complete.json",
        dict(
            status=(
                "PRECONTACT_RECEIVER_OFFLINE_CANDIDATE"
                if a.precontact_proxy
                else "FUNCTIONAL_TIMED_RECEIVER_OFFLINE_CANDIDATE"
                if timing_by_key
                else "FUNCTIONAL_RECEIVER_OFFLINE_CANDIDATE"
                if a.functional_proxy
                else "MOTION_RECEIVER_OFFLINE_CANDIDATE"
                if a.receiver_motion_only
                else "DENSE_RECEIVER_OFFLINE_CANDIDATE"
            )
            if result["accepted"]
            else (
                "PRECONTACT_RECEIVER_OFFLINE_REJECTED"
                if a.precontact_proxy
                else "FUNCTIONAL_TIMED_RECEIVER_OFFLINE_REJECTED"
                if timing_by_key
                else "FUNCTIONAL_RECEIVER_OFFLINE_REJECTED"
                if a.functional_proxy
                else "MOTION_RECEIVER_OFFLINE_REJECTED"
                if a.receiver_motion_only
                else "DENSE_RECEIVER_OFFLINE_REJECTED"
            ),
            source_head=a.expected_head,
            dataset_hash=a.dataset_hash,
            dense_hash=a.dense_hash,
            dense_audit_hash=a.dense_audit_hash,
            feedback_profile="precontact-force-timed-v1"
            if a.precontact_proxy
            else "functional-receive-timed-proxy-v2"
            if timing_by_key
            else "functional-receive-proxy-v1"
            if a.functional_proxy
            else "receiver-motion-kinematic-v2"
            if a.receiver_motion_only
            else "combined-distance-v1",
            checkpoint_hash=digest(checkpoint),
            seed=a.seed,
            sampled_controls=18903,
            episodes=72,
            strict_positive_conditions=0,
            critic_updates=3,
            critic_losses=losses,
            dense_weight=0.0 if a.functional_proxy else 0.2,
            proxy_window_rows=proxy_windows if a.functional_proxy else None,
            proxy_is_not_possession=a.functional_proxy,
            contact_feedback_is_not_legality=a.precontact_proxy,
            failure_timing_hash=a.functional_failure_timing_hash,
            gamma=0.99,
            **result,
            actual_new_physics=False,
            online_rl=False,
            fresh_holdout=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
