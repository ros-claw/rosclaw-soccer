"""Bounded SIM-only PPO pilot on preflighted incoming-ball initial states.

Robot states and motion references are kick-derived. This is neither a
qualified receiving skill nor an independent match/promotion evaluation.
"""

import argparse
import json
import math
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "incoming-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "incoming-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if a.output.exists() or subprocess.check_output(["git", "status", "--porcelain"]):
        raise ValueError("unused output and frozen source required")
    if subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip() != a.expected_head:
        raise ValueError("exact source head required")
    qual_path, incoming_path = (
        a.qualification_root / "complete.json",
        a.incoming_root / "complete.json",
    )
    if digest(qual_path) != a.qualification_hash or digest(incoming_path) != a.incoming_hash:
        raise ValueError("exact source receipts required")
    qual, incoming = json.loads(qual_path.read_text()), json.loads(incoming_path.read_text())
    if (
        qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL"
        or incoming.get("incoming") is not True
        or incoming.get("window") != 40
        or incoming.get("physics_identical") is not True
        or incoming.get("training") is not False
    ):
        raise ValueError("completed experimental incoming probe required, not skill qualification")
    bank_path = a.incoming_root / "incoming-bank.npz"
    pins = {
        **qual["input_file_hashes"],
        str(qual_path): a.qualification_hash,
        str(incoming_path): a.incoming_hash,
        str(bank_path): incoming["incoming_bank_hash"],
    }
    check_pins(pins)

    def pinned(suffix):
        found = [Path(k) for k in pins if k.endswith(suffix)]
        if len(found) != 1:
            raise ValueError("one exact pinned asset required: " + suffix)
        return found[0]

    import mujoco
    import numpy as np
    import torch

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.mjlab_motion_curriculum import TRAIN_MOTION_NAMES
    from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse
    from rosclaw_soccer.rsi.native_contact_ppo import (
        course_gae,
        make_contact_actor_critic,
        native_ppo_loss,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
    from scripts.rsi_native_course_inputs import load_body_references

    torch.set_num_threads(1)
    torch.manual_seed(7923)
    np.random.seed(7923)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise ValueError("one isolated CUDA device required")
    cfg = pinned("/policy/velocity/v0/params/deploy.yaml")
    onnx = pinned("/policy/velocity/v0/exported/policy.onnx")
    checkpoint = pinned("/ball-aware-update200.pt")
    model_path = pinned("/rsi-body-bound-foot-only-qualification-20261021/compiled_model.mjb")
    rehearsal_path = pinned("/rsi-pelvis-online-ppo-smoke-20261021/native-rehearsal1200.npz")
    params = VelocityPolicyParameters.load(cfg, expected_hash=pins[str(cfg)])
    with np.load(bank_path, allow_pickle=False) as z:
        bank = {k: z[k].copy() for k in z.files}
    with np.load(rehearsal_path, allow_pickle=False) as z:
        old = z["observation"].copy()
    if old.shape != (1200, 98) or old.dtype != np.float32 or not np.isfinite(old).all():
        raise ValueError("complete finite native rehearsal bank required")
    rehearsal = torch.from_numpy(
        np.concatenate((old, np.zeros((1200, 79), np.float32)), axis=1)
    ).cuda()
    plans = []
    for name in TRAIN_MOTION_NAMES:
        path = pinned(
            "/rsi-train-starts-body-bound-index-check-20261021/"
            + name.removesuffix(".npz")
            + "-plan.npz"
        )
        with np.load(path, allow_pickle=False) as z:
            plans.append({k: z[k].copy() for k in z.files})
    model = mujoco.MjModel.from_binary_path(str(model_path))
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    refs = load_body_references(a.reference_root, model, layout, plans)
    actor = VelocityPriorActor.from_onnx(
        onnx, expected_hash=pins[str(onnx)], reference_features=79
    ).cuda()
    parent = torch.load(checkpoint, map_location="cpu", weights_only=True)
    actor.load_state_dict(
        {k: parent["algorithm"]["actor_state_dict"]["mlp." + k] for k in actor.state_dict()},
        strict=True,
    )
    policy = make_contact_actor_critic(actor).cuda()
    original = {k: v.detach().cpu().clone() for k, v in policy.actor.state_dict().items()}
    optimizer = torch.optim.Adam(
        [
            dict(params=policy.actor.parameters(), lr=3e-5),
            dict(params=policy.critic.parameters(), lr=3e-4),
            dict(params=[policy.log_std], lr=1e-5),
        ]
    )

    def make_course():
        return NativeContactCourse(
            model=model,
            params=params,
            bank=bank,
            plans=plans,
            body_references=refs,
            worlds=8,
            lead=10,
            threads=2,
            receive_reward=True,
            receive_window=40,
        )

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="incoming40-ppo-pilot-v1",
            source_head=a.expected_head,
            source_receipts=[a.qualification_hash, a.incoming_hash],
            input_file_hashes=pins,
            seed=7923,
            worlds=8,
            training_episodes=6,
            critic_warmup_episodes=2,
            eval_episode_index=7,
            kick_derived_motion_reference=True,
            cpu_physics=True,
            one_gpu_learner=True,
            independent_fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    env = make_course()
    histories, stats, updates = [], [], 0
    try:
        for episode in range(6):
            env.reset()
            history = []
            while not env.ended.all():
                obs = torch.from_numpy(env.observation).cuda()
                with torch.no_grad():
                    dist = policy.distribution(obs)
                    action = dist.sample()
                    old_logp = dist.log_prob(action).sum(-1)
                    value = policy.value(obs)
                result = env.step(action.cpu().numpy().copy())
                with torch.no_grad():
                    next_value = policy.value(torch.from_numpy(result["observation"]).cuda())
                history.append(
                    {
                        **result,
                        "action": action.cpu().numpy(),
                        "old_logp": old_logp.cpu().numpy(),
                        "old_value": value.cpu().numpy(),
                        "next_value": next_value.cpu().numpy(),
                    }
                )
            raw = {k: np.stack([row[k] for row in history]) for k in history[0]}
            file = a.output / f"train-{episode:02d}.npz"
            np.savez_compressed(file, bank_index=env.index, **raw)
            adv, returns = course_gae(
                raw["reward"].astype(np.float32),
                raw["old_value"],
                raw["next_value"],
                raw["terminated"],
                raw["truncated"],
                raw["valid"],
                maximum_controls=40,
            )
            mask = raw["valid"].reshape(-1)
            values = dict(
                obs=raw["observation_before"],
                action=raw["action"],
                old_logp=raw["old_logp"],
                old_value=raw["old_value"],
                advantage=adv,
                returns=returns,
            )
            data = {
                k: torch.from_numpy(v.reshape((-1, *v.shape[2:]))[mask].copy()).cuda()
                for k, v in values.items()
            }
            data["advantage"] = (data["advantage"] - data["advantage"].mean()) / (
                data["advantage"].std(unbiased=False) + 1e-8
            )
            warmup = episode < 2
            stop = False
            update_metrics = []
            for _ in range(2):
                for ids in torch.tensor_split(torch.randperm(int(mask.sum()), device="cuda"), 4):
                    if not len(ids):
                        continue
                    rehearsal_ids = (torch.arange(256, device="cuda") + updates * 256) % 1200
                    loss, metrics = native_ppo_loss(
                        policy,
                        data["obs"][ids],
                        data["action"][ids],
                        data["old_logp"][ids],
                        data["old_value"][ids],
                        data["advantage"][ids],
                        data["returns"][ids],
                        rehearsal[rehearsal_ids],
                        value_only=warmup,
                    )
                    if not warmup and float(metrics["approximate_kl"]) > 0.02:
                        update_metrics.append(
                            dict(
                                optimizer_step=False,
                                reason="kl_limit",
                                metrics={key: float(value) for key, value in metrics.items()},
                            )
                        )
                        stop = True
                        break
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), 1.0)
                    if not torch.isfinite(loss) or not torch.isfinite(norm):
                        raise ValueError("finite PPO loss and gradients required")
                    optimizer.step()
                    updates += 1
                    update_metrics.append(
                        dict(
                            optimizer_step=True,
                            gradient_norm=float(norm),
                            loss=float(loss.detach()),
                            metrics={key: float(value) for key, value in metrics.items()},
                        )
                    )
                    with torch.no_grad():
                        policy.log_std.clamp_(math.log(0.01), math.log(0.1))
                    if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
                        raise ValueError("finite policy and critic required")
                if stop:
                    break
            stats.append(
                dict(
                    episode=episode,
                    valid=int(mask.sum()),
                    clean_foot=int(raw["credit_first_clean_foot"].sum()),
                    nonfoot=int(raw["credit_first_nonfoot"].sum()),
                    capture=int(raw["credit_sustained_capture"].sum()),
                    failed=int(raw["terminated"].sum()),
                    warmup=warmup,
                    optimizer_updates=updates,
                    update_metrics=update_metrics,
                    raw_hash=digest(file),
                )
            )
            histories.append(env.index.copy())
        actor_changed = any(
            not torch.equal(v.detach().cpu(), original[k])
            for k, v in policy.actor.state_dict().items()
        )
        if not actor_changed or updates <= 16:
            raise ValueError("actual PPO actor update required")
        checkpoint_path = a.output / "candidate.pt"
        torch.save(
            dict(
                policy_state_dict=policy.state_dict(),
                optimizer_state_dict=optimizer.state_dict(),
                seed=7923,
            ),
            checkpoint_path,
        )
        evaluations = {}
        for label, evaluated_actor in (("parent", policy.baseline), ("candidate", policy.actor)):
            exam = make_course()
            try:
                exam.episode = 6
                exam.reset()
                if any(np.intersect1d(exam.index, x).size for x in histories):
                    raise ValueError("development evaluation source indices overlap training")
                rows = []
                while not exam.ended.all():
                    with torch.no_grad():
                        action = (
                            evaluated_actor(torch.from_numpy(exam.observation).cuda())
                            .cpu()
                            .numpy()
                            .copy()
                        )
                    result = exam.step(action)
                    rows.append({**result, "action": action})
                raw = {k: np.stack([row[k] for row in rows]) for k in rows[0]}
                path = a.output / (label + "-development.npz")
                np.savez_compressed(path, bank_index=exam.index, **raw)
                evaluations[label] = dict(
                    raw_hash=digest(path),
                    clean_foot=int(raw["credit_first_clean_foot"].sum()),
                    nonfoot=int(raw["credit_first_nonfoot"].sum()),
                    capture=int(raw["credit_sustained_capture"].sum()),
                    failed=int(raw["terminated"].sum()),
                    native_steps=exam.native_steps,
                )
            finally:
                exam.close()
        check_pins(pins)
        if subprocess.check_output(["git", "status", "--porcelain"]):
            raise ValueError("source changed during training")
        write_once(
            a.output / "complete.json",
            dict(
                status="INCOMING_PPO_PILOT_REQUIRES_INDEPENDENT_EXAM",
                source_head=a.expected_head,
                optimizer_updates=updates,
                actor_changed=actor_changed,
                checkpoint_hash=digest(checkpoint_path),
                train=stats,
                development=evaluations,
                independent_fresh_exam=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
