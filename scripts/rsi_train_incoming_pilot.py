"""Bounded SIM-only PPO pilot on preflighted incoming-ball initial states.

Kicking or MotionDecode kinematic references are explicitly declared. This is
neither a qualified receiving skill nor an independent match evaluation.
"""

import argparse
import copy
import json
import random
import subprocess
import sys
from pathlib import Path


def curriculum_episode(episode: int, mixed: bool) -> tuple[bool, int]:
    """Alternating speeds each traverse every balanced source pool independently."""
    if type(episode) is not int or episode < 0 or type(mixed) is not bool:
        raise ValueError("nonnegative episode and explicit curriculum required")
    return (episode % 2 == 0, episode // 2) if mixed else (False, episode)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("qualification-root", "incoming-root", "reference-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("qualification-hash", "incoming-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--motion-reference-root", type=Path)
    p.add_argument("--motion-reference-hash")
    p.add_argument("--dev-reference-root", type=Path)
    p.add_argument("--dev-reference-hash")
    p.add_argument("--dev-incoming-root", type=Path)
    p.add_argument("--dev-incoming-hash")
    p.add_argument("--train-target-rate-filter", action="store_true")
    p.add_argument("--ball-only-adaptation", action="store_true")
    p.add_argument("--dense-shin-risk", action="store_true")
    p.add_argument("--strict-native-capture", action="store_true")
    p.add_argument("--demonstrations-root", type=Path)
    p.add_argument("--demonstrations-hash")
    p.add_argument("--curriculum-slow-root", type=Path)
    p.add_argument("--curriculum-slow-hash")
    p.add_argument("--resume-checkpoint", type=Path)
    p.add_argument("--resume-hash")
    p.add_argument("--stop-after-episode", type=int)
    p.add_argument("--training-episodes", type=int, choices=(4, 6, 12))
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
    motion = a.motion_reference_root is not None
    dev_args = (
        a.dev_reference_root,
        a.dev_reference_hash,
        a.dev_incoming_root,
        a.dev_incoming_hash,
    )
    expanded = all(dev_args)
    episode_count = a.training_episodes or (12 if expanded else 6)
    if not expanded and episode_count != 6:
        raise ValueError("non-expanded historical protocol requires six episodes")
    if bool(a.resume_checkpoint) != bool(a.resume_hash):
        raise ValueError("resume requires a checkpoint and its exact hash")
    if a.stop_after_episode is not None and not 1 <= a.stop_after_episode < episode_count:
        raise ValueError("stop-after-episode must be a completed prefix, not a full run")
    if any(dev_args) and not expanded:
        raise ValueError("all separate development evidence arguments are required together")
    if a.train_target_rate_filter and not expanded:
        raise ValueError("rate-filter training requires the expanded separate development bank")
    if a.ball_only_adaptation and (not expanded or a.train_target_rate_filter):
        raise ValueError("ball-only adaptation requires separate dev bank and no target filter")
    if a.dense_shin_risk and not expanded:
        raise ValueError("dense shin risk requires the expanded receiving protocol")
    if a.strict_native_capture and not expanded:
        raise ValueError("strict native capture requires the expanded receiving protocol")
    if bool(a.demonstrations_root) != bool(a.demonstrations_hash):
        raise ValueError("demonstration root and exact receipt hash required together")
    if bool(a.curriculum_slow_root) != bool(a.curriculum_slow_hash):
        raise ValueError("slow course root and exact receipt hash required together")
    if a.curriculum_slow_root and (not a.demonstrations_root or not a.strict_native_capture):
        raise ValueError("mixed curriculum requires strict capture and demonstrations")
    if a.demonstrations_root and (
        not expanded or a.ball_only_adaptation or a.train_target_rate_filter or a.dense_shin_risk
    ):
        raise ValueError("imitation initialization requires isolated unfiltered expanded study")
    motion_multi = False
    if motion != bool(a.motion_reference_hash):
        raise ValueError("motion reference requires an exact root and hash together")
    if qual.get("status") != "NATIVE_CONTACT24_COURSE_QUALIFIED_NOT_SKILL":
        raise ValueError("qualified source assets required")
    if motion:
        reference_path = a.motion_reference_root / "complete.json"
        if digest(reference_path) != a.motion_reference_hash:
            raise ValueError("exact MotionDecode reference receipt required")
        reference = json.loads(reference_path.read_text())
        motion_multi = (
            reference.get("status") == "MOTIONDECODE_TRAIN8_KINEMATIC_FK_NOT_TRACKED_SKILL"
        )
        if (
            incoming.get("status") != "MOTIONDECODE_INCOMING_SINGLE_ACTOR_PROBE_NOT_SKILL"
            or incoming.get("reference_hash") != a.motion_reference_hash
            or (
                not motion_multi
                and reference.get("status")
                != "MOTIONDECODE_KINEMATIC_NATIVE_FK_REFERENCE_NOT_TRACKED_SKILL"
            )
            or incoming.get("reference_clip_count") != (8 if motion_multi else 1)
            or incoming.get("launch_frame") != "heading"
            or incoming.get("training") is not False
        ):
            raise ValueError("matched heading incoming and kinematic references required")
        bank_hash = incoming["bank_hash"]
    elif (
        incoming.get("incoming") is not True
        or incoming.get("window") != 40
        or incoming.get("physics_identical") is not True
        or incoming.get("training") is not False
    ):
        raise ValueError("completed experimental incoming probe required, not skill qualification")
    else:
        bank_hash = incoming["incoming_bank_hash"]
    if expanded:
        if (
            not motion_multi
            or incoming.get("lateral_study") is not True
            or incoming.get("receive_window") != 60
        ):
            raise ValueError("expanded course requires 60-control multi-reference lateral source")
        dev_ref_path = a.dev_reference_root / "complete.json"
        dev_inc_path = a.dev_incoming_root / "complete.json"
        if (
            digest(dev_ref_path) != a.dev_reference_hash
            or digest(dev_inc_path) != a.dev_incoming_hash
        ):
            raise ValueError("exact development receipts required")
        dev_reference = json.loads(dev_ref_path.read_text())
        dev_incoming = json.loads(dev_inc_path.read_text())
        if (
            dev_reference.get("status") != "MOTIONDECODE_TRAIN8_KINEMATIC_FK_NOT_TRACKED_SKILL"
            or len(dev_reference.get("records", [])) != 8
            or dev_reference.get("model_hash") != reference["model_hash"]
            or dev_incoming.get("reference_hash") != a.dev_reference_hash
            or dev_incoming.get("status") != "MOTIONDECODE_INCOMING_SINGLE_ACTOR_PROBE_NOT_SKILL"
            or dev_incoming.get("lateral_study") is not True
            or dev_incoming.get("receive_window") != 60
            or dev_incoming.get("course_binding") != incoming.get("course_binding")
            or {x["source_hash"] for x in dev_reference["records"]}
            & {x["source_hash"] for x in reference["records"]}
        ):
            raise ValueError("same course protocol and distinct development source clips required")
    bank_path = a.incoming_root / "incoming-bank.npz"
    pins = {
        **qual["input_file_hashes"],
        str(qual_path): a.qualification_hash,
        str(incoming_path): a.incoming_hash,
        str(bank_path): bank_hash,
    }
    if motion:
        pins[str(reference_path)] = a.motion_reference_hash
        if motion_multi:
            if len(reference["records"]) != 8:
                raise ValueError("complete declared eight clip references required")
            for i, record in enumerate(reference["records"]):
                pins[str(a.motion_reference_root / f"reference-{i:02d}.npz")] = record[
                    "reference_hash"
                ]
        else:
            pins[str(a.motion_reference_root / "reference.npz")] = reference["raw_hash"]
    if expanded:
        pins[str(dev_ref_path)] = a.dev_reference_hash
        pins[str(dev_inc_path)] = a.dev_incoming_hash
        pins[str(a.dev_incoming_root / "incoming-bank.npz")] = dev_incoming["bank_hash"]
        for i, record in enumerate(dev_reference["records"]):
            pins[str(a.dev_reference_root / f"reference-{i:02d}.npz")] = record["reference_hash"]
    if a.curriculum_slow_root:
        slow_path = a.curriculum_slow_root / "complete.json"
        if digest(slow_path) != a.curriculum_slow_hash:
            raise ValueError("exact slow-course receipt required")
        slow_incoming = json.loads(slow_path.read_text())
        if (
            slow_incoming.get("status") != "MOTIONDECODE_INCOMING_SINGLE_ACTOR_PROBE_NOT_SKILL"
            or slow_incoming.get("reference_hash") != a.motion_reference_hash
            or slow_incoming.get("launch_frame") != "heading"
            or slow_incoming.get("lateral_study") is not True
            or slow_incoming.get("receive_window") != 60
            or slow_incoming.get("reference_clip_count") != 8
            or slow_incoming.get("training") is not False
            or slow_incoming.get("course_binding") == incoming.get("course_binding")
        ):
            raise ValueError("distinct matched slow receiving course required")
        pins[str(slow_path)] = a.curriculum_slow_hash
        pins[str(a.curriculum_slow_root / "incoming-bank.npz")] = slow_incoming["bank_hash"]
    demonstration = None
    if a.demonstrations_root:
        demo_path = a.demonstrations_root / "complete.json"
        if digest(demo_path) != a.demonstrations_hash:
            raise ValueError("exact demonstration receipt required")
        demonstration = json.loads(demo_path.read_text())
        if (
            demonstration.get("status") != "NATIVE_RECEIVE_DEMONSTRATIONS_NOT_SKILL"
            or demonstration.get("qualification_hash") != a.qualification_hash
            or demonstration.get("source_speed_mps") != 1.25
            or demonstration.get("selected_worlds") != 2
            or demonstration.get("negative_worlds") != 30
            or incoming.get("fixed_speed_mps") != 3.0
            or any(
                demonstration.get(k) is not False
                for k in ("hardware_authorized", "promotion_authorized", "independent_fresh")
            )
        ):
            raise ValueError("two strict slow demonstrations and fast curriculum required")
        source_paths = [
            Path(path)
            for path, value in demonstration["input_hashes"].items()
            if value == demonstration["source_hash"] and path.endswith("/complete.json")
        ]
        if len(source_paths) != 2 or any(
            json.loads(path.read_text()).get("reference_hash") != a.motion_reference_hash
            for path in source_paths
        ):
            raise ValueError("same training references required for demonstrated motion")
        extra = {
            **demonstration["input_hashes"],
            str(demo_path): a.demonstrations_hash,
            str(a.demonstrations_root / "demonstrations.npz"): demonstration["dataset_hash"],
        }
        if any(key in pins and pins[key] != value for key, value in extra.items()):
            raise ValueError("demonstration input commitments disagree")
        pins.update(extra)
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
        enable_ball_only_actor_gradient,
        guarded_ppo_optimizer_step,
        make_contact_actor_critic,
        native_ppo_loss,
    )
    from rosclaw_soccer.rsi.native_receive_imitation import fit_receive_demonstrations
    from rosclaw_soccer.rsi.native_target_rate_filter import limit_target_rate
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
    from scripts.rsi_incoming_checkpoint import restore_checkpoint, save_checkpoint
    from scripts.rsi_native_course_inputs import load_body_references

    torch.set_num_threads(1)
    torch.manual_seed(7923)
    np.random.seed(7923)
    random.seed(7923)
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
    model = mujoco.MjModel.from_binary_path(str(model_path))
    layout = NativeVelocityLayout(model, robot_prefix="robot/", ball_body_name="ball/ball")
    if a.curriculum_slow_root:
        with np.load(a.curriculum_slow_root / "incoming-bank.npz", allow_pickle=False) as z:
            slow_bank = {k: z[k].copy() for k in z.files}
        for candidate, speed in ((slow_bank, 1.25), (bank, 3.0)):
            velocity = candidate["qvel"][:, layout.bv : layout.bv + 3]
            if not np.allclose(np.linalg.norm(velocity, axis=1), speed, atol=1e-6, rtol=0):
                raise ValueError("all native initial ball velocities must match curriculum")
    if motion:
        if pins[str(model_path)] != reference["model_hash"]:
            raise ValueError("actual reference model must match")
        plans, refs = [], []
        for i in range(8):
            path = a.motion_reference_root / (
                f"reference-{i:02d}.npz" if motion_multi else "reference.npz"
            )
            with np.load(path, allow_pickle=False) as z:
                plans.append(
                    {k.removeprefix("plan_"): z[k].copy() for k in z.files if k.startswith("plan_")}
                )
                refs.append(
                    {k.removeprefix("body_"): z[k].copy() for k in z.files if k.startswith("body_")}
                )
    else:
        plans = []
        for name in TRAIN_MOTION_NAMES:
            path = pinned(
                "/rsi-train-starts-body-bound-index-check-20261021/"
                + name.removesuffix(".npz")
                + "-plan.npz"
            )
            with np.load(path, allow_pickle=False) as z:
                plans.append({k: z[k].copy() for k in z.files})
        refs = load_body_references(a.reference_root, model, layout, plans)
    if expanded:
        with np.load(a.dev_incoming_root / "incoming-bank.npz", allow_pickle=False) as z:
            dev_bank = {k: z[k].copy() for k in z.files}
        dev_plans, dev_refs = [], []
        for i in range(8):
            with np.load(a.dev_reference_root / f"reference-{i:02d}.npz", allow_pickle=False) as z:
                dev_plans.append(
                    {k.removeprefix("plan_"): z[k].copy() for k in z.files if k.startswith("plan_")}
                )
                dev_refs.append(
                    {k.removeprefix("body_"): z[k].copy() for k in z.files if k.startswith("body_")}
                )
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
    imitation_metrics, imitation_actor = None, None
    if demonstration is not None:
        with np.load(a.demonstrations_root / "demonstrations.npz", allow_pickle=False) as z:
            selected = z["selected"]
            if (
                selected.shape != (32,)
                or selected.dtype != np.bool_
                or int(selected.sum()) != 2
                or not z["valid"][:, selected].all()
                or z["observation"].shape != (60, 32, 177)
                or z["actual_action"].shape != (60, 32, 29)
            ):
                raise ValueError("complete qualified selected teacher sequences required")
            teacher_obs = torch.from_numpy(
                z["observation"][:, selected].reshape(-1, 177).copy()
            ).cuda()
            teacher_action = torch.from_numpy(
                z["actual_action"][:, selected].reshape(-1, 29).copy()
            ).cuda()
        imitation_metrics = fit_receive_demonstrations(
            policy, teacher_obs, teacher_action, rehearsal
        )
        imitation_actor = copy.deepcopy(policy.actor).eval()
    if a.ball_only_adaptation:
        ball_weight = policy.actor.mlp[0].weight
        ball_gradient_handle = enable_ball_only_actor_gradient(policy.actor)
        actor_groups = [
            dict(params=[ball_weight], lr=3e-5),
            dict(params=[p for p in policy.actor.parameters() if p is not ball_weight], lr=0.0),
        ]
    else:
        ball_gradient_handle = None
        actor_groups = [dict(params=policy.actor.parameters(), lr=3e-6)]
    optimizer = torch.optim.Adam(
        [
            *actor_groups,
            dict(params=policy.critic.parameters(), lr=3e-4),
            dict(params=[policy.log_std], lr=0.0 if a.ball_only_adaptation else 1e-5),
        ]
    )

    def make_course(*, development=False, slow=False):
        return NativeContactCourse(
            model=model,
            params=params,
            bank=dev_bank if development and expanded else slow_bank if slow else bank,
            plans=dev_plans if development and expanded else plans,
            body_references=dev_refs if development and expanded else refs,
            worlds=32 if expanded else 8,
            lead=10,
            threads=8 if expanded else 2,
            receive_reward=True,
            receive_window=60 if expanded else 40,
            dense_shin_risk=a.dense_shin_risk and not development,
            strict_native_capture=a.strict_native_capture,
        )

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="incoming60-diverse-ppo-pilot-v1" if expanded else "incoming40-ppo-pilot-v1",
            source_head=a.expected_head,
            source_receipts=[a.qualification_hash, a.incoming_hash],
            input_file_hashes=pins,
            seed=7923,
            worlds=32 if expanded else 8,
            training_episodes=episode_count,
            critic_warmup_episodes=2,
            eval_episode_index=0 if expanded else 7,
            distinct_dev_clip_files=expanded,
            training_course_repeated_states=expanded,
            kick_derived_motion_reference=not motion,
            motiondecode_reference=motion,
            motion_reference_hash=a.motion_reference_hash,
            motion_reference_clip_count=(8 if motion_multi else 1) if motion else 0,
            overlapping_reference_windows=motion,
            reference_group_independence_unproven=motion,
            planar_capture_distance=True,
            train_target_rate_filter=a.train_target_rate_filter,
            latent_action_log_probability=True,
            executed_action_in_policy_history=True,
            parent_filter_ablation_in_development=a.train_target_rate_filter,
            actor_learning_rate=3e-5 if a.ball_only_adaptation else 3e-6,
            ball_only_adaptation=a.ball_only_adaptation,
            dense_shin_risk=a.dense_shin_risk,
            strict_native_capture=a.strict_native_capture,
            demonstrations_hash=a.demonstrations_hash,
            curriculum_slow_hash=a.curriculum_slow_hash,
            curriculum_schedule="alternate-slow-fast" if a.curriculum_slow_root else None,
            imitation=imitation_metrics,
            dense_shin_risk_weight=0.1 if a.dense_shin_risk else None,
            ball_trainable_columns=[168, 169, 170, 171, 172, 173]
            if a.ball_only_adaptation
            else None,
            post_update_rollout_kl_limit=0.02,
            rollback_policy_and_optimizer_on_excess_kl=True,
            cpu_physics=True,
            one_gpu_learner=True,
            independent_fresh_exam=False,
            hardware_authorized=False,
            promotion_authorized=False,
            resume_checkpoint_hash=a.resume_hash,
            stop_after_episode=a.stop_after_episode,
        ),
    )
    fast_env = make_course()
    slow_env = make_course(slow=True) if a.curriculum_slow_root else None
    histories, stats, updates = [], [], 0
    commitment = dict(
        source_head=a.expected_head,
        input_file_hashes=pins,
        train_target_rate_filter=a.train_target_rate_filter,
        ball_only_adaptation=a.ball_only_adaptation,
        dense_shin_risk=a.dense_shin_risk,
        strict_native_capture=a.strict_native_capture,
        demonstrations_hash=a.demonstrations_hash,
        expanded=expanded,
        training_episodes=episode_count,
        seed=7923,
        torch_version=str(torch.__version__),
        cuda_version=str(torch.version.cuda),
        device_name=torch.cuda.get_device_name(0),
        mujoco_version=mujoco.__version__,
        numpy_version=np.__version__,
        python_version=sys.version,
    )
    if a.curriculum_slow_root:
        commitment["curriculum_slow_hash"] = a.curriculum_slow_hash
    try:
        first_episode = 0
        if a.resume_checkpoint:
            state = restore_checkpoint(
                a.resume_checkpoint,
                a.resume_hash,
                policy=policy,
                optimizer=optimizer,
                commitment=commitment,
            )
            first_episode = state["next_episode"]
            if first_episode >= episode_count or (
                a.stop_after_episode is not None and a.stop_after_episode <= first_episode
            ):
                raise ValueError("resume must advance within declared episode count")
            stats, updates = state["stats"], state["updates"]
            histories = [np.asarray(x, dtype=np.int64) for x in state["histories"]]
        for episode in range(first_episode, episode_count):
            use_slow, bank_episode = curriculum_episode(episode, slow_env is not None)
            env = slow_env if use_slow else fast_env
            env.episode = bank_episode - 1
            env.reset()
            history = []
            while not env.ended.all():
                obs = torch.from_numpy(env.observation).cuda()
                with torch.no_grad():
                    dist = policy.distribution(obs)
                    action = dist.sample()
                    old_logp = dist.log_prob(action).sum(-1)
                    value = policy.value(obs)
                applied_action = action.cpu().numpy().copy()
                if a.train_target_rate_filter:
                    for i in np.flatnonzero(~env.ended):
                        applied_action[i], _ = limit_target_rate(
                            action=applied_action[i],
                            previous_action=env.last_action[i],
                            params=params,
                        )
                result = env.step(applied_action)
                with torch.no_grad():
                    next_value = policy.value(torch.from_numpy(result["observation"]).cuda())
                history.append(
                    {
                        **result,
                        "action": action.cpu().numpy(),
                        "executed_action": applied_action,
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
                maximum_controls=60 if expanded else 40,
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
                    if warmup:
                        optimizer.step()
                        accepted, post_kl = True, None
                    else:
                        accepted, post_kl = guarded_ppo_optimizer_step(
                            policy, optimizer, data["obs"], data["action"], data["old_logp"]
                        )
                    if not accepted:
                        update_metrics.append(
                            dict(
                                optimizer_step=False,
                                reason="post_step_kl_rollback",
                                post_update_kl=post_kl,
                                gradient_norm=float(norm),
                                metrics={key: float(value) for key, value in metrics.items()},
                            )
                        )
                        stop = True
                        break
                    updates += 1
                    update_metrics.append(
                        dict(
                            optimizer_step=True,
                            gradient_norm=float(norm),
                            post_update_kl=post_kl,
                            loss=float(loss.detach()),
                            metrics={key: float(value) for key, value in metrics.items()},
                        )
                    )
                    if any(not torch.isfinite(v).all() for v in policy.state_dict().values()):
                        raise ValueError("finite policy and critic required")
                if stop:
                    break
            stats.append(
                dict(
                    episode=episode,
                    curriculum_bank="slow" if use_slow else "fast",
                    curriculum_bank_episode=bank_episode,
                    valid=int(mask.sum()),
                    clean_foot=int(raw["credit_first_clean_foot"].sum()),
                    nonfoot=int(raw["credit_first_nonfoot"].sum()),
                    capture=int(raw["credit_sustained_capture"].sum()),
                    failed=int(raw["terminated"].sum()),
                    warmup=warmup,
                    optimizer_updates=updates,
                    update_metrics=update_metrics,
                    shin_risk_sum=float(raw["shin_risk"].sum()) if a.dense_shin_risk else None,
                    raw_hash=digest(file),
                    raw_path=str(file.resolve()),
                )
            )
            histories.append(env.index.copy())
            check_pins(pins)
            checkpoint_file = a.output / f"episode-{episode:02d}.pt"
            checkpoint_hash = save_checkpoint(
                checkpoint_file,
                policy=policy,
                optimizer=optimizer,
                commitment=commitment,
                episode=episode,
                updates=updates,
                stats=stats,
                histories=histories,
            )
            if a.stop_after_episode == episode + 1:
                write_once(
                    a.output / "partial.json",
                    dict(
                        status="INCOMING_PPO_EPISODE_BOUNDARY_STOP_NOT_SKILL",
                        checkpoint_path=str(checkpoint_file.resolve()),
                        checkpoint_hash=checkpoint_hash,
                        completed_episodes=episode + 1,
                        source_head=a.expected_head,
                        promotion_authorized=False,
                        hardware_authorized=False,
                    ),
                )
                return
        actor_changed = any(
            not torch.equal(v.detach().cpu(), original[k])
            for k, v in policy.actor.state_dict().items()
        )
        if not actor_changed or updates <= 16:
            raise ValueError("actual PPO actor update required")
        if a.ball_only_adaptation:
            for key, value in policy.actor.state_dict().items():
                current = value.detach().cpu()
                expected = original[key]
                if key == "mlp.0.weight":
                    frozen = torch.ones(current.shape[1], dtype=torch.bool)
                    frozen[168:174] = False
                    if not torch.equal(current[:, frozen], expected[:, frozen]):
                        raise ValueError("non-ball actor columns changed")
                elif not torch.equal(current, expected):
                    raise ValueError("non-ball actor parameters changed")
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
        evaluation_arms = [("parent", policy.baseline, False)]
        imitation_hash = None
        if imitation_actor is not None:
            imitation_path = a.output / "imitation.pt"
            torch.save({"actor_state_dict": imitation_actor.state_dict()}, imitation_path)
            imitation_hash = digest(imitation_path)
            evaluation_arms.append(("imitation", imitation_actor, False))
        if a.train_target_rate_filter:
            evaluation_arms.append(("parent-filter", policy.baseline, True))
        evaluation_arms.append(("candidate", policy.actor, a.train_target_rate_filter))
        for label, evaluated_actor, use_filter in evaluation_arms:
            exam = make_course(development=expanded)
            try:
                exam.episode = -1 if expanded else 6
                exam.reset()
                if not expanded and any(np.intersect1d(exam.index, x).size for x in histories):
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
                    nominal_action = action.copy()
                    if use_filter:
                        for i in np.flatnonzero(~exam.ended):
                            action[i], _ = limit_target_rate(
                                action=action[i], previous_action=exam.last_action[i], params=params
                            )
                    result = exam.step(action)
                    rows.append({**result, "action": action, "nominal_action": nominal_action})
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
                imitation=imitation_metrics,
                imitation_hash=imitation_hash,
                checkpoint_hash=digest(checkpoint_path),
                train=stats,
                development=evaluations,
                independent_fresh_exam=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    finally:
        if ball_gradient_handle is not None:
            ball_gradient_handle.remove()
        fast_env.close()
        if slow_env is not None:
            slow_env.close()


if __name__ == "__main__":
    main()
