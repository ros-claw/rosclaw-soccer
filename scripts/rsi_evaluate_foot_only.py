"""Paired public DEV for new foot-only intent; no learning or skill promotion."""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("learner-root", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("learner-hash", "expected-head"):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("exact frozen source and new output required")
    receipt = a.learner_root / "complete.json"
    if digest(receipt) != a.learner_hash:
        raise ValueError("actual completed foot-only learner required")
    learner = json.loads(receipt.read_text())
    if (
        learner.get("status") != "FOOT_ONLY_PPO_PILOT_NOT_SKILL_QUALIFIED"
        or learner.get("optimizer_counts") != dict(attempted=336, completed=336)
        or learner.get("gpu_world_steps") != 122880
    ):
        raise ValueError("actual new foot-only pilot, not legacy qualification required")
    pins = dict(learner["input_file_hashes"])
    pins[str(receipt)] = a.learner_hash
    checkpoint = a.learner_root / learner["checkpoint_file"]
    pins[str(checkpoint)] = learner["checkpoint_hash"]
    training_source = next(
        Path(k).parent.parent for k in pins if k.endswith("/scripts/rsi_train_foot_only.py")
    )
    for rel in (
        "scripts/rsi_mjlab_foot_only_backend.py",
        "scripts/rsi_mjlab_ball_aware_backend.py",
        "src/rosclaw_soccer/rsi/identity_kick_credit.py",
        "scripts/rsi_mjwarp_addressed_contact_force.py",
    ):
        if digest(Path.cwd() / rel) != pins.get(str(training_source / rel)):
            raise ValueError("same qualified contact and reward implementation required")
    for root in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        pins.update({str(f): digest(f) for f in root.rglob("*.py")})
    check_pins(pins)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            input_file_hashes=pins,
            source_head=a.expected_head,
            worlds=32,
            horizon=300,
            seeds=[1501, 1502],
            arms=["initial", "final"],
            sources=[7, 10],
            first_episode_only=True,
            known_public_dev_not_blind_fresh=True,
            foot_threshold_n=10.0,
            nonfoot_threshold_n=1.0,
            maximum_gpu_world_steps=384000,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    env = None
    rows = []
    summaries = []
    steps = nn = 0
    try:
        import mujoco
        import numpy as np
        import torch

        from scripts.rsi_mjwarp_addressed_contact_force import install

        install()
        sys.path.insert(0, str(a.reference_root))
        from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
        from rosclaw_soccer.rsi.mjlab_motion_curriculum import DEVELOPMENT_MOTION_NAMES
        from rosclaw_soccer.rsi.mjlab_motion_evaluation import MotionFirstEpisode
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor
        from scripts.rsi_foot_only_calibration import body_bindings
        from scripts.rsi_mjlab_foot_only_backend import (
            FootOnlyMotionEnv,
            make_foot_only_training_cfg,
        )

        torch.set_num_threads(1)
        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        loaded = torch.load(checkpoint, map_location="cpu", weights_only=True)
        if loaded.get("foot_only_intent") != "foot-only-v1" or loaded.get("iteration") != 16:
            raise ValueError("explicit hash-bound new-intent checkpoint required")
        onnx = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/exported/policy.onnx"))
        config = next(Path(k) for k in pins if k.endswith("/policy/velocity/v0/params/deploy.yaml"))
        params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
        actor = (
            VelocityPriorActor.from_onnx(onnx, expected_hash=pins[str(onnx)], reference_features=79)
            .cuda()
            .eval()
        )
        for si, source in enumerate(DEVELOPMENT_MOTION_NAMES):
            paired = None
            for arm in ("initial", "final"):
                state = (
                    loaded["initial_actor_state_dict"]
                    if arm == "initial"
                    else loaded["algorithm"]["actor_state_dict"]
                )
                actor.load_state_dict(
                    {k: state["mlp." + k] for k in actor.state_dict()}, strict=True
                )
                cfg = make_foot_only_training_cfg(
                    parameters=params,
                    command_mode="reference_position_velocity",
                    command_point="pelvis",
                    motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
                    motion_glob=source,
                    num_envs=32,
                    seed=1501 + si,
                    balanced_initial_sources=False,
                )
                cfg.commands["motion"].sampling_mode = "uniform"
                cfg.sim.nan_guard.enabled = True
                env = FootOnlyMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
                obs, _ = env.reset(seed=1501 + si)
                model = env.sim.mj_model
                layout = NativeVelocityLayout(
                    model, robot_prefix="robot/", ball_body_name="ball/ball"
                )
                body_bindings(model, layout)
                mujoco.mj_saveModel(model, str(a.output / f"{si}-{arm}.mjb"), None)
                motion = env.command_manager.get_term("motion")
                if motion.motion.motion_names != [source.removesuffix(".npz")] or torch.any(
                    motion.time_steps != 0
                ):
                    raise ValueError("declared DEV source and frame0 required")
                snapshot = {
                    k: v.cpu().numpy().copy()
                    for k, v in dict(
                        qpos=env.sim.data.qpos, qvel=env.sim.data.qvel, observation=obs["actor"]
                    ).items()
                }
                if paired is None:
                    paired = snapshot
                elif any(not np.array_equal(v, paired[k]) for k, v in snapshot.items()):
                    raise ValueError("initial/final states and observations must be bitwise paired")
                score = MotionFirstEpisode(32)
                rows = []
                for _ in range(300):
                    raw = dict(
                        active_before=score.active.copy(),
                        qpos_before=env.sim.data.qpos.cpu().numpy().copy(),
                        qvel_before=env.sim.data.qvel.cpu().numpy().copy(),
                        observation=obs["actor"].cpu().numpy().copy(),
                        source_phase=motion.time_steps.cpu().numpy().copy(),
                        expected_kicking_foot=motion.kick_leg.cpu().numpy().copy(),
                        goal_target_world=(motion.target_destination_pos + env.scene.env_origins)
                        .cpu()
                        .numpy()
                        .copy(),
                    )
                    with torch.inference_mode():
                        action = actor(obs["actor"])
                    nn += 32
                    obs, reward, term, trunc, _ = env.step(action)
                    steps += 320
                    physical = env.physical_step_data()
                    if not torch.equal(physical["terminal_capture_mask"], term | trunc):
                        raise ValueError("terminal physical observations must precede reset")
                    raw.update({k: t.cpu().numpy().copy() for k, t in physical.items()})
                    raw.update(
                        action=action.cpu().numpy().copy(),
                        reward=reward.cpu().numpy().copy(),
                        terminated=term.cpu().numpy().copy(),
                        truncated=trunc.cpu().numpy().copy(),
                    )
                    rows.append(raw)
                    score.observe(
                        raw["reward"],
                        raw["terminated"],
                        raw["truncated"],
                        raw["reference_exhausted"],
                    )
                    if not score.active.any():
                        break
                first, bad, clean, peaks = [], [], [], []
                for world, count in enumerate(score.steps):
                    ff = np.array(
                        [r["foot_force_history"][world] for r in rows[: int(count)]]
                    ).reshape(-1, 2)
                    nf = np.array(
                        [r["nonfoot_force_history"][world] for r in rows[: int(count)]]
                    ).ravel()
                    hit = np.flatnonzero((ff > 10.0).any(axis=1))
                    side = int(rows[0]["expected_kicking_foot"][world])
                    good = bool(
                        len(hit) and (ff[hit[0]] > 10.0).sum() == 1 and ff[hit[0], side] > 10.0
                    )
                    forbidden = bool((nf > 1.0).any())
                    first.append(good)
                    bad.append(forbidden)
                    clean.append(good and not forbidden)
                    peaks.append(float(nf.max()))
                path = a.output / f"{si}-{arm}-raw.npz"
                np.savez_compressed(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
                summary = dict(
                    source=source,
                    arm=arm,
                    seed=1501 + si,
                    **score.summary(),
                    intended_foot_first=first,
                    forbidden_nonfoot=bad,
                    clean_foot_first=clean,
                    nonfoot_peak_n=peaks,
                    raw_hash=digest(path),
                    model_hash=digest(a.output / f"{si}-{arm}.mjb"),
                    batched_controls=len(rows),
                )
                write_once(a.output / f"{si}-{arm}.json", summary)
                summaries.append(summary)
                env.close()
                env = None
                rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FOOT_ONLY_PAIRED_PUBLIC_DEV_NOT_SKILL_QUALIFIED",
                input_file_hashes=pins,
                results=summaries,
                gpu_world_steps=steps,
                nn_rows=nn,
                optimizer_updates=0,
                all_initial_states_paired_exact=True,
                known_public_dev_not_blind_fresh=True,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as exc:
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "partial.npz", **{k: np.array([r[k] for r in rows]) for k in rows[0]}
            )
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(exc),
                gpu_world_steps=steps,
                nn_rows=nn,
                optimizer_updates=0,
                hardware_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
