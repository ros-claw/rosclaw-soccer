"""Paired initial/final neural motion tracking on two public DEV sources.

No learning, physical-body edits during steps, football promotion or video.
Every role/state/physics comparison after this remains an independent exam.
"""

import argparse
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("course-root", "reference-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("course-unit", "course-hash", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_mjlab_motion_learning_smoke import changed_weight_names, cloned_tensors
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("frozen evaluator and unused external output required")
    wait_for_terminal_collection(a.course_unit)
    course_path = a.course_root / "complete.json"
    if digest(course_path) != a.course_hash:
        raise ValueError("declared actual course receipt required")
    course = json.loads(course_path.read_text())
    if (
        course["status"] != "MJLAB_MOTION_GPU_PPO_COURSE_NOT_SKILL_QUALIFIED"
        or course["new_optimizer_steps"] <= 0
        or not course["weights_only_checkpoint_reload"]
        or course["football_skill_proven"]
        or course["hardware_authorized"]
    ):
        raise ValueError("actual completed non-promoted motion course required")
    pins = dict(course["input_file_hashes"])
    pins[str(course_path)] = a.course_hash
    checkpoint_path = a.course_root / f"learner-update{course['iterations']}.pt"
    pins[str(checkpoint_path)] = course["checkpoint_hash"]
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts", a.reference_root / "src"):
        for source in directory.rglob("*.py"):
            pins[str(source)] = digest(source)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            num_envs=32,
            horizon=300,
            reset_seeds=[1501, 1502],
            first_episode_only=True,
            mode="SIM_ONLY",
            max_gpu_world_substeps=384000,
            new_optimizer_steps=0,
            public_source_development_not_fresh_world=True,
            football_skill_proven=False,
            hardware_authorized=False,
        ),
    )
    env = None
    steps_total = 0
    neural_samples = 0
    rows = []
    try:
        import numpy as np
        import torch
        from tensordict import TensorDict

        if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
            raise ValueError("one isolated GPU required")
        sys.path.insert(0, str(a.reference_root))
        from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
        from src.tasks.soccer.config.g1.rl_cfg import unitree_g1_soccer_ppo_runner_cfg

        from rosclaw_soccer.rsi.mjlab_motion_curriculum import DEVELOPMENT_MOTION_NAMES
        from rosclaw_soccer.rsi.mjlab_motion_evaluation import MotionFirstEpisode
        from scripts.rsi_mjlab_motion_backend import (
            ExplicitBoundaryMotionEnv,
            make_motion_training_cfg,
        )

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        initial = checkpoint["initial_actor_state_dict"]
        final = checkpoint["algorithm"]["actor_state_dict"]
        if not changed_weight_names(initial, cloned_tensors(final)):
            raise ValueError("actual initial and final neural networks required")
        summaries = []
        for source_index, source in enumerate(DEVELOPMENT_MOTION_NAMES):
            initial_state = None
            for arm, state in (("initial", initial), ("final", final)):
                rows = []
                cfg = make_motion_training_cfg(
                    motion_dir=a.reference_root / "src/assets/soccer/motions/shooter",
                    num_envs=32,
                    seed=1501 + source_index,
                    motion_glob=source,
                )
                # uniform loads the actual source frame0 body with reset perturbations;
                # the template's "start" mode only resets time and is NOT appropriate.
                cfg.commands["motion"].sampling_mode = "uniform"
                cfg.sim.nan_guard.enabled = True
                env = ExplicitBoundaryMotionEnv(cfg=cfg, device="cuda:0", render_mode=None)
                agent_cfg = unitree_g1_soccer_ppo_runner_cfg()
                wrapped = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
                runner = MjlabOnPolicyRunner(wrapped, asdict(agent_cfg), None, "cuda:0")
                policy = runner.get_inference_policy()
                policy.load_state_dict(state, strict=True)
                loaded_state = cloned_tensors(policy.state_dict())
                obs, _ = env.reset(seed=1501 + source_index)
                motion = env.command_manager.get_term("motion")
                if motion.motion.motion_names != [source.removesuffix(".npz")] or torch.any(
                    motion.time_steps != 0
                ):
                    raise ValueError("single declared DEV source from actual frame0 required")
                snapshot = dict(
                    qpos=env.sim.data.qpos.cpu().numpy().copy(),
                    qvel=env.sim.data.qvel.cpu().numpy().copy(),
                    observation=obs["actor"].cpu().numpy().copy(),
                    phase=motion.time_steps.cpu().numpy().copy(),
                )
                if initial_state is None:
                    initial_state = snapshot
                elif any(not np.array_equal(snapshot[k], initial_state[k]) for k in snapshot):
                    raise ValueError("initial/final arms must start from exactly identical states")
                np.savez_compressed(a.output / f"{source_index}-{arm}-initial.npz", **snapshot)
                score = MotionFirstEpisode(32)
                counter = env._sim_step_counter
                with torch.inference_mode():
                    for _frame in range(300):
                        raw = dict(
                            qpos_before=env.sim.data.qpos.cpu().numpy().copy(),
                            qvel_before=env.sim.data.qvel.cpu().numpy().copy(),
                            observation=obs["actor"].cpu().numpy().copy(),
                            source_phase=motion.time_steps.cpu().numpy().copy(),
                            active_before=score.active.copy(),
                        )
                        actions = policy(TensorDict(obs, batch_size=[32]), stochastic_output=False)
                        neural_samples += 32
                        if not torch.isfinite(actions).all():
                            raise ValueError("finite deterministic neural actions required")
                        raw["action"] = actions.cpu().numpy().copy()
                        applied = (
                            actions
                            if agent_cfg.clip_actions is None
                            else torch.clamp(
                                actions, -agent_cfg.clip_actions, agent_cfg.clip_actions
                            )
                        )
                        obs, reward, term, trunc, _ = env.step(applied)
                        torch.cuda.synchronize()
                        steps_total += 10
                        if (
                            not torch.isfinite(env.sim.data.qpos).all()
                            or not torch.isfinite(env.sim.data.qvel).all()
                        ):
                            raise ValueError("finite actual body state required")
                        raw.update(
                            qpos_after=env.sim.data.qpos.cpu().numpy().copy(),
                            qvel_after=env.sim.data.qvel.cpu().numpy().copy(),
                            reward=reward.cpu().numpy().copy(),
                            terminated=term.cpu().numpy().copy(),
                            truncated=trunc.cpu().numpy().copy(),
                            reference_exhausted=env.termination_manager.get_term(
                                "reference_exhausted"
                            )
                            .cpu()
                            .numpy()
                            .copy(),
                        )
                        score.observe(
                            raw["reward"],
                            raw["terminated"],
                            raw["truncated"],
                            raw["reference_exhausted"],
                        )
                        rows.append(raw)
                        if not score.active.any():
                            break
                if env._sim_step_counter - counter != len(rows) * 10:
                    raise ValueError("actual GPU physics counter differs")
                if any(
                    not torch.equal(v, loaded_state[k])
                    for k, v in cloned_tensors(policy.state_dict()).items()
                ):
                    raise ValueError("inference must not update model or normalization state")
                path = a.output / f"{source_index}-{arm}-raw.npz"
                np.savez_compressed(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
                summary = dict(
                    source=source,
                    arm=arm,
                    seed=1501 + source_index,
                    raw_hash=digest(path),
                    batched_control_steps=len(rows),
                    gpu_world_substeps=len(rows) * 10 * 32,
                    initial_phase_zero=True,
                    **score.summary(),
                )
                write_once(a.output / f"{source_index}-{arm}.json", summary)
                summaries.append(summary)
                print(json.dumps(summary, allow_nan=False), flush=True)
                env.close()
                env = None
                rows = []
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="MJLAB_PAIRED_DEV_MOTION_TRACKING_NOT_FOOTBALL_QUALIFIED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=summaries,
                gpu_step_calls=steps_total,
                gpu_world_substeps=steps_total * 32,
                actual_neural_control_samples=neural_samples,
                new_optimizer_steps=0,
                all_initial_states_paired_exact=True,
                first_episode_only=True,
                public_source_development_not_fresh_world=True,
                football_skill_proven=False,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        if rows:
            import numpy as np

            np.savez_compressed(
                a.output / "partial-failed-arm.npz",
                **{k: np.array([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                gpu_step_calls=steps_total,
                active_env_gpu_step_counter=env._sim_step_counter if env is not None else None,
                actual_neural_control_samples=neural_samples,
                new_optimizer_steps=0,
                hardware_authorized=False,
                automatic_retry_authorized=False,
            ),
        )
        raise
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
