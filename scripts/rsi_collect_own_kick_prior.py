"""Eight actual own-policy raw-action trajectories; no optimization or efficacy claim."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--audit-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or a.audit.is_symlink() or digest(a.audit) != a.audit_hash:
        raise ValueError("new output and exact closed native backend parity required")
    audit = json.loads(a.audit.read_text())
    if audit[
        "status"
    ] != "ALL80_KICK_BACKEND_TRIALS_NATIVE_LABELS_AND_SOURCE_IDENTITY_VERIFIED" or (
        audit["passes"] != {"onnx": 30, "torch": 30}
        or audit["lost_success_indices"]
        or audit["safety_regression_indices"]
        or audit["actor_optimizer_updates"] != 0
        or audit["promotion_authorized"] is not False
    ):
        raise ValueError("complete untrained backend parity without regressions required")
    pins = dict(audit["input_file_hashes"])
    pins[str(a.audit)] = a.audit_hash
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("unique bound source asset required: " + suffix)
        return paths[0]

    import mujoco
    import numpy as np
    import torch
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.rsi.correlated_motor_exploration import conditional_logp
    from rosclaw_soccer.rsi.kick_prior_actor import torch_kick_policy_type
    from rosclaw_soccer.rsi.kick_prior_own_policy import OwnKickPlayer, OwnKickSession
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.walk_kick_teacher_audit import relabel_arrays
    from rosclaw_soccer.rsi.walk_kick_teacher_episode import evaluate_episode
    from rosclaw_soccer.rsi.walk_kick_teacher_plan import search_course_bank
    from rosclaw_soccer.sim.contracts import hash_json
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    torch.set_num_threads(1)
    courses = search_course_bank(expanded_public=True)
    preflight_receiving_courses(courses)
    indices = [0, 1, 10, 11, 20, 21, 30, 31]
    originals = {}
    for index in indices:
        matches = [
            Path(k)
            for k in pins
            if k.endswith(f"/course-{index:02d}/onnx.json")
            and "/rsi-kick-prior-torch-native-recovery2-" in k
        ]
        if len(matches) != 1:
            raise ValueError("unique authenticated original physical program required")
        row = json.loads(matches[0].read_text())
        if row["course"] != list(courses[index].course):
            raise ValueError("initial course differs from original body experiment")
        originals[index] = row
    vm, vc = one("/velocity/v0/exported/policy.onnx"), one("/velocity/v0/params/deploy.yaml")
    # The earlier physically authenticated envelope pins its exact reference source.
    configs = []
    for name in pins:
        path = Path(name)
        if path.suffix == ".json" and path.stat().st_size < 2048:
            cfg = json.loads(path.read_text())
            if cfg.get("schema") == "soccer.native_single_finisher_experimental_approach.v1":
                configs.append(cfg)
    if len(configs) != 1:
        raise ValueError("unique physically audited approach reference binding required")
    cfg = configs[0]
    assets = dict(
        velocity_model=vm,
        velocity_config=vc,
        velocity_model_hash=pins[str(vm)],
        velocity_config_hash=pins[str(vc)],
        reference_source=Path(cfg["reference_source"]),
        reference_hash=cfg["reference_hash"],
    )
    state, output, policy, mapping = _load_robonaldo(
        one("/policy/robonaldo/FreeKick.py").parents[2]
    )
    deterministic_factory = torch_kick_policy_type(
        policy, one("/policy/robonaldo/model/policy-obs-aic.onnx")
    )
    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(
            one("/cached-standard/declared-scene.xml").read_text(), prefixes=("a/",)
        )
    )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.g1.own_raw_kick_policy_collection.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            course_indices=indices,
            worlds=8,
            native_step_ceiling=28800,
            seed_base=926100,
            raw_action_std=0.015,
            raw_ar_rho=0.9,
            likelihood_semantics=(
                "raw Gaussian AR draw BEFORE original deterministic clip/map/PD codec"
            ),
            commit_semantics="only after shared world verifies exact PD target execution",
            raw_likelihood_replay_max_error=0.002,
            actor_mean_replay_max_error=1e-5,
            frozen_approach_parameters_are_per_course_privileged=True,
            not_a_deployable_approach_policy=True,
            optimizer_updates=0,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    results, total_steps, samples = [], 0, 0
    try:
        for index in indices:
            sessions = []

            def create_policy(*args, _seed=926100 + index, _sessions=sessions, **kwargs):
                instance = deterministic_factory(*args, **kwargs)
                session = OwnKickSession(instance.ort_session.actor, seed=_seed)
                instance.ort_session = session
                _sessions.append(session)
                return instance

            original = originals[index]
            row, arrays = evaluate_episode(
                model=model,
                assets=dict(assets, experimental_fast=original["experimental_fast_approach"]),
                policy_types=(state, output, create_policy, mapping),
                course=courses[index].course,
                parameters=original["parameters"],
                hard_limits=np.array(G1_HARD_TORQUE_LIMITS),
                player_type=OwnKickPlayer,
            )
            total_steps += row["native_steps"]
            if hash_json(
                relabel_arrays(
                    arrays,
                    controls=row["controls"],
                    ball_address=row["ball_qpos_address"],
                    root_address=row["robot_qpos_address"],
                )
            ) != hash_json(row["outcome"]):
                raise ValueError("actual outcome readback differs")
            expected_calls = (
                0 if row["entry_control"] is None else row["controls"] - row["entry_control"]
            )
            mean_error, logp_error, clipped = None, None, 0
            rng_artifact = None
            if expected_calls:
                if len(sessions) != 1 or sessions[0].faulted or sessions[0].pending is not None:
                    raise ValueError("one fault-free private episode session required")
                session = sessions[0]
                records = session.samples
                if len(records) != expected_calls or session.calls != expected_calls:
                    raise ValueError("all own policy proposals must have actual native commitments")
                frames = np.array([r["frame"] for r in records], np.int64)
                if not np.array_equal(frames, np.arange(row["entry_control"], row["controls"])):
                    raise ValueError("complete actual control clock required")
                obs = np.stack([r["observation"] for r in records])
                means = np.stack([r["mean"] for r in records])
                actions = np.stack([r["raw_action"] for r in records])
                logps = np.array([r["raw_logp"] for r in records], np.float32)
                pd = np.stack([r["executed_pd_target"] for r in records])
                with torch.no_grad():
                    replay_mean = session.actor(torch.from_numpy(obs)).numpy()
                    replay_logp = conditional_logp(
                        torch.from_numpy(means),
                        torch.from_numpy(actions),
                        session.log_std,
                        rho=session.rho,
                    ).numpy()
                mean_error = float(abs(replay_mean - means).max())
                logp_error = float(abs(replay_logp - logps).max())
                if (
                    mean_error > 1e-5
                    or logp_error > 0.002
                    or not np.array_equal(pd, arrays["joint_targets"][frames])
                ):
                    raise ValueError(
                        "own-action mean, raw likelihood or exact executed target failed replay"
                    )
                clipped = sum(r["clipped_components"] for r in records)
                arrays.update(
                    own_observations=obs,
                    own_raw_actions=actions,
                    own_raw_logp=logps,
                    own_old_means=means,
                    own_control_frames=frames,
                    own_executed_pd_target=pd,
                )
                rng_path = a.output / f"rng-{index:02d}.pt"
                torch.save(
                    dict(
                        generator_state=session.generator.get_state(),
                        seed=926100 + index,
                        actor_optimizer_updates=0,
                    ),
                    rng_path,
                )
                restored = torch.load(rng_path, map_location="cpu", weights_only=True)
                if not torch.equal(restored["generator_state"], session.generator.get_state()):
                    raise ValueError("safe-loaded episode RNG state differs")
                rng_artifact = dict(path=rng_path.name, hash=digest(rng_path))
                samples += len(records)
            artifact = a.output / f"course-{index:02d}.npz"
            np.savez_compressed(artifact, **arrays)
            row.update(
                course_index=index,
                own_committed_neural_samples=expected_calls,
                actor_mean_replay_max_error=mean_error,
                raw_likelihood_replay_max_error=logp_error,
                original_codec_clipped_raw_components=clipped,
                rng_checkpoint=rng_artifact,
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            write_once(a.output / f"course-{index:02d}.json", row)
            results.append(row)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="EIGHT_OWN_POLICY_KICK_TRAJECTORIES_COMMITTED_AND_RAW_LIKELIHOODS_REPLAYED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                results=results,
                worlds=8,
                native_steps=total_steps,
                own_committed_neural_samples=samples,
                actor_optimizer_updates=0,
                critic_optimizer_updates=0,
                physical_football_gain_proven=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error_type=type(error).__name__,
                error=str(error),
                completed_worlds=len(results),
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
