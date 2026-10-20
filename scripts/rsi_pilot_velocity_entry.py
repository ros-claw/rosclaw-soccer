"""New explicit velocity actor with unchanged measured-ball entry teacher, SIM only."""

import argparse
import json
import subprocess
from pathlib import Path


def run_episode(a, arm, world, config, model, params, affordance, course, pins):
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.g1_velocity_observation import (
        G1VelocityObservation,
        world_command_to_body,
    )
    from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
    from rosclaw_soccer.rsi.ball_entry_teacher import ball_entry_decision
    from rosclaw_soccer.rsi.velocity_entry_evidence import review_entry
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = a.output / arm
    folder.mkdir(parents=True)
    check_pins(pins)
    actor = G1VelocityPolicy(
        model, config, expected_model_hash=pins[str(model)], expected_config_hash=pins[str(config)]
    )
    sensor = G1VelocityObservation(world, expected_world_hash=pins[str(a.world)])
    body = G1TargetEpisode(
        world,
        world_hash=pins[str(a.world)],
        kp=np.asarray(params.stiffness),
        kd=np.asarray(params.damping),
        defaults=np.asarray(params.defaults),
        course=course,
        record_all_ball_contacts=True,
    )
    rows = []
    try:
        for f in range(300):
            q, v = body.canonical()
            decision = ball_entry_decision(q, v, frame=f, affordance=affordance)
            cmd = decision.command if arm == "measured_entry" else (0.0, 0.0, 0.0)
            omega = sensor.angular_velocity_body(q, v)
            bc = world_command_to_body(q, cmd)
            target = actor.propose(q, v, omega, bc, frame=f)
            row = body.step(f, target)
            row.update(
                velocity_policy_observation=actor.last_observation[None].copy(),
                velocity_policy_action_hg=actor.last_action[None].copy(),
                command_body_mps=bc[None].copy(),
                command_world_mps=np.array([cmd]),
                angular_velocity_body_radps=omega[None].copy(),
                gait_phase=np.array([actor.phase], dtype=np.float32),
                entry_target_root_xy_m=np.array([decision.target_root_xy_m]),
                entry_error_m=np.array([decision.horizontal_error_m]),
                entry_ready=np.array([decision.ready], dtype=np.bool_),
                entry_timed_out=np.array([decision.timed_out], dtype=np.bool_),
            )
            rows.append(row)
        if (
            actor.neural_calls != 300
            or actor.attempted_neural_calls != 300
            or actor.next_frame != 300
        ):
            raise ValueError("all 300 real neural calls required")
        raw = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        result = review_entry(raw, params, sensor, affordance, course, arm)
        np.savez_compressed(folder / "native300.npz", **raw)
        result.update(
            native_steps=3000,
            neural_calls=300,
            attempted_neural_calls=300,
            body_report=body.review(),
            native_hash=digest(folder / "native300.npz"),
            policy_contract_hash=actor.contract_hash,
            entry_contract_hash=affordance.contract_hash,
        )
        check_pins(pins)
        write_once(folder / "complete.json", result)
        print(
            "VELOCITY_ENTRY",
            arm,
            result["first_ready_frame"],
            result["minimum_entry_error_frames30_to90_m"],
            result["physical_response_guard"],
            flush=True,
        )
        return result
    except Exception as error:
        if rows:
            np.savez_compressed(
                folder / "partial-native.npz",
                **{k: np.concatenate([r[k] for r in rows]) for k in rows[0]},
            )
        write_once(
            folder / "failure.json",
            dict(
                error=repr(error),
                integrated_steps=body.integrated_steps,
                attempted_steps=body.attempted_steps,
                neural_calls=actor.neural_calls,
                attempted_neural_calls=actor.attempted_neural_calls,
                automatic_retry_authorized=False,
            ),
        )
        raise


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("produce", "reproduce"), required=True)
    for k in ("review", "calibration", "past-prior", "output", "core-source"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "review-unit", "expected-review-hash"):
        p.add_argument("--" + k, required=True)
    p.add_argument("--producer", type=Path)
    p.add_argument("--producer-unit")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import (
        source_suffix_hash_index,
        wait_for_terminal_collection,
    )
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("clean pinned source and unused output required")
    wait_for_terminal_collection(a.review_unit)
    if digest(a.review / "complete.json") != a.expected_review_hash:
        raise ValueError("exact closed whole native response review required")
    review = json.loads((a.review / "complete.json").read_text())
    if (
        review["status"] != "G1_VELOCITY_NATIVE_90K_BANK_AUDITED_NOT_FOOTBALL"
        or review["physical_guard_passes"] != 15
        or review["research_response_gate_passes"] < 14
        or review["development_validation_gate_passes"] != 6
        or not review["all_15_independent_nn_native_repeats_exact"]
    ):
        raise ValueError("whole stable response capability required before entry trial")
    pins = dict(review["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.review.rglob("*") if f.is_file()})
    for f in (a.calibration / "declaration.json", a.calibration / "complete.json", a.past_prior):
        if pins.get(str(f)) != digest(f):
            raise ValueError("exact historical entry prior required")
    index = source_suffix_hash_index(pins)
    for rel in (
        "src/rosclaw_soccer/providers/g1/velocity_policy.py",
        "src/rosclaw_soccer/physics/g1_velocity_observation.py",
        "src/rosclaw_soccer/rsi/ball_entry_teacher.py",
    ):
        if index.get(rel) != {digest(Path(rel))}:
            raise ValueError(
                "unchanged qualified velocity codec and existing entry teacher required"
            )
    producer = None
    if a.mode == "reproduce":
        if a.producer is None or not a.producer_unit:
            raise ValueError("actual producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "G1_VELOCITY_ENTRY_PRODUCED"
            or producer["source_head"] != a.expected_head
            or producer["native_steps"] != 6000
        ):
            raise ValueError("same whole native predecessor required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    import mujoco

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.ball_entry_teacher import SkillEntryAffordance
    from rosclaw_soccer.rsi.native_entry_calibration import NativeCalibratedEntry
    from scripts.rsi_collect_velocity_policy_response import compare_velocity_result
    from scripts.rsi_probe_compact_source import exact_artifact, load_arrays

    d = json.loads((a.calibration / "declaration.json").read_text())
    stored = json.loads(a.past_prior.read_text())
    parent = SkillEntryAffordance(
        tuple(stored["ball_offset_xy_m"]),
        stored["historical_frame"],
        stored["historical_forward_speed_mps"],
        stored["historical_artifact_hash"],
    )
    if (
        parent.contract_hash != stored["contract_hash"]
        or parent.contract_hash != d["fit"]["parent_affordance_hash"]
    ):
        raise ValueError("exact ancestor of historical calibrated affordance required")
    affordance = NativeCalibratedEntry(
        tuple(d["entry_ball_offset_xy_m"]),
        parent.historical_frame,
        parent.historical_forward_speed_mps,
        parent.historical_artifact_hash,
        parent.contract_hash,
        d["fit"]["motion_configuration_hash"],
        d["fit_hash"],
    )
    # The consumed past context is not a new independent or blind world.
    oldpath = a.calibration / "produce/right-law0/native300.npz"
    if pins.get(str(oldpath)) != digest(oldpath):
        raise ValueError("exact historical physical initial ball context required")
    old = load_arrays(oldpath)
    course = tuple(
        float(x) for x in (*old["canonical_qpos"][0, 36:38], old["canonical_qvel"][0, 35])
    )
    model = next(Path(k) for k in pins if k.endswith("/velocity/v0/exported/policy.onnx"))
    config = next(Path(k) for k in pins if k.endswith("/velocity/v0/params/deploy.yaml"))
    a.world = next(Path(k) for k in pins if k.endswith("/attempt-0004/compiled_model.mjb"))
    params = VelocityPolicyParameters.load(config, expected_hash=pins[str(config)])
    runtime = {
        str(f): digest(f)
        for dr in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in dr.rglob("*.py")
    }
    pins.update(runtime)
    world = mujoco.MjModel.from_binary_path(str(a.world))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            mode=a.mode,
            course=course,
            arms=["idle", "measured_entry"],
            independent_initial_worlds=1,
            maximum_native_steps=6000,
            maximum_neural_calls=600,
            optimizer_steps=0,
            entry_contract_hash=affordance.contract_hash,
            parameter_contract_hash=params.contract_hash,
            old_4cm_frame30_to90_gate_unchanged=True,
            no_motor_handoff=True,
            new_velocity_defaults_and_fixed_pd=True,
            football_skill_proven=False,
            publication_license_cleared=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    results = []
    try:
        for arm in ("idle", "measured_entry"):
            row = run_episode(a, arm, world, config, model, params, affordance, course, pins)
            if producer:
                compare_velocity_result(row, producer["results"][len(results)])
                exact_artifact(a.output / arm / "native300.npz", a.producer / arm / "native300.npz")
            results.append(row)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="G1_VELOCITY_ENTRY_REPRODUCED" if producer else "G1_VELOCITY_ENTRY_PRODUCED",
                source_head=a.expected_head,
                results=results,
                native_steps=6000,
                neural_calls=600,
                optimizer_steps=0,
                goal_complete=False,
                football_skill_proven=False,
                motor_handoff_performed=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_episodes=len(results),
                actual_native_neural_counts_in_episode_receipts=True,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
