"""Fifteen exogenous velocity commands, fresh actor/native body and full repeats."""

import argparse
import json
import subprocess
from pathlib import Path


def episode(a, world, course, model, config, qualification, runtime):
    import numpy as np

    from rosclaw_soccer.physics.g1_target_episode import G1TargetEpisode
    from rosclaw_soccer.physics.g1_velocity_observation import (
        G1VelocityObservation,
        world_command_to_body,
    )
    from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
    from rosclaw_soccer.rsi.velocity_policy_native_evidence import review_velocity_recording
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    folder = a.output / f"command{course.index:02d}"
    folder.mkdir(parents=True, exist_ok=False)
    actor, body = None, None
    rows = []
    try:
        check_pins(runtime)
        actor = G1VelocityPolicy(
            model,
            config,
            expected_model_hash=qualification["model_hash"],
            expected_config_hash=qualification["config_hash"],
        )
        if actor.contract_hash != a.policy_hash or actor.params.contract_hash != a.parameter_hash:
            raise ValueError("same qualified actor/codec required before physical body allocation")
        p = actor.params
        sensor = G1VelocityObservation(world, expected_world_hash=digest(a.world))
        body = G1TargetEpisode(
            world,
            world_hash=digest(a.world),
            kp=np.asarray(p.stiffness, dtype=np.float64),
            kd=np.asarray(p.damping, dtype=np.float64),
            defaults=np.asarray(p.defaults, dtype=np.float64),
            course=course.ball_course,
            record_all_ball_contacts=True,
        )
        for frame in range(300):
            q, v = body.canonical()
            cmd = course.command(frame)
            omega = sensor.angular_velocity_body(q, v)
            body_command = world_command_to_body(q, cmd)
            target = actor.propose(q, v, omega, body_command, frame=frame)
            row = body.step(frame, target)
            row.update(
                velocity_policy_observation=actor.last_observation[None].copy(),
                velocity_policy_action_hg=actor.last_action[None].copy(),
                command_body_mps=body_command[None].copy(),
                command_world_mps=np.asarray([cmd], dtype=np.float64),
                angular_velocity_body_radps=omega[None].copy(),
                gait_phase=np.asarray([actor.phase], dtype=np.float32),
            )
            rows.append(row)
        if (
            body.integrated_steps != 3000
            or body.attempted_steps != 3000
            or body.next_frame != 300
            or actor.neural_calls != 300
            or actor.attempted_neural_calls != 300
            or actor.next_frame != 300
        ):
            raise ValueError("exact complete neural and native frame counts required")
        raw = {k: np.concatenate([r[k] for r in rows]) for k in rows[0]}
        result = review_velocity_recording(raw, p, sensor, course)
        np.savez_compressed(folder / "native300.npz", **raw)
        result.update(
            course_index=course.index,
            split=course.split,
            native_steps=3000,
            neural_calls=300,
            attempted_neural_calls=300,
            body_report=body.review(),
            native_hash=digest(folder / "native300.npz"),
            policy_contract_hash=actor.contract_hash,
            parameter_contract_hash=p.contract_hash,
            optimizer_steps=0,
            hardware_authorized=False,
            promotion_authorized=False,
        )
        check_pins(runtime)
        write_once(folder / "complete.json", result)
        print(
            "VELOCITY_POLICY_NATIVE",
            course.index,
            course.split,
            result["mean_world_velocity_last_pulse_second_mps"],
            result["physical_response_guard"],
            result["command_response_research_gate"],
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
                integrated_steps=body.integrated_steps if body else 0,
                attempted_steps=body.attempted_steps if body else 0,
                actual_neural_calls=actor.neural_calls if actor else 0,
                attempted_neural_calls=actor.attempted_neural_calls if actor else 0,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


def compare_velocity_result(actual, expected):
    a = {k: v for k, v in actual.items() if k != "native_hash"}
    b = {k: v for k, v in expected.items() if k != "native_hash"}
    if json.loads(json.dumps(a)) != json.loads(json.dumps(b)):
        raise ValueError("full independent NN/native response metadata repeat differs")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=("collect", "reproduce"), required=True)
    p.add_argument("--part", type=int, choices=(0, 1, 2), required=True)
    for k in ("qualification", "inputs", "core-source", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "qualification-unit", "expected-qualification-hash"):
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
        raise ValueError("unused output and frozen velocity-native source required")
    wait_for_terminal_collection(a.qualification_unit)
    if digest(a.qualification / "complete.json") != a.expected_qualification_hash:
        raise ValueError("exact actual closed NN/C++ codec qualification required")
    q = json.loads((a.qualification / "complete.json").read_text())
    d = json.loads((a.qualification / "declaration.json").read_text())
    if (
        q["status"] != "G1_VELOCITY_POLICY_CPP_CODEC_NN_QUALIFIED_NOT_PHYSICS"
        or q["new_neural_calls"] != 8
        or not q["cpp_float32_target_exact"]
        or not q["second_full_cpu_session_repeat_exact"]
    ):
        raise ValueError("qualified deployment codec is not yet physical capability")
    pins = dict(q["input_file_hashes"])
    pins.update({str(f): digest(f) for f in a.qualification.rglob("*") if f.is_file()})
    qualified_sources = source_suffix_hash_index(pins)
    for relative in (
        "src/rosclaw_soccer/providers/g1/velocity_policy.py",
        "src/rosclaw_soccer/physics/g1_velocity_observation.py",
    ):
        if qualified_sources.get(relative) != {digest(Path.cwd() / relative)}:
            raise ValueError("actual actor and sensor math must match the qualified source exactly")
    if pins.get(str(a.inputs)) != digest(a.inputs):
        raise ValueError("original world input manifest required")
    a.policy_hash, a.parameter_hash = q["policy_contract_hash"], q["parameter_contract_hash"]
    producer = None
    if a.mode == "reproduce":
        if a.producer is None or not a.producer_unit:
            raise ValueError("actual whole producer required")
        wait_for_terminal_collection(a.producer_unit, wait=True)
        producer = json.loads((a.producer / "complete.json").read_text())
        if (
            producer["status"] != "G1_VELOCITY_NATIVE_PART_COLLECTED"
            or producer["source_head"] != a.expected_head
            or producer["part"] != a.part
            or producer["native_steps"] != 15000
            or producer["neural_calls"] != 1500
        ):
            raise ValueError("same source, whole same-part NN/native predecessor required")
        pins.update({str(f): digest(f) for f in a.producer.rglob("*") if f.is_file()})
    check_pins(pins)
    spec = json.loads(a.inputs.read_text())
    a.world = Path(spec["world"])
    model, config = Path(d["model_file"]), Path(d["config_file"])
    runtime = {
        str(f): digest(f)
        for directory in (Path.cwd() / "src", Path.cwd() / "scripts", a.core_source)
        for f in directory.rglob("*.py")
    }
    for f in (a.world, model, config):
        if pins.get(str(f)) != digest(f):
            raise ValueError("unchanged world, velocity weights and deployment parameters required")
        runtime[str(f)] = digest(f)
    pins.update(runtime)
    import mujoco

    from rosclaw_soccer.providers.g1.velocity_policy import VelocityPolicyParameters
    from rosclaw_soccer.rsi.locomotion_response_protocol import (
        preflight_response_courses,
        response_courses,
    )
    from scripts.rsi_probe_compact_source import exact_artifact

    courses = response_courses()
    bank = preflight_response_courses(courses)
    params = VelocityPolicyParameters.load(config, expected_hash=d["config_hash"])
    world = mujoco.MjModel.from_binary_path(str(a.world))
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            mode=a.mode,
            part=a.part,
            bank_hash=bank,
            policy_contract_hash=a.policy_hash,
            parameter_contract_hash=params.contract_hash,
            explicit_new_initial_joint_defaults=params.defaults,
            explicit_new_fixed_stiffness=params.stiffness,
            explicit_new_fixed_damping=params.damping,
            same_compiled_world_and_ball=True,
            not_strict_same_initial_state_or_gain_comparison_to_sonic=True,
            independent_initial_worlds=1,
            maximum_native_steps=15000,
            maximum_neural_calls=1500,
            optimizer_steps=0,
            model_file=str(model),
            config_file=str(config),
            world_file=str(a.world),
            publication_license_cleared=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    rows = []
    try:
        for course in courses[a.part :: 3]:
            row = episode(a, world, course, model, config, d, runtime)
            if producer is not None:
                folder = f"command{course.index:02d}"
                expected = json.loads((a.producer / folder / "complete.json").read_text())
                compare_velocity_result(row, expected)
                exact_artifact(
                    a.output / folder / "native300.npz", a.producer / folder / "native300.npz"
                )
            rows.append(row)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="G1_VELOCITY_NATIVE_PART_REPRODUCED"
                if producer
                else "G1_VELOCITY_NATIVE_PART_COLLECTED",
                source_head=a.expected_head,
                part=a.part,
                bank_hash=bank,
                results=rows,
                native_steps=15000,
                neural_calls=1500,
                policy_contract_hash=a.policy_hash,
                parameter_contract_hash=a.parameter_hash,
                new_optimizer_steps=0,
                football_skill_proven=False,
                goal_complete=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                completed_episodes=len(rows),
                actual_native_neural_counts_in_episode_receipts=True,
                automatic_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
