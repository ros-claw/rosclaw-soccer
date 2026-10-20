"""Pin the external G1 actor and independently verify its codec before dynamics."""

import argparse
import json
import subprocess
import xml.etree.ElementTree as ET
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent", "inputs", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in (
        "expected-head",
        "expected-parent-hash",
        "parent-unit",
        "expected-model-hash",
        "expected-config-hash",
    ):
        p.add_argument("--" + k, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_collect_shadow_motor import wait_for_terminal_collection
    from scripts.rsi_native_context_recovery_common import check_pins, digest

    if (
        a.output.exists()
        or subprocess.check_output(["git", "status", "--porcelain"])
        or subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
        != a.expected_head
    ):
        raise ValueError("clean frozen qualification source and unused output required")
    wait_for_terminal_collection(a.parent_unit)
    if digest(a.parent / "complete.json") != a.expected_parent_hash:
        raise ValueError("exact native waypoint/input-channel failure review required")
    parent = json.loads((a.parent / "complete.json").read_text())
    if (
        parent["status"] != "ROOT_WAYPOINT_NATIVE24K_AND_INPUT_CHANNEL_AUDITED"
        or parent["physics_improvement_proven"] is not False
    ):
        raise ValueError(
            "this changed mechanism follows the verified failed position-channel candidate"
        )
    pins = dict(parent["input_file_hashes"])
    pins[str(a.parent / "complete.json")] = a.expected_parent_hash
    if pins.get(str(a.inputs)) != digest(a.inputs):
        raise ValueError("same frozen world/asset manifest required")
    folder = a.reference_root / "deploy/robots/g1/config/policy/velocity/v0"
    model, config = folder / "exported/policy.onnx", folder / "params/deploy.yaml"
    if digest(model) != a.expected_model_hash or digest(config) != a.expected_config_hash:
        raise ValueError("exact independently pinned external actor and parameter file required")
    sourcefiles = [model, config, a.reference_root / "src/assets/robots/unitree_g1/xmls/g1.xml"]
    sourcefiles += [
        a.reference_root / "deploy/include" / n
        for n in (
            "isaaclab/envs/mdp/observations/observations.h",
            "unitree_articulation.h",
            "isaaclab/envs/manager_based_rl_env.h",
            "isaaclab/manager/observation_manager.h",
            "isaaclab/envs/mdp/actions/joint_actions.h",
        )
    ]
    for f in sourcefiles:
        pins[str(f)] = digest(f)
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in directory.rglob("*.py")})
    cpp = Path.cwd() / "scripts/velocity_codec_reference.cpp"
    pins[str(cpp)] = digest(cpp)
    check_pins(pins)
    import mujoco
    import numpy as np

    from rosclaw_soccer.physics.g1_velocity_observation import G1VelocityObservation
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.providers.g1.velocity_policy import (
        G1VelocityPolicy,
        VelocityPolicyParameters,
    )

    names = tuple(
        j.attrib["name"] for j in ET.parse(sourcefiles[2]).getroot().findall(".//worldbody//joint")
    )
    if names != G1_DDS_JOINT_NAMES:
        raise ValueError("upstream robot XML and canonical HG29 joint order differ")
    spec = json.loads(a.inputs.read_text())
    world_path = Path(spec["world"])
    world = mujoco.MjModel.from_binary_path(str(world_path))
    sensor = G1VelocityObservation(world, expected_world_hash=digest(world_path))
    params = VelocityPolicyParameters.load(config, expected_hash=a.expected_config_hash)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            model_file=str(model),
            config_file=str(config),
            model_hash=a.expected_model_hash,
            config_hash=a.expected_config_hash,
            parameter_contract_hash=params.contract_hash,
            maximum_neural_calls=8,
            new_native_steps=0,
            private_kinematic_data_only_not_dynamic_body=True,
            optimizer_steps=0,
            original_sonic_gains_or_defaults_not_overwritten=True,
            upstream_hardware_code_read_only_not_executed=True,
            publication_license_cleared=False,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    attempted = 0
    try:
        binary = a.output / "velocity-codec-reference"
        compile_args = [
            "g++",
            "-std=c++17",
            "-O2",
            "-ffp-contract=off",
            "-I/usr/include/eigen3",
            str(cpp),
            "-o",
            str(binary),
        ]
        compiled = subprocess.run(compile_args, capture_output=True, text=True, timeout=60)
        if compiled.returncode:
            raise ValueError("independent C++ codec fixture compile failed: " + compiled.stderr)
        records = []
        for frame, cmd in enumerate(
            ((0.0, 0.0, 0.0), (0.3, 0.0, 0.0), (0.0, 0.3, 0.0), (-0.14, 0.14, 0.1))
        ):
            q, v = np.zeros(43), np.zeros(41)
            q[2] = 0.793
            q[39] = 1.0
            q[36:39] = [3.0, 3.0, 0.13]
            axis = np.array([1.0, 2.0, 3.0])
            axis /= np.linalg.norm(axis)
            angle = frame * 0.07
            q[3:7] = [np.cos(angle / 2), *(axis * np.sin(angle / 2))]
            q[7:36] = np.asarray(params.defaults) + 0.01 * np.sin(np.arange(29) + frame)
            v[3:6] = [0.01 * frame, -0.02 * frame, 0.03 * frame]
            v[6:35] = 0.02 * np.cos(np.arange(29) + frame)
            omega = sensor.angular_velocity_body(q, v)
            np.testing.assert_allclose(omega, v[3:6], rtol=0, atol=1e-12)
            records.append((q, v, omega, np.asarray(cmd, dtype=np.float32)))
        original = None
        contract = None
        for repeat in (0, 1):
            actor = G1VelocityPolicy(
                model,
                config,
                expected_model_hash=a.expected_model_hash,
                expected_config_hash=a.expected_config_hash,
            )
            values = []
            for frame, (q, v, omega, cmd) in enumerate(records):
                attempted += 1
                target = actor.propose(q, v, omega, cmd, frame=frame)
                values.append(
                    dict(
                        qpos=q.copy(),
                        qvel=v.copy(),
                        angular_velocity_body=omega.copy(),
                        command_body=cmd.copy(),
                        observation=actor.last_observation.copy(),
                        action=actor.last_action.copy(),
                        target=target.copy(),
                        phase=np.asarray(actor.phase),
                    )
                )
            trace = {k: np.stack([r[k] for r in values]) for k in values[0]}
            np.savez_compressed(a.output / f"pass{repeat}.npz", **trace)
            if original is None:
                original = trace
                contract = actor.contract_hash
            else:
                if actor.contract_hash != contract or any(
                    not np.array_equal(trace[k], original[k]) for k in original
                ):
                    raise ValueError(
                        "second CPU-session sequential observation/action/target repeat differs"
                    )
        assert original is not None
        fields = ["4"]
        fields += [
            format(float(x), ".9g")
            for values in (params.defaults, params.action_scale, params.action_offset)
            for x in values
        ]
        for i in range(4):
            parts = (
                original["qpos"][i, 3:7],
                original["angular_velocity_body"][i],
                original["qpos"][i, 7:36],
                original["qvel"][i, 6:35],
                original["command_body"][i],
                original["action"][i],
            )
            fields += [format(float(np.float32(x)), ".9g") for vector in parts for x in vector]
        reference = subprocess.run(
            [str(binary)], input=" ".join(fields), capture_output=True, text=True, timeout=10
        )
        if reference.returncode:
            raise ValueError("independent C++ codec fixture failed")
        cpp_values = np.fromstring(reference.stdout, sep=" ", dtype=np.float32)
        if cpp_values.shape != (4 * 127,):
            raise ValueError("complete independent98 observations+29 targets required")
        cpp_values = cpp_values.reshape(4, 127)
        difference = float(np.max(abs(cpp_values[:, :98] - original["observation"])))
        if difference > 2e-6 or not np.array_equal(
            cpp_values[:, 98:].astype(np.float64), original["target"]
        ):
            raise ValueError(f"independent C++ codec mismatch max_observation_error={difference}")
        np.savez_compressed(
            a.output / "cpp_reference.npz",
            observation=cpp_values[:, :98],
            target=cpp_values[:, 98:],
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="G1_VELOCITY_POLICY_CPP_CODEC_NN_QUALIFIED_NOT_PHYSICS",
                source_head=a.expected_head,
                input_file_hashes=pins,
                policy_contract_hash=contract,
                parameter_contract_hash=params.contract_hash,
                new_neural_calls=8,
                new_native_steps=0,
                optimizer_steps=0,
                cpp_binary_hash=digest(binary),
                cpp_compile_command=compile_args,
                cpp_observation_max_abs_error=difference,
                cpp_float32_target_exact=True,
                second_full_cpu_session_repeat_exact=True,
                upstream_joint_order_verified=True,
                synthetic_observations_not_physical_rollouts=True,
                physics_qualified=False,
                goal_complete=False,
                publication_license_cleared=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                attempted_neural_calls=attempted,
                new_native_steps=0,
                optimizer_steps=0,
                automatic_physics_retry_authorized=False,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
