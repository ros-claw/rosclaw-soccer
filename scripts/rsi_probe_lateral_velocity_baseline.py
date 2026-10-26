"""SIM-only bilateral velocity baseline, not a kick or a trained policy."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--stage-source", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.review) != a.review_hash:
        raise ValueError("unused output and exact review required")
    review = json.loads(a.review.read_text())
    source_path = a.stage_source / "complete.json"
    if (
        review["status"] != "CFTRACK_STAGED_LATERAL_NN_NATIVE_AND_BOUNDARY_EXACT"
        or review["source_hash"] != digest(source_path)
        or review["new_neural_calls"] != 314
        or review["new_native_steps"] != 3140
    ):
        raise ValueError("exact closed independently replayed stage source required")
    source = json.loads(source_path.read_text())
    pins = dict(review["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    if not (len(scenes) == len(models) == len(configs) == 1):
        raise ValueError("exact one canonical scene, velocity model and codec required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash
    from rosclaw_soccer.rsi.stage_tracking_diagnostic import signed_stage_tracking

    scene = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(scene)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact original physical model required")
    layout = NativeVelocityLayout(scene, robot_prefix="a/", ball_body_name="soccer_ball")
    cases = (("new-left", 0.25), ("new-right", -0.25))
    rows = []
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.bilateral_velocity_baseline.v1",
            source_head=a.expected_head,
            source_review_hash=a.review_hash,
            input_file_hashes=pins,
            cases=cases,
            controls_per_case=300,
            native_step_budget=6000,
            source_initial_state_exact=True,
            source_reference_and_stage_guard_not_run=True,
            independent_velocity_pd_profile=True,
            no_skill_handoff_or_kick=True,
            initial_stand_to_lateral_command_abrupt=True,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    for name, lateral in cases:
        matching = [row for row in source["records"] if row["case"] == name]
        if len(matching) != 1:
            raise ValueError("one source-bound stage case required")
        record = matching[0]
        artifact = a.stage_source / record["artifact"]["path"]
        if digest(artifact) != record["artifact"]["hash"]:
            raise ValueError("exact source trajectory required")
        with np.load(artifact, allow_pickle=False) as previous:
            q0, v0 = previous["initial_qpos"].copy(), previous["initial_qvel"].copy()
        stream = ContinuousVelocityStream(
            models[0],
            configs[0],
            expected_model_hash=pins[str(models[0])],
            expected_config_hash=pins[str(configs[0])],
            horizon_controls=301,
        )
        data = mujoco.MjData(scene)
        data.qpos[:], data.qvel[:] = q0, v0
        mujoco.mj_forward(scene, data)
        pd = NativePDParameters(
            np.array(stream.params.stiffness),
            np.array(stream.params.damping),
            0.85 * np.array(G1_HARD_TORQUE_LIMITS),
        )
        world = SharedNativeMotorWorld(
            scene, data, players={"a/": pd}, ball_body_name="soccer_ball"
        )
        raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
        targets, obs, actions = [], [], []
        for frame in range(300):
            q, v = layout.canonical(data)
            target = stream.propose(
                q,
                v,
                layout.angular_velocity_body(data),
                np.array([0.0, lateral, 0.0], dtype=np.float32),
                frame=frame,
            )
            result = world.advance({"a/": target})
            for key in raw:
                raw[key].append(result[key])
            targets.append(target.copy())
            obs.append(stream.last_observation.copy())
            actions.append(stream.last_action.copy())
            if world.failed:
                break
        arrays = {key: np.concatenate(value) for key, value in raw.items()}
        arrays.update(
            initial_qpos=q0,
            initial_qvel=v0,
            joint_targets=np.stack(targets),
            neural_obs=np.stack(obs),
            neural_actions=np.stack(actions),
        )
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **arrays)
        n = min(140, world.native_steps)
        positions = np.vstack(
            [q0[layout.rq : layout.rq + 2], arrays["qpos"][:n, layout.rq : layout.rq + 2]]
        )
        velocities = np.vstack(
            [v0[layout.rv : layout.rv + 2], arrays["qvel"][:n, layout.rv : layout.rv + 2]]
        )
        waypoint = np.array(
            record["reference"]["geometry"]["planned_waypoint"][:2], dtype=np.float64
        )
        diagnostic = signed_stage_tracking(
            positions,
            velocities,
            intended_displacement=waypoint - positions[0],
            native_timestep_s=0.002,
        )
        row = dict(
            case=name,
            lateral_command_body_m_s=lateral,
            controls=world.control_steps,
            native_steps=world.native_steps,
            neural_calls=stream.neural_calls,
            body_failed=world.failed,
            minimum_pelvis_z_m=float(np.min(arrays["qpos"][:, layout.rq + 2])),
            first_140_native_diagnostic=diagnostic,
            ball_contact_force_max_n=float(np.max(np.abs(arrays["ball_forces"]))),
            artifact=dict(path=path.name, hash=digest(path)),
        )
        rows.append(row)
        write_once(a.output / (name + ".json"), row)
        print(json.dumps(row), flush=True)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="BILATERAL_VELOCITY_BASELINE_RECORDED",
            source_head=a.expected_head,
            source_review_hash=a.review_hash,
            records=rows,
            input_file_hashes=pins,
            optimizer_updates=0,
            independent_replay=False,
            task_success_not_measured=True,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
