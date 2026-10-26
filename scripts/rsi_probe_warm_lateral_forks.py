"""Compare three private SIM futures at one physically warmed support state."""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--review", type=Path, required=True)
    p.add_argument("--review-hash", required=True)
    p.add_argument("--baseline", type=Path, required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    baseline_path = a.baseline / "complete.json"
    if a.output.exists() or digest(a.review) != a.review_hash:
        raise ValueError("unused output and exact independent velocity review required")
    review = json.loads(a.review.read_text())
    if (
        review["status"] != "BILATERAL_VELOCITY_NN_AND_MANUAL_NATIVE_EXACT"
        or review["source_hash"] != digest(baseline_path)
        or any(not r["values_exact"] for r in review["records"])
    ):
        raise ValueError("complete independent baseline required")
    baseline = json.loads(baseline_path.read_text())
    pins = dict(review["input_file_hashes"])
    pins[str(a.review)] = a.review_hash
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    if not (len(scenes) == len(models) == len(configs) == 1):
        raise ValueError("one exact scene and velocity model/config required")
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.velocity_stream_branch import (
        fork_velocity_stream,
        velocity_stream_state_hash,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("original canonical body required")
    layout = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    source_row = next(r for r in baseline["records"] if r["case"] == "new-left")
    source_file = a.baseline / source_row["artifact"]["path"]
    if digest(source_file) != source_row["artifact"]["hash"]:
        raise ValueError("exact verified initial source trace required")
    with np.load(source_file, allow_pickle=False) as old:
        q0, v0 = old["initial_qpos"].copy(), old["initial_qvel"].copy()
    stream = ContinuousVelocityStream(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
        horizon_controls=301,
    )
    d = mujoco.MjData(model)
    d.qpos[:], d.qvel[:] = q0, v0
    mujoco.mj_forward(model, d)
    pd = NativePDParameters(
        np.array(stream.params.stiffness),
        np.array(stream.params.damping),
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    donor = SharedNativeMotorWorld(model, d, players={"a/": pd}, ball_body_name="soccer_ball")
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            review_hash=a.review_hash,
            input_file_hashes=pins,
            warm_controls=25,
            warm_command=[0.0, 0.0, 0.0],
            future_controls=32,
            fork_commands={
                "left": [0.0, 0.25, 0.0],
                "hold": [0.0, 0.0, 0.0],
                "right": [0.0, -0.25, 0.0],
            },
            native_step_ceiling=1210,
            branch_worlds_are_counterfactual=True,
            stream_phase_and_action_history_are_not_reset=True,
            shared_warm_physical_state_and_history=True,
            optimizer_updates=0,
            no_kick_or_handoff=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    warm_raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
    warm_targets, warm_obs, warm_actions = [], [], []
    for frame in range(25):
        q, v = layout.canonical(d)
        target = stream.propose(
            q, v, layout.angular_velocity_body(d), np.zeros(3, np.float32), frame=frame
        )
        result = donor.advance({"a/": target})
        for key in warm_raw:
            warm_raw[key].append(result[key])
        warm_targets.append(target.copy())
        warm_obs.append(stream.last_observation.copy())
        warm_actions.append(stream.last_action.copy())
        if donor.failed:
            raise ValueError("warm up body failed; no branches authorized")
    warm = {key: np.concatenate(value) for key, value in warm_raw.items()}
    warm.update(
        initial_qpos=q0,
        initial_qvel=v0,
        joint_targets=np.stack(warm_targets),
        neural_obs=np.stack(warm_obs),
        neural_actions=np.stack(warm_actions),
    )
    warm_path = a.output / "warm.npz"
    np.savez_compressed(warm_path, **warm)
    before = snapshot(d)
    before_stream = velocity_stream_state_hash(stream, completed_control=25)
    write_once(
        a.output / "warm.json",
        dict(
            controls=25,
            native_steps=250,
            phase=float(stream.phase),
            stream_hash=before_stream,
            artifact=dict(path=warm_path.name, hash=digest(warm_path)),
            body_failed=False,
        ),
    )
    rows = []
    for name, lateral in (("left", 0.25), ("hold", 0.0), ("right", -0.25)):
        copy_data = mujoco.MjData(model)
        mujoco.mj_copyData(copy_data, model, d)
        ghost = fork_velocity_stream(stream, completed_control=25)
        world = SharedNativeMotorWorld(
            model, copy_data, players={"a/": pd}, ball_body_name="soccer_ball"
        )
        raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
        targets, observations, actions = [], [], []
        for local in range(32):
            q, v = layout.canonical(copy_data)
            command = np.array([0.0, lateral, 0.0], dtype=np.float32)
            target = ghost.propose(
                q, v, layout.angular_velocity_body(copy_data), command, frame=25 + local
            )
            result = world.advance({"a/": target})
            for key in raw:
                raw[key].append(result[key])
            targets.append(target.copy())
            observations.append(ghost.last_observation.copy())
            actions.append(ghost.last_action.copy())
            if world.failed:
                break
        arrays = {key: np.concatenate(value) for key, value in raw.items()}
        arrays.update(
            joint_targets=np.stack(targets),
            neural_obs=np.stack(observations),
            neural_actions=np.stack(actions),
        )
        path = a.output / (name + ".npz")
        np.savez_compressed(path, **arrays)
        donor._require_current_boundary()
        if (
            any(not np.array_equal(v, snapshot(d)[k]) for k, v in before.items())
            or velocity_stream_state_hash(stream, completed_control=25) != before_stream
        ):
            raise ValueError("counterfactual forecast mutated actual donor or NN history")
        row = dict(
            case=name,
            controls=world.control_steps,
            native_steps=world.native_steps,
            body_failed=world.failed,
            signed_y_displacement_14_m=float(
                arrays["qpos"][min(139, len(arrays["qpos"]) - 1), layout.rq + 1]
                - d.qpos[layout.rq + 1]
            ),
            signed_y_displacement_end_m=float(
                copy_data.qpos[layout.rq + 1] - d.qpos[layout.rq + 1]
            ),
            artifact=dict(path=path.name, hash=digest(path)),
            donor_unchanged=True,
            forecast_not_actual_execution=True,
        )
        rows.append(row)
        write_once(a.output / (name + ".json"), row)
        print(json.dumps(row), flush=True)
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="WARM_LATERAL_COUNTERFACTUALS_RECORDED",
            source_head=a.expected_head,
            review_hash=a.review_hash,
            records=rows,
            input_file_hashes=pins,
            neural_calls=25 + sum(r["controls"] for r in rows),
            native_steps=250 + sum(r["native_steps"] for r in rows),
            branches_not_actual_execution=True,
            independent_replay=False,
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )


if __name__ == "__main__":
    main()
