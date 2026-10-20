"""Eight CPU planner calls on one historical measured context; zero dynamics.

Four fixed feeds x two fresh ONNX sessions: original flag0, flag0 sham with
goal values, flag1 real known goal, flag1 reflected negative-control goal.
"""

import argparse
import json
import subprocess
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for k in ("parent", "pilot", "inputs", "reference-root", "output"):
        p.add_argument("--" + k, type=Path, required=True)
    for k in ("expected-head", "expected-parent-hash", "parent-unit"):
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
        raise ValueError("clean frozen probe and unused output required")
    wait_for_terminal_collection(a.parent_unit)
    if digest(a.parent / "complete.json") != a.expected_parent_hash:
        raise ValueError("exact whole locomotion raw audit required")
    parent = json.loads((a.parent / "complete.json").read_text())
    if (
        parent["status"] != "LOCOMOTION_RESPONSES_AUDITED_AND_DELAYED_MODEL_FITTED"
        or parent["predictive_control_research_eligible"] is not False
    ):
        raise ValueError("this changed mechanism requires the rejected simple-predictor evidence")
    pins = dict(parent["input_file_hashes"])
    pins[str(a.parent / "complete.json")] = a.expected_parent_hash
    for f in (a.inputs, a.pilot / "complete.json", a.pilot / "declaration.json"):
        if pins.get(str(f)) != digest(f):
            raise ValueError("same historical context/known entry goal evidence required")
    spec = json.loads(a.inputs.read_text())
    model_root = Path(spec["model_root"])
    model_file = model_root / "planner_sonic.onnx"
    if pins.get(str(model_file)) != digest(model_file):
        raise ValueError("exact already-qualified frozen planner required")
    for directory in (Path.cwd() / "src", Path.cwd() / "scripts"):
        pins.update({str(f): digest(f) for f in directory.rglob("*.py")})
    for f in (
        a.reference_root / "docs/source/references/planner_onnx.md",
        a.reference_root / "motionbricks/motionbricks/motion_backbone/demo/full_agent.py",
    ):
        pins[str(f)] = digest(f)
    check_pins(pins)
    import mujoco
    import numpy as np
    import onnxruntime as ort

    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.planner_waypoint_intent import (
        RootWaypointIntent,
        apply_waypoint_intent,
        past_context_at_30hz,
        validate_planner_output,
    )
    from rosclaw_soccer.sim.contracts import hash_json
    from scripts.rsi_probe_compact_source import load_arrays

    raw_file = a.pilot / "produce/right-law0/native300.npz"
    if pins.get(str(raw_file)) != digest(raw_file):
        raise ValueError("whole audited current-state context identity required")
    raw = load_arrays(raw_file)
    declaration = json.loads((a.pilot / "declaration.json").read_text())
    frame = 60
    q, v = raw["canonical_qpos"][frame], raw["canonical_qvel"][frame]
    offset = np.asarray(declaration["entry_ball_offset_xy_m"], dtype=np.float64)
    goal = q[36:38] + 0.1 * v[35:37] - offset
    if not np.array_equal(goal, raw["entry_target_root_xy_m"][frame]):
        raise ValueError(
            "goal must be derivable from the same current ball and declared entry prior"
        )
    observation = hash_json(dict(frame=frame, qpos=q.tolist(), qvel=v.tolist()))
    positive = RootWaypointIntent(
        (float(goal[0]), float(goal[1]), float(q[2])), 0.0, observation, declaration["fit_hash"]
    )
    neg = 2 * q[:2] - goal
    negative = RootWaypointIntent(
        (float(neg[0]), float(neg[1]), float(q[2])),
        0.0,
        observation,
        hash_json(dict(negative_control_of=positive.contract_hash)),
    )
    command = raw["entry_command"][frame]
    speed = float(np.linalg.norm(command[:2]))
    facing = float(np.arctan2(2 * (q[3] * q[6] + q[4] * q[5]), 1 - 2 * (q[5] ** 2 + q[6] ** 2)))
    direction = command[:2] / speed if speed > 1e-8 else np.array([np.cos(facing), np.sin(facing)])
    heading = facing + float(command[2]) * 0.4
    original = dict(
        context_mujoco_qpos=past_context_at_30hz(raw["canonical_qpos"], frame),
        target_vel=np.array([speed], dtype=np.float32),
        mode=np.array([0 if speed < 0.02 else 1], dtype=np.int64),
        movement_direction=np.array([[*direction, 0.0]], dtype=np.float32),
        facing_direction=np.array([[np.cos(heading), np.sin(heading), 0.0]], dtype=np.float32),
        random_seed=np.array([30300], dtype=np.int64),
        has_specific_target=np.zeros((1, 1), dtype=np.int64),
        specific_target_positions=np.zeros((1, 4, 3), dtype=np.float32),
        specific_target_headings=np.zeros((1, 4), dtype=np.float32),
        allowed_pred_num_tokens=np.array([[1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]], dtype=np.int64),
        height=np.array([-1.0], dtype=np.float32),
    )
    feeds = dict(
        original=original,
        sham=apply_waypoint_intent(original, positive, activate=False),
        goal=apply_waypoint_intent(original, positive, activate=True),
        opposite=apply_waypoint_intent(original, negative, activate=True),
    )
    world_path = Path(spec["world"])
    world = mujoco.MjModel.from_binary_path(str(world_path))
    limits = world.jnt_range[[world.joint(n).id for n in G1_DDS_JOINT_NAMES]].copy()
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            input_file_hashes=pins,
            measured_frame=frame,
            history_frames_float=[55.0, 56.0 + 2.0 / 3.0, 58.0 + 1.0 / 3.0, 60.0],
            context_resampling="past_only_linear_with_sign_continuous_quaternion_nlerp_50_to_30Hz",
            positive_intent=positive.report(),
            negative_control_intent=negative.report(),
            maximum_new_neural_calls=8,
            new_native_steps=0,
            optimizer_steps=0,
            goal_is_current_observation_not_future_measured_body=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    executed = 0
    results = {}
    try:
        for repeat in (0, 1):
            options = ort.SessionOptions()
            options.intra_op_num_threads = 1
            options.inter_op_num_threads = 1
            session = ort.InferenceSession(
                str(model_file), sess_options=options, providers=["CPUExecutionProvider"]
            )
            if set(n.name for n in session.get_inputs()) != set(original):
                raise ValueError("exact twelve-input frozen planner contract required")
            for name, feed in feeds.items():
                folder = a.output / f"pass{repeat}-{name}"
                folder.mkdir()
                np.savez_compressed(folder / "inputs.npz", **feed)
                executed += 1
                output, count = session.run(None, feed)
                np.savez_compressed(folder / "outputs.npz", output=output, count=count)
                diagnostic = validate_planner_output(output, count, limits)
                if repeat:
                    old, oc = results[name]
                    if not np.array_equal(output, old) or not np.array_equal(count, oc):
                        raise ValueError(
                            "separate ONNX-session same-input mathematical repeat differs"
                        )
                else:
                    results[name] = (output.copy(), count.copy())
                write_once(
                    folder / "complete.json",
                    dict(
                        **diagnostic,
                        inputs_hash=digest(folder / "inputs.npz"),
                        outputs_hash=digest(folder / "outputs.npz"),
                    ),
                )
            del session
        baseline, sham, yes, no = [results[k] for k in ("original", "sham", "goal", "opposite")]
        ignored = np.array_equal(baseline[0], sham[0]) and np.array_equal(baseline[1], sham[1])
        consumed = not (np.array_equal(sham[0], yes[0]) and np.array_equal(sham[1], yes[1]))
        valid = {k: validate_planner_output(*x, limits) for k, x in results.items()}
        delta = goal - q[:2]
        if np.linalg.norm(delta) <= 1e-6:
            raise ValueError("nonzero known goal contrast required")
        contrast = float(
            np.dot(
                np.asarray(valid["goal"]["final_root_xyz_m"][:2])
                - valid["opposite"]["final_root_xyz_m"][:2],
                delta / np.linalg.norm(delta),
            )
        )
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="FROZEN_PLANNER_WAYPOINT_INPUT_PROBED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                new_neural_calls=executed,
                new_native_steps=0,
                results=valid,
                flag0_sham_exact=bool(ignored),
                waypoint_flag_consumed=bool(consumed),
                goal_vs_opposite_projected_root_contrast_m=contrast,
                all_outputs_within_original_joint_limits=all(
                    d["reference_within_native_joint_limits"] for d in valid.values()
                ),
                nn_input_feature_probe_not_dynamic_admission=True,
                physics_success_proven=False,
                optimizer_steps=0,
                goal_complete=False,
                promotion_authorized=False,
                hardware_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failure.json",
            dict(
                error=repr(error),
                attempted_neural_calls=executed,
                new_native_steps=0,
                optimizer_steps=0,
                automatic_retry_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
