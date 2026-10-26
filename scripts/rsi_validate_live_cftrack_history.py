"""Compare owned live five-frame capture to previously audited scratch histories."""

import argparse
import atexit
import json
import sys
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--history", type=Path, required=True)
    p.add_argument("--history-hash", required=True)
    p.add_argument("--history-review", type=Path, required=True)
    p.add_argument("--history-review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete, hist, review = (
        a.source / "complete.json",
        a.history / "complete.json",
        a.history_review,
    )
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (complete, hist, review))
        or digest(complete) != a.source_hash
        or digest(hist) != a.history_hash
        or digest(review) != a.history_review_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output, exact lineage and isolated import required")
    source, candidate, checked = (json.loads(x.read_text()) for x in (complete, hist, review))
    if (
        source["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or source["controls"] != 157
        or candidate["status"] != "MEASURED_CFTRACK_HISTORY_CANDIDATES_NOT_INSTALLED"
        or candidate["source_hash"] != a.source_hash
        or checked["status"] != "MEASURED_CFTRACK_HISTORY_REBUILT_EXACT_NOT_INSTALLED"
        or checked["source_hash"] != a.history_hash
    ):
        raise ValueError("exact source and independently read back history required")
    trace = a.source / source["artifact"]["path"]
    histories = a.history / candidate["artifact"]["path"]
    if (
        trace.is_symlink()
        or trace.parent != a.source
        or digest(trace) != source["artifact"]["hash"]
        or histories.is_symlink()
        or histories.parent != a.history
        or digest(histories) != candidate["artifact"]["hash"]
    ):
        raise ValueError("exact confined source and history artifacts required")
    pins = dict(candidate["input_file_hashes"])
    pins.update(
        {
            str(complete): a.source_hash,
            str(hist): a.history_hash,
            str(review): a.history_review_hash,
            str(trace): source["artifact"]["hash"],
            str(histories): candidate["artifact"]["hash"],
        }
    )
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    if not (len(scenes) == len(models) == len(configs) == len(external_files) == 1):
        raise ValueError("pinned scene, velocity codec and scratch CFtrack source required")
    external = external_files[0].parents[2]
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_action_history import describe_previous_target_encoding
    from rosclaw_soccer.rsi.cftrack_history_observation import (
        assemble_current_ball_observation,
        flatten_five_frame_history,
    )
    from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
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
        raise ValueError("exact native model required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(trace, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in f.files}
    with np.load(histories, allow_pickle=False) as f:
        expected = {k: f[k].copy() for k in f.files}
    stream = ContinuousVelocityStream(
        models[0],
        configs[0],
        expected_model_hash=pins[str(models[0])],
        expected_config_hash=pins[str(configs[0])],
        horizon_controls=301,
    )
    data = mujoco.MjData(model)
    data.qpos[:], data.qvel[:] = old["initial_qpos"], old["initial_qvel"]
    mujoco.mj_forward(model, data)
    pd = NativePDParameters(
        np.array(stream.params.stiffness),
        np.array(stream.params.damping),
        0.85 * np.array(G1_HARD_TORQUE_LIMITS),
    )
    world = SharedNativeMotorWorld(model, data, players={"a/": pd}, ball_body_name="soccer_ball")
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    policy = OmniContact(
        StateAndCmd(29),
        PolicyOutput(29),
        onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
    )
    if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
        raise ValueError("canonical scratch FK joint order required")
    policy.task, policy.reference_source, policy.active_object_name = "kickball", "CFgen", "ball"
    policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
    policy.bbox_scale = policy.box_dims * 2
    policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            history_review_hash=a.history_review_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            maximum_controls=82,
            maximum_native_steps=820,
            real_owned_boundary_capture=True,
            CFtrack_inference_or_history_install=False,
            no_body_or_ball_reset=True,
            hardware_authorized=False,
        ),
    )
    raw = {k: [] for k in ("qpos", "qvel", "torque", "ball_forces")}
    targets, observations, actions, results = [], [], [], []

    def retain_partial():
        if (a.output / "complete.json").exists():
            return
        partial = {k: np.concatenate(v) for k, v in raw.items() if v}
        if targets:
            partial.update(
                joint_targets=np.stack(targets),
                neural_obs=np.stack(observations),
                neural_actions=np.stack(actions),
            )
        path = a.output / "partial.npz"
        np.savez_compressed(path, **partial)
        write_once(
            a.output / "failed.json",
            dict(
                status="PARTIAL_NO_AUTOMATIC_RETRY",
                controls_recorded=len(targets),
                native_steps_recorded=len(partial.get("qpos", [])),
                partial_hash=digest(path),
                hardware_authorized=False,
                goal_complete=False,
            ),
        )

    atexit.register(retain_partial)
    collectors = {57: CFtrackLiveHistory(world, "a/"), 82: CFtrackLiveHistory(world, "a/")}
    for frame in range(82):
        q, v = lo.canonical(data)
        command = np.array([0.0, 0.25 if 25 <= frame < 57 else 0.0, 0.0], np.float32)
        target = stream.propose(q, v, lo.angular_velocity_body(data), command, frame=frame)
        result = world.advance({"a/": target})
        for key in raw:
            raw[key].append(result[key])
        targets.append(target.copy())
        observations.append(stream.last_observation.copy())
        actions.append(stream.last_action.copy())
        if (
            any(not np.array_equal(result[k], old[k][frame * 10 : (frame + 1) * 10]) for k in raw)
            or not np.array_equal(target, old["joint_targets"][frame])
            or not np.array_equal(stream.last_observation, old["neural_obs"][frame])
            or not np.array_equal(stream.last_action, old["neural_actions"][frame])
        ):
            raise ValueError(f"live original velocity trace mismatch at {frame}")
        boundary = frame + 1
        for entry, collector in collectors.items():
            if entry - 4 <= boundary <= entry:
                collector.capture()
            if boundary == entry:
                before = snapshot(data)
                measured = collector.at_current_boundary()
                rows = []
                for sample in measured:
                    q2, v2 = np.asarray(sample.canonical_qpos), np.asarray(sample.canonical_qvel)
                    s = policy.state_cmd
                    s.q, s.dq = q2[7:36].copy(), v2[6:35].copy()
                    s.base_pos, s.base_quat = q2[:3].copy(), q2[3:7].copy()
                    s.ang_vel = v2[3:6].copy()
                    w, x, y, z = s.base_quat
                    s.gravity_ori = np.array(
                        [2 * (-z * x + w * y), -2 * (z * y + w * x), 1 - 2 * (w * w + z * z)],
                        np.float32,
                    )
                    s.obj_pos, s.obj_quat = (
                        np.asarray(sample.cached_ball_position),
                        np.asarray(sample.cached_ball_quaternion),
                    )
                    policy._get_fk_info()
                    encoded = describe_previous_target_encoding(
                        np.asarray(sample.previous_executed_target),
                        policy.default_angles_lab,
                        policy.action_scale_lab,
                        policy.mj2lab,
                        policy.lab2mj,
                    )
                    policy.action = encoded["implied_action_lab"].copy()
                    rows.append(assemble_current_ball_observation(policy))
                assembled = np.stack(rows)
                if not np.array_equal(
                    assembled, expected[f"entry{entry}_observations"]
                ) or not np.array_equal(
                    flatten_five_frame_history(assembled[:4], assembled[4]),
                    expected[f"entry{entry}_flattened"],
                ):
                    raise ValueError(f"live owned history disagrees with scratch entry {entry}")
                after = snapshot(data)
                if any(not np.array_equal(before[k], after[k]) for k in before):
                    raise ValueError("capture or scratch feature computation changed live physics")
                results.append(
                    dict(
                        entry_control=entry,
                        sample_controls=[s.control for s in measured],
                        scratch_history_exact=True,
                        live_physics_unchanged=True,
                    )
                )
        if world.failed:
            raise ValueError("live world failed during declared capture")
    path = a.output / "live.npz"
    np.savez_compressed(
        path,
        **{k: np.concatenate(v) for k, v in raw.items()},
        joint_targets=np.stack(targets),
        neural_obs=np.stack(observations),
        neural_actions=np.stack(actions),
    )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="LIVE_MEASURED_CFTRACK_HISTORY_EXACT_NOT_INSTALLED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            history_review_hash=a.history_review_hash,
            input_file_hashes=pins,
            records=results,
            controls=82,
            native_steps=820,
            velocity_neural_calls=stream.neural_calls,
            cftrack_neural_calls=0,
            artifact=dict(path=path.name, hash=digest(path)),
            history_installed=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    atexit.unregister(retain_partial)


if __name__ == "__main__":
    main()
