"""Install two measured histories at true live boundaries; no kick inference."""

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
    p.add_argument("--live-review", type=Path, required=True)
    p.add_argument("--live-review-hash", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    complete, hist, live = a.source / "complete.json", a.history / "complete.json", a.live_review
    if (
        a.output.exists()
        or any(x.is_symlink() for x in (complete, hist, live))
        or digest(complete) != a.source_hash
        or digest(hist) != a.history_hash
        or digest(live) != a.live_review_hash
        or any(n in sys.modules for n in ("common", "policy", "FSM"))
    ):
        raise ValueError("unused output and exact isolated source lineage required")
    source, history, prior = (json.loads(x.read_text()) for x in (complete, hist, live))
    if (
        source["status"] != "WARM_LEFT_SETTLING_MEASURED_NOT_KICK"
        or history["status"] != "MEASURED_CFTRACK_HISTORY_CANDIDATES_NOT_INSTALLED"
        or prior["status"] != "LIVE_MEASURED_CFTRACK_HISTORY_EXACT_NOT_INSTALLED"
        or history["source_hash"] != a.source_hash
        or prior["source_hash"] != a.source_hash
        or prior["history_hash"] != a.history_hash
        or prior["controls"] != 82
    ):
        raise ValueError("exact previously validated actual source and scratch history required")
    trace, histories = (
        a.source / source["artifact"]["path"],
        a.history / history["artifact"]["path"],
    )
    if (
        trace.is_symlink()
        or trace.parent != a.source
        or digest(trace) != source["artifact"]["hash"]
        or histories.is_symlink()
        or histories.parent != a.history
        or digest(histories) != history["artifact"]["hash"]
    ):
        raise ValueError("exact confined source and scratch artifacts required")
    pins = dict(prior["input_file_hashes"])
    pins.update(
        {
            str(complete): a.source_hash,
            str(hist): a.history_hash,
            str(live): a.live_review_hash,
            str(trace): source["artifact"]["hash"],
            str(histories): history["artifact"]["hash"],
        }
    )
    source_pins(pins)
    check_pins(pins)
    scenes = [Path(x) for x in pins if x.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(x) for x in pins if x.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(x) for x in pins if x.endswith("/velocity/v0/params/deploy.yaml")]
    external_files = [Path(x) for x in pins if x.endswith("/policy/omnicontact/OmniContact.py")]
    if not (len(scenes) == len(models) == len(configs) == len(external_files) == 1):
        raise ValueError("one pinned scene, velocity codec and scratch policy required")
    external = external_files[0].parents[2]
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.cftrack_live_history import CFtrackLiveHistory
    from rosclaw_soccer.rsi.cftrack_measured_history_install import install_owned_cftrack_history
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_omnicontact_player import SharedOmniContactPlayer
    from rosclaw_soccer.rsi.shared_target_forecast import compiled_model_hash

    model = mujoco.MjModel.from_xml_string(
        shared_native_scene(scenes[0].read_text(), prefixes=("a/",))
    )
    if (
        compiled_model_hash(model)
        != "sha256:115bfd4a136b66903a54bc2807ec728cc2fa251d3e782f1906f352b45199568d"
    ):
        raise ValueError("exact original SIM body required")
    lo = NativeVelocityLayout(model, robot_prefix="a/", ball_body_name="soccer_ball")
    with np.load(trace, allow_pickle=False) as f:
        old = {k: f[k].copy() for k in f.files}
    with np.load(histories, allow_pickle=False) as f:
        expected = {k: f[k].copy() for k in f.files}
    sys.path.insert(0, str(external))
    from common.ctrlcomp import PolicyOutput, StateAndCmd
    from policy.omnicontact.OmniContact import OmniContact

    a.output.mkdir()
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            live_review_hash=a.live_review_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            entry_controls=[57, 82],
            maximum_velocity_calls=139,
            maximum_native_steps=1390,
            cftrack_neural_calls=0,
            no_body_or_ball_reset=True,
            software_pd_switch_explicit=True,
            cftrack_reference_not_qualified=True,
            no_kick=True,
            hardware_authorized=False,
        ),
    )
    results, partial_raw = [], {}
    counts = {"controls": 0, "native_steps": 0}

    def retain_partial():
        if (a.output / "complete.json").exists():
            return
        path = a.output / "partial.npz"
        np.savez_compressed(path, **{k: np.concatenate(v) for k, v in partial_raw.items() if v})
        write_once(
            a.output / "failed.json",
            dict(
                status="PARTIAL_NO_AUTOMATIC_RETRY",
                **counts,
                partial_hash=digest(path),
                hardware_authorized=False,
                goal_complete=False,
            ),
        )

    atexit.register(retain_partial)
    for entry in (57, 82):
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
        effort = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
        pd = NativePDParameters(
            np.array(stream.params.stiffness), np.array(stream.params.damping), effort
        )
        world = SharedNativeMotorWorld(
            model, data, players={"a/": pd}, ball_body_name="soccer_ball"
        )
        history_collector = CFtrackLiveHistory(world, "a/")
        partial_raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
        for frame in range(entry):
            q, v = lo.canonical(data)
            command = np.array([0.0, 0.25 if 25 <= frame < 57 else 0.0, 0.0], np.float32)
            target = stream.propose(q, v, lo.angular_velocity_body(data), command, frame=frame)
            result = world.advance({"a/": target})
            for key in partial_raw:
                partial_raw[key].append(result[key])
            counts["controls"] += 1
            counts["native_steps"] += len(result["qpos"])
            if (
                any(
                    not np.array_equal(result[k], old[k][frame * 10 : (frame + 1) * 10])
                    for k in partial_raw
                )
                or not np.array_equal(target, old["joint_targets"][frame])
                or not np.array_equal(stream.last_observation, old["neural_obs"][frame])
                or not np.array_equal(stream.last_action, old["neural_actions"][frame])
            ):
                raise ValueError(
                    f"actual source trajectory mismatch at entry {entry} control {frame}"
                )
            if frame + 1 >= entry - 4:
                history_collector.capture()
            if world.failed:
                raise ValueError("live body failed before declared history installation")
        before = snapshot(data)
        cached_ball = data.xpos[lo.ball].copy(), data.xquat[lo.ball].copy()
        policy = OmniContact(
            StateAndCmd(29),
            PolicyOutput(29),
            onnx_path=str(external / "policy/omnicontact/model/kick_50k.onnx"),
        )
        if tuple(policy.kinematics.joint_names) != tuple(G1_DDS_JOINT_NAMES):
            raise ValueError("canonical scratch FK order required")
        policy.task, policy.reference_source, policy.active_object_name = (
            "kickball",
            "CFgen",
            "ball",
        )
        policy.box_dims = policy.ball_dims = np.full(3, 0.11, np.float32)
        policy.bbox_scale = policy.box_dims * 2
        policy.bbox_offsets_scaled = policy.bbox_offsets * policy.bbox_scale.reshape(1, 3)
        kp, kd = policy.kps_lab[policy.lab2mj], policy.kds_lab[policy.lab2mj]
        switch = world.switch_pd_profile("a/", NativePDParameters(kp, kd, effort))
        player = SharedOmniContactPlayer(
            world, "a/", policy, expected_policy_type=OmniContact, goal=np.array([5.0, 0.0, 0.11])
        )
        receipt = install_owned_cftrack_history(player, history_collector)
        if (
            not np.array_equal(
                policy.obs_history_buffer[1:], expected[f"entry{entry}_observations"][:4]
            )
            or not np.array_equal(policy.action, expected[f"entry{entry}_observations"][4, 79:108])
            or np.any(policy.obs_history_buffer[0])
            or policy.counter_step != 0
            or player.commits != 0
            or player.pending is not None
        ):
            raise ValueError("installed history must match qualified actual scratch observation")
        after = snapshot(data)
        if (
            any(not np.array_equal(before[k], after[k]) for k in before)
            or not np.array_equal(cached_ball[0], data.xpos[lo.ball])
            or not np.array_equal(cached_ball[1], data.xquat[lo.ball])
        ):
            raise ValueError("gain switch or history install modified physical state or ball cache")
        trace_path = a.output / f"entry{entry}.npz"
        np.savez_compressed(trace_path, **{k: np.concatenate(v) for k, v in partial_raw.items()})
        results.append(
            dict(
                entry_control=entry,
                source_prefix_exact=True,
                installed_rows_exact=True,
                physics_unchanged=True,
                pd_switch_physics_unchanged=switch["physics_unchanged"],
                history_receipt=receipt,
                artifact=dict(path=trace_path.name, hash=digest(trace_path)),
            )
        )
    check_pins(pins)
    clean(a.expected_head)
    write_once(
        a.output / "complete.json",
        dict(
            status="LIVE_CFTRACK_HISTORY_INSTALLED_NOT_EXECUTED",
            source_head=a.expected_head,
            source_hash=a.source_hash,
            history_hash=a.history_hash,
            live_review_hash=a.live_review_hash,
            input_file_hashes=pins,
            records=results,
            **counts,
            velocity_neural_calls=139,
            cftrack_neural_calls=0,
            task_reference_qualified=False,
            hardware_authorized=False,
            promotion_authorized=False,
            goal_complete=False,
        ),
    )
    atexit.unregister(retain_partial)


if __name__ == "__main__":
    main()
