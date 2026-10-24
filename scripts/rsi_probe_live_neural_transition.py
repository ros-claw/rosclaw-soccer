"""SIM live/live recovery blend on fixed team-scene starts, not new learning.

Keep both private neural histories causal through ACTUALLY executed targets.
No held action, body reset, ball rewrite, contact tuning or old-exam regrading.
Both targets and PD profiles interpolate over a predeclared20-control window.
"""

import argparse
import contextlib
import io
import json
import math
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "own", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "own-hash", "expected-head"):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--contact-release", action="store_true")
    parser.add_argument("--aim-y", type=float, default=0.0)
    parser.add_argument("--single-player-only", action="store_true")
    a = parser.parse_args()
    if not math.isfinite(a.aim_y) or abs(a.aim_y) > 1.2:
        parser.error("finite qualified neural aim input in [-1.2,1.2] required")
    if (a.single_player_only or a.aim_y != 0.0) and not (
        a.single_player_only and a.contact_release
    ):
        parser.error(
            "new aim reachability protocol requires contact release and single-player-only"
        )
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if (
        a.output.exists()
        or digest(a.source / "complete.json") != a.source_hash
        or digest(a.own / "complete.json") != a.own_hash
    ):
        raise ValueError("unused output and exact closed source/own pair required")
    source = json.loads((a.source / "complete.json").read_text())
    own = json.loads((a.own / "complete.json").read_text())
    if (
        source["status"] != "ROBONALDO_NEURAL_VELOCITY_HANDOFF_DIAGNOSED_NOT_PROMOTED"
        or own["status"] != "ROBONALDO_VELOCITY_RECOVERY_MANUAL_CLOCK_REVIEWED_NOT_PROMOTED"
        or own["source_hash"] != a.source_hash
        or not own["all_exact"]
        or len(source["records"]) != 12
        or len(own["records"]) != 12
        or not all(row["healthy_full_horizon"] for row in source["records"])
    ):
        raise ValueError("complete healthy12, independently replayed predecessor required")
    pins = dict(source["input_file_hashes"])
    for key, value in own["input_file_hashes"].items():
        if key in pins and pins[key] != value:
            raise ValueError("conflicting predecessor source")
        pins[key] = value
    pins[str(a.source / "complete.json")], pins[str(a.own / "complete.json")] = (
        a.source_hash,
        a.own_hash,
    )
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.live_neural_transition import (
        ContactReleaseGate,
        blend_live_targets,
        quintic_weight,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.training.directed_plane_crossing import first_directed_plane_crossing

    roots = {Path(key).parents[2] for key in pins if key.endswith("/policy/robonaldo/FreeKick.py")}
    scenes = [Path(key) for key in pins if key.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(key) for key in pins if key.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(key) for key in pins if key.endswith("/velocity/v0/params/deploy.yaml")]
    if len(roots) != 1 or any(len(values) != 1 for values in (scenes, models, configs)):
        raise ValueError("unique qualified skills and exact team scene required")
    root = roots.pop()
    state_type, output_type, policy_type, mapping = _load_robonaldo(root)
    with np.load(root / "policy/robonaldo/model/freekick_motion.npz") as motion:
        pos, quat, joints = (
            motion["body_pos_w"][0, 0].copy(),
            motion["body_quat_w"][0, 0].copy(),
            motion["joint_pos"][0][mapping].copy(),
        )

    def velocity():
        return ContinuousVelocityStream(
            models[0],
            configs[0],
            expected_model_hash=pins[str(models[0])],
            expected_config_hash=pins[str(configs[0])],
            horizon_controls=652,
        )

    class RecordSession:
        def __init__(self, session):
            self.session, self.calls = session, 0
            self.obs, self.action = np.zeros(547, np.float32), np.zeros(29, np.float32)

        def run(self, names, inputs):
            obs = inputs["obs"]
            out = self.session.run(names, inputs)
            if (
                obs.shape != (1, 547)
                or obs.dtype != np.float32
                or len(out) != 1
                or out[0].shape != (1, 29)
                or not np.isfinite(obs).all()
                or not np.isfinite(out[0]).all()
            ):
                raise ValueError("actual finite547/29 neural tensors required")
            self.obs, self.action = obs[0].copy(), out[0][0].copy()
            self.calls += 1
            return out

    def observe(w, state):
        layout = w.layouts["a/"]
        q, v = layout.canonical(w.data)
        torso = w.model.body("a/torso_link").id
        state.q, state.dq, state.tau_est = (
            q[7:36].copy(),
            v[6:35].copy(),
            w.data.ctrl[layout.ai].copy(),
        )
        state.root_lin_vel_b, state.root_ang_vel_b = v[:3].copy(), v[3:6].copy()
        state.torso_pos_w, state.torso_quat_w = (
            w.data.xpos[torso].copy(),
            w.data.xquat[torso].copy(),
        )
        state.pelvis_pos_w, state.pelvis_quat_w = q[:3].copy(), q[3:7].copy()
        state.ball_pos_w, state.ball_vel_w, state.ball_valid = (
            w.data.xpos[layout.ball].copy(),
            v[35:38].copy(),
            True,
        )

    starts = [(x, y) for x in (0.97, 1.03) for y in (-0.03, 0.03)]
    prepared = []
    arms = (
        (("single", ("a/",)),)
        if a.single_player_only
        else (("single", ("a/",)), ("heterogeneous", ("a/", "b/")))
    )
    for arm, prefixes in arms:
        model = mujoco.MjModel.from_xml_string(
            shared_native_scene(scenes[0].read_text(), prefixes=prefixes)
        )
        for index, (bx, by) in enumerate(starts):
            data = mujoco.MjData(model)
            stream, other = velocity(), velocity() if len(prefixes) == 2 else None
            state, output = state_type(29), output_type(29)
            with contextlib.redirect_stdout(io.StringIO()):
                kick = policy_type(state, output)
            recorder = RecordSession(kick.ort_session)
            kick.ort_session = recorder
            kick.target_pos_w = np.array([5.0, a.aim_y, 0.11], np.float32)
            cap = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
            oldpd = NativePDParameters(np.array(kick.kps), np.array(kick.kds), cap).owned_copy()
            newpd = NativePDParameters(
                np.array(stream.params.stiffness), np.array(stream.params.damping), cap
            ).owned_copy()
            params = {"a/": oldpd}
            for number, prefix in enumerate(prefixes):
                layout = NativeVelocityLayout(
                    model,
                    robot_prefix=prefix,
                    ball_body_name="soccer_ball",
                    player_count=len(prefixes),
                )
                data.qpos[layout.rq : layout.rq + 3] = pos + np.array([0.0, number * 8.0, 0.0])
                data.qpos[layout.rq + 3 : layout.rq + 7] = quat
                data.qpos[layout.qi] = (
                    joints if prefix == "a/" else np.array(stream.params.defaults)
                )
                if prefix == "b/":
                    params[prefix] = newpd
            data.qpos[layout.bq : layout.bq + 3], data.qpos[layout.bq + 3 : layout.bq + 7] = (
                (bx, by, 0.11),
                (1.0, 0.0, 0.0, 0.0),
            )
            mujoco.mj_forward(model, data)
            world = SharedNativeMotorWorld(
                model, data, players=params, ball_body_name="soccer_ball"
            )
            observe(world, state)
            with contextlib.redirect_stdout(io.StringIO()):
                kick.enter()
            prepared.append(
                (arm, index, world, state, output, kick, recorder, stream, other, oldpd, newpd)
            )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol=(
                "robonaldo.neural_aim_reachability.v1"
                if a.single_player_only
                else (
                    "robonaldo.contact_release_blend.v1"
                    if a.contact_release
                    else "robonaldo.live_neural_blend.v1"
                )
            ),
            source_head=a.expected_head,
            source_hash=a.source_hash,
            own_hash=a.own_hash,
            input_file_hashes=pins,
            starts=starts,
            worlds=4 if a.single_player_only else 8,
            native_step_ceiling=26080 if a.single_player_only else 52160,
            neural_aim_input_y_m=a.aim_y,
            evaluation_target_world_m=[5.0, 0.0, 0.11],
            aim_input_is_not_evaluation_target=True,
            horizon_controls=652,
            start_control=None if a.contact_release else 285,
            duration_controls=20,
            outgoing_active_until_control_exclusive=None if a.contact_release else 305,
            retired_outgoing_zero_padding_not_training_data=True,
            contact_release=a.contact_release,
            release_condition=dict(clear_controls=3, distance_m_gt=0.35, separation_mps_gt=0.5),
            live_outgoing_proposals=True,
            live_incoming_proposals=True,
            actual_history_both_ABIs=True,
            same_effort_caps=True,
            physics_retuned=False,
            optimizer_updates=0,
            hardware_authorized=False,
            publicity_rights_not_established=True,
            promotion_authorized=False,
            new_protocol_not_old_M0=True,
        ),
    )
    results, steps = [], 0
    try:
        for arm, index, w, state, output, kick, recorder, stream, other, oldpd, newpd in prepared:
            raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
            controls = {
                key: []
                for key in (
                    "targets",
                    "blend_weight",
                    "kp",
                    "kd",
                    "kick_observation",
                    "kick_raw_action",
                    "kick_inferred_mask",
                    "velocity_observation",
                    "velocity_proposed_action",
                    "kick_proposed_target",
                    "velocity_proposed_target",
                    "release_distance_m",
                    "release_separation_mps",
                    "release_foot_contact",
                    "release_clear_controls",
                )
            }
            prefixes = tuple(w.layouts)
            foot = None
            gate = ContactReleaseGate()
            transition_start = None if a.contact_release else 285
            previous_weight = 0.0
            for frame in range(652):
                distance, speed, foot_contact = 0.0, 0.0, False
                if a.contact_release:
                    layout = w.layouts["a/"]
                    foot_body = w.model.body("a/right_ankle_roll_link").id
                    separation = w.data.xpos[layout.ball] - w.data.xpos[foot_body]
                    distance = float(np.linalg.norm(separation))
                    jac = np.zeros((3, w.model.nv))
                    mujoco.mj_jacBody(w.model, w.data, jac, None, foot_body)
                    relative_velocity = w.data.qvel[layout.bv : layout.bv + 3] - jac @ w.data.qvel
                    speed = float(np.dot(separation, relative_velocity) / max(1e-12, distance))
                    foot_contact = "a/right_ankle_roll_link" in w.last_ball_contact_bodies["a/"]
                    transition_start = gate.observe(
                        frame,
                        foot_contact=foot_contact,
                        distance_m=distance,
                        separation_mps=speed,
                    )
                weight = (
                    0.0
                    if transition_start is None
                    else quintic_weight(frame, start=transition_start, duration=20)
                )
                layout = w.layouts["a/"]
                q, v = layout.canonical(w.data)
                incoming = stream.propose(
                    q,
                    v,
                    layout.angular_velocity_body(w.data),
                    np.zeros(3, np.float32),
                    frame=stream.next_frame,
                )
                outgoing = np.zeros(29, np.float64)
                old_calls = recorder.calls
                if weight < 1.0:
                    observe(w, state)
                    with contextlib.redirect_stdout(io.StringIO()):
                        kick.run()
                    outgoing = np.asarray(output.actions, np.float64).copy()
                    if not np.array_equal(output.kps, oldpd.stiffness) or not np.array_equal(
                        output.kds, oldpd.damping
                    ):
                        raise ValueError("original outgoing gains changed")
                target, pd = blend_live_targets(outgoing, incoming, oldpd, newpd, weight=weight)
                if weight > 0.0 and weight != previous_weight:
                    w.switch_pd_profile("a/", pd)
                previous_weight = weight
                proposals = {"a/": target}
                if other is not None:
                    other_layout = w.layouts["b/"]
                    oq, ov = other_layout.canonical(w.data)
                    proposals["b/"] = other.propose(
                        oq,
                        ov,
                        other_layout.angular_velocity_body(w.data),
                        np.zeros(3, np.float32),
                        frame=other.next_frame,
                    )
                inferred = recorder.calls > old_calls
                values = (
                    np.stack([proposals[prefix] for prefix in prefixes]),
                    weight,
                    pd.stiffness.copy(),
                    pd.damping.copy(),
                    recorder.obs.copy() if inferred else np.zeros(547, np.float32),
                    recorder.action.copy() if inferred else np.zeros(29, np.float32),
                    inferred,
                    stream.last_observation.copy(),
                    stream.last_action.copy(),
                    outgoing.copy(),
                    incoming.copy(),
                    distance,
                    speed,
                    foot_contact,
                    gate.clear_controls,
                )
                for key, value in zip(controls, values, strict=True):
                    controls[key].append(value)
                result = w.advance(proposals)
                steps += 10
                if weight < 1.0:
                    stream.last_action = policy_action_from_hg_target(
                        target,
                        np.array(stream.params.action_offset),
                        np.array(stream.params.action_scale),
                        np.arange(29),
                    )
                if 0.0 < weight < 1.0:
                    kick.last_action_il = policy_action_from_hg_target(
                        target, np.array(kick.default_q_mj), np.array(kick.action_scale_mj), mapping
                    )
                if foot is None and any(
                    "ankle_roll_link" in body for body in w.last_ball_contact_bodies["a/"]
                ):
                    foot = frame
                for key in raw:
                    raw[key].append(result[key])
                if w.failed:
                    break
            arrays = {key: np.concatenate(value) for key, value in raw.items()}
            arrays.update({key: np.asarray(value) for key, value in controls.items()})
            primary_layout = w.layouts["a/"]
            ball_positions = arrays["qpos"][:, primary_layout.bq : primary_layout.bq + 3]
            crossing = first_directed_plane_crossing(
                ball_positions[:, :2][:, None, :],
                origin=np.array([5.0, 0.0]),
                forward=np.array([1.0, 0.0]),
            )
            plane = None
            if bool(crossing.crossed[0]):
                ci, fraction = int(crossing.segment_index[0]), float(crossing.fraction[0])
                point = ball_positions[ci] + fraction * (
                    ball_positions[ci + 1] - ball_positions[ci]
                )
                plane = dict(
                    y_m=float(point[1]),
                    z_m=float(point[2]),
                    error_to_fixed_target_m=float(np.linalg.norm(point[1:] - [0.0, 0.11])),
                )
            artifact = a.output / f"{arm}-{index}.npz"
            np.savez_compressed(artifact, **arrays)
            baseline = next(
                row
                for row in source["records"]
                if row["arm"] == "actual-shadow" and row["index"] == index
            )
            prefix_exact = None
            if arm == "single" and a.aim_y == 0.0:
                with np.load(a.source / baseline["artifact"]["path"]) as ref:
                    prefix_end = (
                        min(285, transition_start if transition_start is not None else 285) * 10
                    )
                    prefix_exact = {
                        key: bool(np.array_equal(arrays[key][:prefix_end], ref[key][:prefix_end]))
                        for key in ("qpos", "qvel", "torque")
                    }
            start = transition_start if transition_start is not None else 285
            window = arrays["targets"][max(0, start - 1) : start + 22, 0]
            jump = float(np.max(np.abs(np.diff(window, axis=0)))) if len(window) > 1 else None
            row = dict(
                arm=arm,
                index=index,
                neural_aim_input_y_m=a.aim_y,
                actual_goal_plane=plane,
                illegal_body_ball_contact=bool(np.any(arrays["ball_forces"][:, 0, 2] > 1.0)),
                controls=w.control_steps,
                native_steps=w.native_steps,
                healthy_full_horizon=w.control_steps == 652 and not w.failed,
                body_failed=result["body_failed"],
                actual_foot_contact_control=foot,
                transition_start=transition_start,
                contact_release_gate=a.contact_release,
                prehandoff_prefix_exact=prefix_exact,
                max_target_step_in_transition_window_rad=jump,
                actual_kick_NN_calls=recorder.calls,
                actual_velocity_NN_calls=stream.neural_calls,
                other_NN_calls=other.neural_calls if other is not None else 0,
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            results.append(row)
            write_once(a.output / f"record-{arm}-{index}.json", row)
            print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="LIVE_NEURAL_TRANSITION_DIAGNOSED_NOT_PROMOTED",
                records=results,
                native_steps=steps,
                input_file_hashes=pins,
                source_head=a.expected_head,
                optimizer_updates=0,
                independent_manual_replay=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(
                error=repr(error),
                completed=len(results),
                known_completed_control_native_steps=steps,
                incomplete_control_steps_unknown=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
