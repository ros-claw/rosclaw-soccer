"""Bounded SIM walk-to-whole-body-kick history ablation, not a trained teacher."""

import argparse
import contextlib
import io
import json
from collections import deque
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    path = a.source / "complete.json"
    if a.output.exists() or path.is_symlink() or digest(path) != a.source_hash:
        raise ValueError("new output and exact closed physical source required")
    source = json.loads(path.read_text())
    if (
        source["status"] != "NATIVE8_B_AND_B_PLUS_C_CONTACT_PILOT_MEASURED_NOT_LEARNED"
        or source["promotion_authorized"] is not False
        or len(source["results"]) != 5
    ):
        raise ValueError("complete unpromoted native approach study required")
    pins = dict(source["input_file_hashes"])
    pins[str(path)] = a.source_hash
    source_pins(pins)
    check_pins(pins)

    def one(suffix):
        paths = [Path(k) for k in pins if k.endswith(suffix)]
        if len(paths) != 1:
            raise ValueError("one exact asset required: " + suffix)
        return paths[0]

    scene = one("/cached-standard/declared-scene.xml")
    model = one("/velocity/v0/exported/policy.onnx")
    config = one("/velocity/v0/params/deploy.yaml")
    root = one("/policy/robonaldo/FreeKick.py").parents[2]
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.kick_warmstart import rebase_kick_reference
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_kick_history import measured_kick_sample
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer

    state_type, output_type, policy_type, _ = _load_robonaldo(root)
    cap = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    arms = ("walk-only", "cold-reference", "measured-reference")
    speeds = (0.5, 1.0)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="soccer.native_walk_to_kick_measured_history.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            arms=arms,
            speeds_mps=speeds,
            worlds=12,
            horizon_controls=400,
            native_step_ceiling=48000,
            ball_initial_position=[3.0, 0.0, 0.11],
            ball_initial_velocity=[0.0, 0.0, 0.0],
            root_initial_position=[0.0, 0.0, 0.793],
            goal=[8.0, 0.0, 0.11],
            entry="first boundary50..250 with body ball x<=.9 and abs(y)<=.25; no prior contact",
            kick_reference_frame=0,
            kick_warmup_skipped_in_both_kick_arms=True,
            reference_rebased_in_both_kick_arms=True,
            actual_history_only_in_measured_arm=True,
            pd_switch="explicit original kick gains, unchanged .85 hard effort caps",
            torque_continuity_claimed=False,
            physics_retuned=False,
            optimizer_updates=0,
            physical_teacher_qualified=False,
            promotion_authorized=False,
            hardware_authorized=False,
        ),
    )
    records, originals, total = [], {}, 0
    try:
        for speed in speeds:
            for arm in arms:
                for repeat in range(2):
                    stream = ContinuousVelocityStream(
                        model,
                        config,
                        expected_model_hash=pins[str(model)],
                        expected_config_hash=pins[str(config)],
                        horizon_controls=400,
                    )
                    m = mujoco.MjModel.from_xml_string(
                        shared_native_scene(scene.read_text(), prefixes=("a/",))
                    )
                    d = mujoco.MjData(m)
                    lo = NativeVelocityLayout(m, robot_prefix="a/", ball_body_name="soccer_ball")
                    d.qpos[lo.rq : lo.rq + 7] = [0.0, 0.0, 0.793, 1.0, 0.0, 0.0, 0.0]
                    d.qpos[lo.qi] = np.asarray(stream.params.defaults)
                    d.qpos[lo.bq : lo.bq + 7] = [3.0, 0.0, 0.11, 1.0, 0.0, 0.0, 0.0]
                    mujoco.mj_forward(m, d)
                    vp = NativePDParameters(
                        np.array(stream.params.stiffness), np.array(stream.params.damping), cap
                    )
                    w = SharedNativeMotorWorld(
                        m, d, players={"a/": vp}, ball_body_name="soccer_ball"
                    )
                    history = deque(maxlen=5)
                    kick, entry, entry_receipt, pd_receipt = None, None, None, None
                    contacted = False
                    entry_context = None
                    raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
                    targets, commands, previous = [], [], np.zeros(3, np.float32)
                    for frame in range(400):
                        q, v = lo.canonical(d)
                        rot = d.xmat[lo.pelvis].reshape(3, 3)
                        delta = rot.T @ (d.xpos[lo.ball] - q[:3])
                        admit = (
                            entry is None
                            and not contacted
                            and 50 <= frame <= 250
                            and 0 < delta[0] <= 0.9
                            and abs(delta[1]) <= 0.25
                        )
                        if admit:
                            entry = frame
                            entry_context = dict(
                                canonical_qpos=q.tolist(),
                                canonical_qvel=v.tolist(),
                                measured_history=[s.numeric_record().tolist() for s in history],
                                no_prior_ball_contact=True,
                            )
                            if arm != "walk-only":
                                with contextlib.redirect_stdout(io.StringIO()):
                                    policy = policy_type(state_type(29), output_type(29))
                                kp = NativePDParameters(
                                    np.array(policy.kps), np.array(policy.kds), cap
                                )
                                pd_receipt = w.switch_pd_profile("a/", kp)
                                kick = SharedRoboNaldoPlayer(
                                    w, "a/", policy, goal=np.array([8.0, 0.0, 0.11])
                                )
                                if arm == "measured-reference":
                                    entry_receipt = kick.install_measured_entry(
                                        tuple(history), reference_frame=0
                                    )
                                else:
                                    rebase_kick_reference(policy, entry_frame=0)
                                    policy.time_step = policy.WARMUP_STEPS
                        yaw = float(np.arctan2(rot[1, 0], rot[0, 0]))
                        wanted = np.array([speed, 0.0, np.clip(-2 * yaw, -1, 1)], np.float32)
                        command = previous + np.clip(wanted - previous, -0.05, 0.05)
                        previous = command.copy()
                        if kick is None:
                            target = stream.propose(
                                q, v, lo.angular_velocity_body(d), command, frame=frame
                            )
                        else:
                            target = kick.propose()[0]
                        result = w.advance({"a/": target})
                        total += 10
                        if kick is not None:
                            kick.commit()
                        history.append(measured_kick_sample(w, "a/"))
                        contacted |= bool(w.last_ball_contact_bodies["a/"])
                        for key in raw:
                            raw[key].append(result[key])
                        targets.append(target.copy())
                        commands.append(command.copy())
                        if w.failed:
                            break
                    arrays = {key: np.concatenate(values) for key, values in raw.items()}
                    arrays.update(
                        joint_targets=np.stack(targets), approach_commands=np.stack(commands)
                    )
                    key = (speed, arm)
                    prefix_exact = None
                    if arm != "walk-only" and entry is not None:
                        with np.load(originals[(speed, "walk-only")]) as saved:
                            prefix_exact = {
                                k: bool(
                                    np.array_equal(arrays[k][: entry * 10], saved[k][: entry * 10])
                                )
                                for k in raw
                            }
                            prefix_exact["joint_targets"] = bool(
                                np.array_equal(
                                    arrays["joint_targets"][:entry], saved["joint_targets"][:entry]
                                )
                            )
                        if not all(prefix_exact.values()):
                            raise ValueError("walk-only actual prehandoff prefix mismatch")
                    exact = None
                    if repeat:
                        with np.load(originals[key]) as saved:
                            exact = {
                                k: bool(np.array_equal(v, saved[k])) for k, v in arrays.items()
                            }
                        if not all(exact.values()):
                            raise ValueError("fresh NN physical reexecution mismatch")
                    artifact = a.output / f"{speed}-{arm}-{repeat}.npz"
                    np.savez_compressed(artifact, **arrays)
                    if not repeat:
                        originals[key] = artifact
                    force = arrays["ball_forces"][:, 0]
                    feet = np.flatnonzero(force[:, :2].max(axis=1) > 1)
                    row = dict(
                        arm=arm,
                        speed_mps=speed,
                        repeat=repeat,
                        controls=w.control_steps,
                        native_steps=w.native_steps,
                        healthy_full_horizon=not w.failed and w.control_steps == 400,
                        entry_control=entry,
                        entry_context=entry_context,
                        prehandoff_walk_prefix_exact=prefix_exact,
                        handoff_executed=kick is not None,
                        measured_entry_receipt=entry_receipt,
                        pd_switch_physics_unchanged=pd_receipt["physics_unchanged"]
                        if pd_receipt
                        else None,
                        first_foot_contact_native_sample=int(feet[0]) if len(feet) else None,
                        foot_contact_native_samples=int(len(feet)),
                        nonfoot_contact_native_samples=int((force[:, 2] > 1).sum()),
                        nonfoot_peak_force_n=float(force[:, 2].max()),
                        final_ball_position=d.xpos[lo.ball].tolist(),
                        fresh_reexecution_exact=exact,
                        velocity_neural_calls=stream.neural_calls,
                        kick_neural_proposals=kick.proposals if kick else 0,
                        artifact=dict(path=artifact.name, hash=digest(artifact)),
                    )
                    records.append(row)
                    write_once(a.output / f"record-{speed}-{arm}-{repeat}.json", row)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status="WALK_TO_KICK_HISTORY_ABLATION_MEASURED_NOT_LEARNED",
                source_head=a.expected_head,
                input_file_hashes=pins,
                records=records,
                worlds=len(records),
                native_steps=total,
                optimizer_updates=0,
                physical_teacher_qualified=False,
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
                native_steps=total,
                completed_records=len(records),
                no_automatic_retry=True,
            ),
        )
        raise


if __name__ == "__main__":
    main()
