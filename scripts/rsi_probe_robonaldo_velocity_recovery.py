"""SIM paired skill-handoff probe; no learning, resets or physics retuning.

Existing four continuous-kick trials are the matched reference. Compare cold
versus actual-action-shadow velocity recovery, plus a heterogeneous two-player
arm where the off-ball player uses its own velocity network, never a far-ball
kick. The two-player arm is NOT a passing success or autonomous team game.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "velocity-model", "velocity-config", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-hash", "expected-head", "velocity-model-hash", "velocity-config-hash"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.source / "complete.json") != a.source_hash:
        raise ValueError("unused output and exact matched source required")
    reference = json.loads((a.source / "complete.json").read_text())
    if (
        reference["status"] != "ROBONALDO_SHARED_TEAM_SCENE_DIAGNOSED_NOT_PROMOTED"
        or len(reference["records"]) != 8
    ):
        raise ValueError("closed entire team-scene source required")
    pins = dict(reference["input_file_hashes"])
    pins[str(a.source / "complete.json")] = a.source_hash
    pins[str(a.velocity_model)] = a.velocity_model_hash
    pins[str(a.velocity_config)] = a.velocity_config_hash
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.paid_velocity_shadow import policy_action_from_hg_target
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.rsi.shared_robonaldo_player import SharedRoboNaldoPlayer

    roots = {Path(key).parents[2] for key in pins if key.endswith("/policy/robonaldo/FreeKick.py")}
    scenes = [Path(key) for key in pins if key.endswith("/cached-standard/declared-scene.xml")]
    if len(roots) != 1 or len(scenes) != 1:
        raise ValueError("one exact qualified policy root and team scene required")
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
            a.velocity_model,
            a.velocity_config,
            expected_model_hash=a.velocity_model_hash,
            expected_config_hash=a.velocity_config_hash,
            horizon_controls=652,
        )

    starts = [(x, y) for x in (0.97, 1.03) for y in (-0.03, 0.03)]
    arms = [("cold", ("a/",)), ("actual-shadow", ("a/",)), ("heterogeneous-shadow", ("a/", "b/"))]
    models = {
        name: mujoco.MjModel.from_xml_string(
            shared_native_scene(scenes[0].read_text(), prefixes=prefixes)
        )
        for name, prefixes in arms
    }
    prepared = []
    cap = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
    for name, prefixes in arms:
        for index, (bx, by) in enumerate(starts):
            m, stream = models[name], velocity()
            with contextlib.redirect_stdout(io.StringIO()):
                kick = policy_type(state_type(29), output_type(29))
            d = mujoco.MjData(m)
            vp = stream.params
            kp = NativePDParameters(np.array(kick.kps), np.array(kick.kds), cap)
            velp = NativePDParameters(np.array(vp.stiffness), np.array(vp.damping), cap)
            params = {"a/": kp}
            for number, prefix in enumerate(prefixes):
                layout = NativeVelocityLayout(
                    m, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=len(prefixes)
                )
                d.qpos[layout.rq : layout.rq + 3] = pos + np.array([0.0, number * 8.0, 0.0])
                d.qpos[layout.rq + 3 : layout.rq + 7] = quat
                d.qpos[layout.qi] = joints if prefix == "a/" else np.array(vp.defaults)
                if prefix == "b/":
                    params[prefix] = velp
            d.qpos[layout.bq : layout.bq + 3] = (bx, by, 0.11)
            d.qpos[layout.bq + 3 : layout.bq + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(m, d)
            w = SharedNativeMotorWorld(m, d, players=params, ball_body_name="soccer_ball")
            player = SharedRoboNaldoPlayer(w, "a/", kick, goal=np.array([5.0, 0.0, 0.11]))
            prepared.append(
                (name, index, w, player, stream, velocity() if len(prefixes) == 2 else None, velp)
            )
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="robonaldo.velocity_recovery.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            starts=starts,
            arms=[name for name, _ in arms],
            handoff_control=285,
            horizon_controls=652,
            native_step_ceiling=78240,
            worlds=12,
            unchanged_reference_source_hash=a.source_hash,
            physics_retuned=False,
            secondary_initial_joints="original_velocity_default",
            off_ball_command_body=[0.0, 0.0, 0.0],
            optimizer_updates=0,
            hardware_authorized=False,
            promotion_authorized=False,
            new_protocol_not_old_M0=True,
            publicity_rights_not_established=True,
        ),
    )
    results, steps = [], 0
    try:
        for name, index, w, kick, stream, other, velp in prepared:
            raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
            target_rows, obs_rows, action_rows, inferred_masks = [], [], [], []
            first_foot = None
            handoff = None
            prefixes = tuple(w.layouts)
            for frame in range(652):
                if frame == 285:
                    if first_foot is None:
                        raise ValueError("predeclared handoff requires actual earlier foot contact")
                    handoff = w.switch_pd_profile("a/", velp)
                proposals = {}
                al = w.layouts["a/"]
                if name != "cold" or frame >= 285:
                    q, v = al.canonical(w.data)
                    vt = stream.propose(
                        q,
                        v,
                        al.angular_velocity_body(w.data),
                        np.zeros(3, np.float32),
                        frame=stream.next_frame,
                    )
                    inferred = True
                else:
                    vt, inferred = None, False
                if frame < 285:
                    proposals["a/"] = kick.propose()[0]
                else:
                    proposals["a/"] = vt
                if other is not None:
                    bl = w.layouts["b/"]
                    q, v = bl.canonical(w.data)
                    proposals["b/"] = other.propose(
                        q,
                        v,
                        bl.angular_velocity_body(w.data),
                        np.zeros(3, np.float32),
                        frame=other.next_frame,
                    )
                obs_rows.append(stream.last_observation.copy())
                action_rows.append(stream.last_action.copy())
                inferred_masks.append(inferred)
                result = w.advance(proposals)
                steps += 10
                if frame < 285:
                    kick.commit()
                    if name != "cold":
                        stream.last_action = policy_action_from_hg_target(
                            w.last_applied_target("a/"),
                            np.array(stream.params.action_offset),
                            np.array(stream.params.action_scale),
                            np.arange(29),
                        )
                if first_foot is None and any(
                    "ankle_roll_link" in body for body in w.last_ball_contact_bodies["a/"]
                ):
                    first_foot = frame
                for key in raw:
                    raw[key].append(result[key])
                target_rows.append(np.stack([proposals[prefix] for prefix in prefixes]))
                if w.failed:
                    break
            arrays = {key: np.concatenate(value) for key, value in raw.items()}
            arrays.update(
                joint_targets=np.stack(target_rows),
                velocity_observations=np.stack(obs_rows),
                velocity_proposed_actions=np.stack(action_rows),
                velocity_inferred_mask=np.array(inferred_masks, bool),
            )
            artifact = a.output / f"{name}-{index}.npz"
            np.savez_compressed(artifact, **arrays)
            baseline = reference["records"][index]
            # Homogeneous single-player arms preserve exactly the source body,
            # policy and physics up to the fixed control handoff.
            prefix_exact = None
            if name != "heterogeneous-shadow":
                with np.load(a.source / baseline["artifact"]["path"]) as old:
                    prefix_exact = {
                        key: bool(np.array_equal(arrays[key][:2850], old[key][:2850]))
                        for key in ("qpos", "qvel", "torque")
                    }
            row = dict(
                arm=name,
                index=index,
                controls=w.control_steps,
                native_steps=w.native_steps,
                actual_foot_contact_control=first_foot,
                handoff_executed=handoff is not None,
                body_failed=result["body_failed"],
                healthy_full_horizon=w.control_steps == 652 and not w.failed,
                prehandoff_prefix_exact=prefix_exact,
                actual_velocity_inferences=stream.neural_calls,
                other_actual_velocity_inferences=other.neural_calls if other is not None else 0,
                actual_kick_proposals=kick.proposals,
                baseline_controls=baseline["controls"],
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            results.append(row)
            write_once(a.output / f"record-{name}-{index}.json", row)
            print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="ROBONALDO_NEURAL_VELOCITY_HANDOFF_DIAGNOSED_NOT_PROMOTED",
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
