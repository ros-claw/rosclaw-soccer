"""Rebuild shared RoboNaldo trials with fresh NN and a separate manual clock.

No SharedRoboNaldoPlayer, SharedNativeMotorWorld, saved-action playback, model
update or hardware integration. The MuJoCo engine and policy asset are shared;
this verifies independent control plumbing, not an independent simulator.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--source-hash", required=True)
    p.add_argument("--producer-head", required=True)
    p.add_argument("--expected-head", required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--velocity-recovery", action="store_true")
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    path = a.source / "complete.json"
    if a.output.exists() or digest(path) != a.source_hash:
        raise ValueError("unused output and exact closed source required")
    source = json.loads(path.read_text())
    declaration = json.loads((a.source / "declaration.json").read_text())
    if (
        source["source_head"] != a.producer_head
        or source["status"]
        != (
            "ROBONALDO_NEURAL_VELOCITY_HANDOFF_DIAGNOSED_NOT_PROMOTED"
            if a.velocity_recovery
            else "ROBONALDO_SHARED_TEAM_SCENE_DIAGNOSED_NOT_PROMOTED"
        )
        or len(source["records"]) != (12 if a.velocity_recovery else 8)
        or source["optimizer_updates"] != 0
        or declaration["protocol"]
        != (
            "robonaldo.velocity_recovery.v1"
            if a.velocity_recovery
            else "robonaldo.actual_team_scene.v1"
        )
    ):
        raise ValueError("entire fixed eight-case actual team-scene source required")
    pins = dict(source["input_file_hashes"])
    pins[str(path)] = a.source_hash
    pins[str(a.source / "declaration.json")] = digest(a.source / "declaration.json")
    for row in source["records"]:
        path = a.source / row["artifact"]["path"]
        if path.parent.resolve() != a.source.resolve() or digest(path) != row["artifact"]["hash"]:
            raise ValueError("confined exact raw source trajectories required")
        pins[str(path)] = row["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    import mujoco
    import numpy as np
    from rosclaw.simforge.backends.unitree_mujoco_backend import _load_robonaldo
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene

    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    policies = [Path(path) for path in pins if path.endswith("/policy/robonaldo/FreeKick.py")]
    # A teacher may pin its prior helper too; all spellings must resolve to the
    # same qualified deployment root, never pick the first of conflicting roots.
    roots = {path.parents[2].resolve() for path in policies}
    if len(scenes) != 1 or len(roots) != 1:
        raise ValueError("one exact team scene and qualified RoboNaldo root required")
    root = roots.pop()
    state_type, output_type, policy_type, mapping = _load_robonaldo(root)
    velocity_paths = [
        Path(key) for key in pins if key.endswith("/velocity/v0/exported/policy.onnx")
    ]
    velocity_configs = [
        Path(key) for key in pins if key.endswith("/velocity/v0/params/deploy.yaml")
    ]
    if a.velocity_recovery and (len(velocity_paths) != 1 or len(velocity_configs) != 1):
        raise ValueError("one qualified exact recovery velocity policy required")

    def velocity():
        return ContinuousVelocityStream(
            velocity_paths[0],
            velocity_configs[0],
            expected_model_hash=pins[str(velocity_paths[0])],
            expected_config_hash=pins[str(velocity_configs[0])],
            horizon_controls=652,
        )

    with np.load(root / "policy/robonaldo/model/freekick_motion.npz") as motion:
        root_pos, root_quat = motion["body_pos_w"][0, 0].copy(), motion["body_quat_w"][0, 0].copy()
        joints = motion["joint_pos"][0][mapping].copy()
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            manual_control_clock=True,
            independent_physics_engine=False,
            saved_action_playback=False,
            worlds=12 if a.velocity_recovery else 8,
            native_step_ceiling=78240 if a.velocity_recovery else 52160,
            velocity_recovery=a.velocity_recovery,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    results, steps = [], 0
    try:
        for row in source["records"]:
            group = row["arm"] if a.velocity_recovery else row["group"]
            prefixes = (
                ("a/", "b/") if group in ("two-separated", "heterogeneous-shadow") else ("a/",)
            )
            m = mujoco.MjModel.from_xml_string(
                shared_native_scene(scenes[0].read_text(), prefixes=prefixes)
            )
            d = mujoco.MjData(m)
            ball = m.body("soccer_ball").id
            bq = int(m.jnt_qposadr[m.body_jntadr[ball]])
            bv = int(m.jnt_dofadr[m.body_jntadr[ball]])
            players = {}
            for number, prefix in enumerate(prefixes):
                pelvis = m.body(prefix + "pelvis").id
                joint_ids = [m.joint(prefix + name).id for name in G1_DDS_JOINT_NAMES]
                qi, vi = m.jnt_qposadr[joint_ids], m.jnt_dofadr[joint_ids]
                ai = [int(np.flatnonzero(m.actuator_trnid[:, 0] == j)[0]) for j in joint_ids]
                rq, rv = (
                    int(m.jnt_qposadr[m.body_jntadr[pelvis]]),
                    int(m.jnt_dofadr[m.body_jntadr[pelvis]]),
                )
                d.qpos[rq : rq + 3] = root_pos + np.array([0.0, number * 8.0, 0.0])
                d.qpos[rq + 3 : rq + 7], d.qpos[qi] = root_quat, joints
                state, output = state_type(29), output_type(29)
                with contextlib.redirect_stdout(io.StringIO()):
                    policy = policy_type(state, output)
                policy.target_pos_w = np.array([5.0, number * 8.0, 0.11], np.float32)
                players[prefix] = dict(
                    policy=policy,
                    state=state,
                    output=output,
                    rq=rq,
                    rv=rv,
                    qi=qi,
                    vi=vi,
                    ai=ai,
                    torso=m.body(prefix + "torso_link").id,
                    pelvis=pelvis,
                    velocity=velocity() if a.velocity_recovery else None,
                )
                if a.velocity_recovery and prefix == "b/":
                    d.qpos[qi] = np.array(players[prefix]["velocity"].params.defaults)
            ball_xy = (
                declaration["starts"][row["index"]]
                if a.velocity_recovery
                else row["ball_initial_xy"]
            )
            d.qpos[bq : bq + 3] = (*ball_xy, 0.11)
            d.qpos[bq + 3 : bq + 7] = (1.0, 0.0, 0.0, 0.0)
            mujoco.mj_forward(m, d)

            def observe(player, d=d, ball=ball, bv=bv):
                s, rq, rv, torso = player["state"], player["rq"], player["rv"], player["torso"]
                s.q, s.dq = d.qpos[player["qi"]].copy(), d.qvel[player["vi"]].copy()
                s.tau_est = d.ctrl[player["ai"]].copy()
                s.root_lin_vel_b, s.root_ang_vel_b = (
                    d.qvel[rv : rv + 3].copy(),
                    d.qvel[rv + 3 : rv + 6].copy(),
                )
                s.torso_pos_w, s.torso_quat_w = d.xpos[torso].copy(), d.xquat[torso].copy()
                s.pelvis_pos_w, s.pelvis_quat_w = (
                    d.qpos[rq : rq + 3].copy(),
                    d.qpos[rq + 3 : rq + 7].copy(),
                )
                s.ball_pos_w, s.ball_vel_w = d.xpos[ball].copy(), d.qvel[bv : bv + 3].copy()
                s.ball_valid = True

            for player in players.values():
                observe(player)
                with contextlib.redirect_stdout(io.StringIO()):
                    player["policy"].enter()
            actual = {key: [] for key in ("qpos", "qvel", "torque", "joint_targets")}
            stop = False
            with np.load(a.source / row["artifact"]["path"]) as ref:
                for frame in range(row["controls"]):
                    targets = []
                    for prefix, player in players.items():
                        stream = player["velocity"]
                        use_velocity = a.velocity_recovery and (prefix == "b/" or frame >= 285)
                        shadow = (
                            a.velocity_recovery
                            and prefix == "a/"
                            and group != "cold"
                            and frame < 285
                        )
                        vt = None
                        if use_velocity or shadow:
                            rq, rv = player["rq"], player["rv"]
                            q = np.concatenate(
                                (d.qpos[rq : rq + 7], d.qpos[player["qi"]], d.qpos[bq : bq + 7])
                            )
                            v = np.concatenate(
                                (d.qvel[rv : rv + 6], d.qvel[player["vi"]], d.qvel[bv : bv + 6])
                            )
                            jac = np.zeros((3, m.nv))
                            mujoco.mj_jacBody(m, d, None, jac, player["pelvis"])
                            angular = d.xmat[player["pelvis"]].reshape(3, 3).T @ (jac @ d.qvel)
                            vt = stream.propose(
                                q,
                                v,
                                angular.copy(),
                                np.zeros(3, np.float32),
                                frame=stream.next_frame,
                            )
                        if use_velocity:
                            target = vt
                        else:
                            observe(player)
                            with contextlib.redirect_stdout(io.StringIO()):
                                player["policy"].run()
                            target = np.array(player["output"].actions, dtype=np.float64)
                        targets.append(target)
                    actual["joint_targets"].append(np.stack(targets))
                    for _ in range(10):
                        torques = np.zeros(m.nu)
                        for player, target in zip(players.values(), targets, strict=True):
                            if a.velocity_recovery and (
                                player is players.get("b/") or frame >= 285
                            ):
                                kp, kd = (
                                    np.array(player["velocity"].params.stiffness),
                                    np.array(player["velocity"].params.damping),
                                )
                            else:
                                kp, kd = (
                                    np.array(player["output"].kps),
                                    np.array(player["output"].kds),
                                )
                            cap = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
                            torques[player["ai"]] = np.clip(
                                kp * (target - d.qpos[player["qi"]]) - kd * d.qvel[player["vi"]],
                                -cap,
                                cap,
                            )
                        d.ctrl[:] = torques
                        mujoco.mj_step(m, d)
                        steps += 1
                        actual["qpos"].append(d.qpos.copy())
                        actual["qvel"].append(d.qvel.copy())
                        actual["torque"].append(torques.copy())
                        stop |= any(
                            d.qpos[x["rq"] + 2] < 0.65
                            or 1 - 2 * (d.qpos[x["rq"] + 4] ** 2 + d.qpos[x["rq"] + 5] ** 2) < 0.9
                            for x in players.values()
                        )
                    if stop and frame != row["controls"] - 1:
                        raise ValueError("own native body failed earlier than source")
                    if a.velocity_recovery and group != "cold" and frame < 285:
                        stream = players["a/"]["velocity"]
                        stream.last_action = (
                            (targets[0] - np.array(stream.params.action_offset))
                            / np.array(stream.params.action_scale)
                        ).astype(np.float32)
                arrays = {key: np.asarray(values) for key, values in actual.items()}
                exact = {
                    key: bool(np.array_equal(values, ref[key])) for key, values in arrays.items()
                }
                error = {
                    key: float(np.max(np.abs(values - ref[key]))) for key, values in arrays.items()
                }
            out = a.output / f"{group}-{row['index']}.npz"
            np.savez_compressed(out, **arrays)
            result = dict(
                group=group,
                index=row["index"],
                all_exact=all(exact.values()),
                exact=exact,
                max_abs_error=error,
                native_steps=row["controls"] * 10,
                own_body_failure=stop,
                source_body_failure=any(row["body_failed"].values()),
                artifact=dict(path=out.name, hash=digest(out)),
            )
            results.append(result)
            write_once(a.output / f"record-{group}-{row['index']}.json", result)
            print(json.dumps(result), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "ROBONALDO_VELOCITY_RECOVERY_MANUAL_CLOCK_REVIEWED_NOT_PROMOTED"
                    if a.velocity_recovery
                    else "SHARED_ROBONALDO_MANUAL_CLOCK_REVIEWED_NOT_PROMOTED"
                ),
                records=results,
                all_exact=all(row["all_exact"] for row in results),
                native_steps=steps,
                input_file_hashes=pins,
                source_head=a.expected_head,
                source_hash=a.source_hash,
                optimizer_updates=0,
                independent_physics_engine=False,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), completed=len(results), known_native_steps=steps),
        )
        raise


if __name__ == "__main__":
    main()
