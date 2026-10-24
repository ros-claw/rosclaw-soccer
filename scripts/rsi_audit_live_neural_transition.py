"""Fresh neural replay with independent clock, PD blend and contact-release gate.

Does not use SharedNativeMotorWorld, SharedRoboNaldoPlayer, quintic_weight,
blend_live_targets, ContactReleaseGate or saved actions to advance physics.
Same MuJoCo engine and qualified policy assets; no hardware or learning.
"""

import argparse
import contextlib
import io
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("source", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("source-hash", "producer-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists() or digest(a.source / "complete.json") != a.source_hash:
        raise ValueError("unused output and exact source required")
    source = json.loads((a.source / "complete.json").read_text())
    declaration = json.loads((a.source / "declaration.json").read_text())
    release = declaration.get("contact_release", False)
    aim_probe = declaration["protocol"] == "robonaldo.neural_aim_reachability.v1"
    aim_y = declaration.get("neural_aim_input_y_m", 0.0)
    if (
        source["status"] != "LIVE_NEURAL_TRANSITION_DIAGNOSED_NOT_PROMOTED"
        or len(source["records"]) != (4 if aim_probe else 8)
        or source["source_head"] != a.producer_head
        or declaration["duration_controls"] != 20
        or declaration["protocol"]
        != (
            "robonaldo.neural_aim_reachability.v1"
            if aim_probe
            else (
                "robonaldo.contact_release_blend.v1"
                if release
                else "robonaldo.live_neural_blend.v1"
            )
        )
        or (aim_probe and (not release or not -1.2 <= aim_y <= 1.2))
    ):
        raise ValueError("entire exact fixed eight-case live transition protocol required")
    pins = dict(source["input_file_hashes"])
    pins[str(a.source / "complete.json")] = a.source_hash
    pins[str(a.source / "declaration.json")] = digest(a.source / "declaration.json")
    for row in source["records"]:
        path = a.source / row["artifact"]["path"]
        if path.parent.resolve() != a.source.resolve() or digest(path) != row["artifact"]["hash"]:
            raise ValueError("exact confined raw trajectory required")
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

    roots = {
        Path(path).parents[2] for path in pins if path.endswith("/policy/robonaldo/FreeKick.py")
    }
    scenes = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(path) for path in pins if path.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(path) for path in pins if path.endswith("/velocity/v0/params/deploy.yaml")]
    if len(roots) != 1 or any(len(values) != 1 for values in (scenes, models, configs)):
        raise ValueError("unique exact body scene and original neural assets required")
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

    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_hash=a.source_hash,
            source_head=a.expected_head,
            input_file_hashes=pins,
            worlds=4 if aim_probe else 8,
            native_step_ceiling=26080 if aim_probe else 52160,
            independent_clock=True,
            independent_gate_and_blend=True,
            independent_physics_engine=False,
            optimizer_updates=0,
            hardware_authorized=False,
        ),
    )
    records, steps = [], 0
    try:
        for row in source["records"]:
            prefixes = ("a/", "b/") if row["arm"] == "heterogeneous" else ("a/",)
            m = mujoco.MjModel.from_xml_string(
                shared_native_scene(scenes[0].read_text(), prefixes=prefixes)
            )
            d = mujoco.MjData(m)
            ball = m.body("soccer_ball").id
            bq, bv = int(m.jnt_qposadr[m.body_jntadr[ball]]), int(m.jnt_dofadr[m.body_jntadr[ball]])
            streams = {prefix: velocity() for prefix in prefixes}
            layouts = {}
            for number, prefix in enumerate(prefixes):
                pelvis = m.body(prefix + "pelvis").id
                joint_ids = [m.joint(prefix + name).id for name in G1_DDS_JOINT_NAMES]
                qi, vi = m.jnt_qposadr[joint_ids], m.jnt_dofadr[joint_ids]
                ai = [
                    int(np.flatnonzero(m.actuator_trnid[:, 0] == joint)[0]) for joint in joint_ids
                ]
                rq, rv = (
                    int(m.jnt_qposadr[m.body_jntadr[pelvis]]),
                    int(m.jnt_dofadr[m.body_jntadr[pelvis]]),
                )
                layouts[prefix] = dict(pelvis=pelvis, qi=qi, vi=vi, ai=ai, rq=rq, rv=rv)
                d.qpos[rq : rq + 3] = pos + np.array([0.0, number * 8.0, 0.0])
                d.qpos[rq + 3 : rq + 7] = quat
                d.qpos[qi] = joints if prefix == "a/" else np.array(streams[prefix].params.defaults)
            d.qpos[bq : bq + 3], d.qpos[bq + 3 : bq + 7] = (
                (*declaration["starts"][row["index"]], 0.11),
                (1.0, 0.0, 0.0, 0.0),
            )
            mujoco.mj_forward(m, d)
            state, output = state_type(29), output_type(29)
            with contextlib.redirect_stdout(io.StringIO()):
                kick = policy_type(state, output)
            kick.target_pos_w = np.array([5.0, aim_y, 0.11], np.float32)
            cached = {}
            build = kick._build_obs

            def capture(build=build, cached=cached):
                obs = build()
                cached["obs"] = obs.copy()
                return obs

            kick._build_obs = capture
            primary = layouts["a/"]
            torso, foot_body = m.body("a/torso_link").id, m.body("a/right_ankle_roll_link").id

            def observe(d=d, primary=primary, torso=torso, ball=ball, bv=bv, state=state):
                rq, rv = primary["rq"], primary["rv"]
                state.q, state.dq, state.tau_est = (
                    d.qpos[primary["qi"]].copy(),
                    d.qvel[primary["vi"]].copy(),
                    d.ctrl[primary["ai"]].copy(),
                )
                state.root_lin_vel_b, state.root_ang_vel_b = (
                    d.qvel[rv : rv + 3].copy(),
                    d.qvel[rv + 3 : rv + 6].copy(),
                )
                state.pelvis_pos_w, state.pelvis_quat_w = (
                    d.qpos[rq : rq + 3].copy(),
                    d.qpos[rq + 3 : rq + 7].copy(),
                )
                state.torso_pos_w, state.torso_quat_w = d.xpos[torso].copy(), d.xquat[torso].copy()
                state.ball_pos_w, state.ball_vel_w, state.ball_valid = (
                    d.xpos[ball].copy(),
                    d.qvel[bv : bv + 3].copy(),
                    True,
                )

            observe()
            with contextlib.redirect_stdout(io.StringIO()):
                kick.enter()
            oldkp, oldkd = np.array(kick.kps, np.float64), np.array(kick.kds, np.float64)
            newkp, newkd = (
                np.array(streams["a/"].params.stiffness),
                np.array(streams["a/"].params.damping),
            )
            cap = 0.85 * np.array(G1_HARD_TORQUE_LIMITS)
            raw = {
                key: []
                for key in (
                    "qpos",
                    "qvel",
                    "torque",
                    "targets",
                    "blend_weight",
                    "kp",
                    "kd",
                    "kick_observation",
                    "velocity_observation",
                )
            }
            foot_seen, last_contact, clear = False, False, 0
            start = None if release else 285
            stopped = False
            for frame in range(row["controls"]):
                if release:
                    separation = d.xpos[ball] - d.xpos[foot_body]
                    distance = float(np.linalg.norm(separation))
                    jac = np.zeros((3, m.nv))
                    mujoco.mj_jacBody(m, d, jac, None, foot_body)
                    speed = float(
                        np.dot(separation, d.qvel[bv : bv + 3] - jac @ d.qvel)
                        / max(1e-12, distance)
                    )
                    foot_seen |= last_contact
                    clear = 0 if last_contact else clear + 1
                    if (
                        start is None
                        and foot_seen
                        and clear >= 3
                        and distance > 0.35
                        and speed > 0.5
                    ):
                        start = frame
                t = 0.0 if start is None else min(1.0, max(0.0, (frame - start) / 20.0))
                weight = t * t * t * (10.0 + t * (-15.0 + 6.0 * t))
                targets = {}
                for prefix, layout in layouts.items():
                    rq, rv = layout["rq"], layout["rv"]
                    q = np.concatenate(
                        (d.qpos[rq : rq + 7], d.qpos[layout["qi"]], d.qpos[bq : bq + 7])
                    )
                    v = np.concatenate(
                        (d.qvel[rv : rv + 6], d.qvel[layout["vi"]], d.qvel[bv : bv + 6])
                    )
                    jac = np.zeros((3, m.nv))
                    mujoco.mj_jacBody(m, d, None, jac, layout["pelvis"])
                    omega = d.xmat[layout["pelvis"]].reshape(3, 3).T @ (jac @ d.qvel)
                    stream = streams[prefix]
                    targets[prefix] = stream.propose(
                        q, v, omega.copy(), np.zeros(3, np.float32), frame=stream.next_frame
                    )
                incoming = targets["a/"].copy()
                inferred = False
                if weight < 1.0:
                    observe()
                    inferred = kick.time_step >= kick.WARMUP_STEPS
                    with contextlib.redirect_stdout(io.StringIO()):
                        kick.run()
                    outgoing = np.asarray(output.actions, np.float64).copy()
                if weight == 0.0:
                    targets["a/"], kp, kd = outgoing, oldkp, oldkd
                elif weight == 1.0:
                    kp, kd = newkp, newkd
                else:
                    targets["a/"] = (1.0 - weight) * outgoing + weight * incoming
                    kp, kd = (
                        (1.0 - weight) * oldkp + weight * newkp,
                        (1.0 - weight) * oldkd + weight * newkd,
                    )
                raw["targets"].append(np.stack([targets[prefix] for prefix in prefixes]))
                raw["blend_weight"].append(weight)
                raw["kp"].append(kp.copy())
                raw["kd"].append(kd.copy())
                raw["kick_observation"].append(
                    cached["obs"].copy() if inferred else np.zeros(547, np.float32)
                )
                raw["velocity_observation"].append(streams["a/"].last_observation.copy())
                last_contact = False
                for _ in range(10):
                    torque = np.zeros(m.nu)
                    for prefix, layout in layouts.items():
                        pg, dg = (kp, kd) if prefix == "a/" else (newkp, newkd)
                        torque[layout["ai"]] = np.clip(
                            pg * (targets[prefix] - d.qpos[layout["qi"]])
                            - dg * d.qvel[layout["vi"]],
                            -cap,
                            cap,
                        )
                    d.ctrl[:] = torque
                    mujoco.mj_step(m, d)
                    steps += 1
                    for key, value in (("qpos", d.qpos), ("qvel", d.qvel), ("torque", torque)):
                        raw[key].append(value.copy())
                    stopped |= any(
                        d.qpos[layout["rq"] + 2] < 0.65
                        or 1 - 2 * (d.qpos[layout["rq"] + 4] ** 2 + d.qpos[layout["rq"] + 5] ** 2)
                        < 0.9
                        for layout in layouts.values()
                    )
                    if release:
                        for ci in range(d.ncon):
                            bodies = {int(m.geom_bodyid[geom]) for geom in d.contact[ci].geom}
                            if bodies == {ball, foot_body}:
                                force = np.zeros(6)
                                mujoco.mj_contactForce(m, d, ci, force)
                                last_contact |= float(np.linalg.norm(force[:3])) > 1.0
                if weight < 1.0:
                    params = streams["a/"].params
                    streams["a/"].last_action = (
                        (targets["a/"] - np.array(params.action_offset))
                        / np.array(params.action_scale)
                    ).astype(np.float32)
                if 0.0 < weight < 1.0:
                    encoded = np.empty(29, np.float32)
                    encoded[mapping] = (
                        (targets["a/"] - np.array(kick.default_q_mj))
                        / np.array(kick.action_scale_mj)
                    ).astype(np.float32)
                    kick.last_action_il = encoded
                if stopped and frame != row["controls"] - 1:
                    raise ValueError("own physical failure before source endpoint")
            arrays = {key: np.asarray(value) for key, value in raw.items()}
            with np.load(a.source / row["artifact"]["path"]) as ref:
                exact = {
                    key: bool(np.array_equal(value, ref[key])) for key, value in arrays.items()
                }
                errors = {
                    key: float(np.max(np.abs(value - ref[key]))) for key, value in arrays.items()
                }
            if "transition_start" in row and start != row["transition_start"]:
                raise ValueError("independent release event differs from source")
            artifact = a.output / f"{row['arm']}-{row['index']}.npz"
            np.savez_compressed(artifact, **arrays)
            result = dict(
                arm=row["arm"],
                index=row["index"],
                all_exact=all(exact.values()),
                exact=exact,
                max_abs_error=errors,
                native_steps=row["controls"] * 10,
                transition_start=start,
                own_failed=stopped,
                source_failed=any(row["body_failed"].values()),
                artifact=dict(path=artifact.name, hash=digest(artifact)),
            )
            records.append(result)
            write_once(a.output / f"record-{row['arm']}-{row['index']}.json", result)
            print(json.dumps(result), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="LIVE_NEURAL_TRANSITION_MANUAL_CLOCK_REVIEWED_NOT_PROMOTED",
                records=records,
                all_exact=all(row["all_exact"] for row in records),
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
            dict(error=repr(error), completed=len(records), known_native_steps=steps),
        )
        raise


if __name__ == "__main__":
    main()
