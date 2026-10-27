"""Independent native/RNN/ONNX replay of all actual-pass receiver role worlds.

No shared-world motor driver, saved-action execution or borrowed memory. All
24 outcomes, including failed bodies and rejected captures, must reproduce.
Actual per-body ball contact witnesses diagnose why a capture was rejected.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("source-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("source-head", "source-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    a = p.parse_args()
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, receipt, source_pins

    clean(a.expected_head)
    rp, dp = a.source_root / "complete.json", a.source_root / "declaration.json"
    if (
        a.output.exists()
        or any(p.is_symlink() for p in (rp, dp))
        or (a.source_root / "failed.json").exists()
        or digest(rp) != a.source_hash
    ):
        raise ValueError("unused audit and exact completed receiver course source required")
    s, declaration = [json.loads(p.read_text()) for p in (rp, dp)]
    bilateral_study = (
        declaration["protocol"] == "dual256.actual_pass.capture_bilateral_sweep.development.v7"
    )
    priority_study = (
        declaration["protocol"] == "dual256.actual_pass.capture_priority_sweep.development.v6"
    )
    sweep_study = (
        declaration["protocol"] == "dual256.actual_pass.capture_geometry_sweep.development.v5"
    )
    development_study = (
        bilateral_study
        or priority_study
        or sweep_study
        or (declaration["protocol"] == "dual256.actual_pass.prealigned_foot_teacher.development.v4")
    )
    prealigned_study = (
        development_study
        or declaration["protocol"] == "dual256.actual_pass.prealigned_foot_teacher.v3"
    )
    capture_study = (
        prealigned_study
        or declaration["protocol"] == "dual256.actual_pass.measured_foot_teacher.v2"
    )
    expected_worlds = 32 if prealigned_study else 24
    producer_status = (
        "CAPTURE_BILATERAL_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        if bilateral_study
        else "CAPTURE_PRIORITY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        if priority_study
        else "CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        if sweep_study
        else "PREALIGNED_FOOT_CAPTURE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
        if development_study
        else "PREALIGNED_FOOT_CAPTURE_TEACHER32_COLLECTED_PENDING_OWN_REPLAY"
        if prealigned_study
        else "ACTUAL_PASS_FOOT_CAPTURE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY"
        if capture_study
        else "ACTUAL_PASS_ROLE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY"
    )
    if (
        s["status"] != producer_status
        or s["source_head"] != a.source_head
        or s["declaration_hash"] != digest(dp)
        or len(s["records"]) != expected_worlds
        or len(declaration["courses"]) != expected_worlds
        or declaration["protocol"]
        not in (
            "dual256.actual_pass.velocity_receiver.teacher_screen.v1",
            "dual256.actual_pass.measured_foot_teacher.v2",
            "dual256.actual_pass.prealigned_foot_teacher.v3",
            "dual256.actual_pass.prealigned_foot_teacher.development.v4",
            "dual256.actual_pass.capture_geometry_sweep.development.v5",
            "dual256.actual_pass.capture_priority_sweep.development.v6",
            "dual256.actual_pass.capture_bilateral_sweep.development.v7",
        )
    ):
        raise ValueError("exact complete preregistered24 paired worlds required")
    context_blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    if development_study:
        block = declaration.get("new_context_block")
        if (
            type(block) is not int
            or block not in (0, 1, 2)
            or declaration.get("context_indices") != list(context_blocks[block])
        ):
            raise ValueError("exact predeclared development context block required")
        indices = context_blocks[block]
    else:
        indices = (1, 4, 20, 21)
    pins = dict(s["input_file_hashes"])
    pins[str(rp)], pins[str(dp)] = a.source_hash, digest(dp)
    for index, row in enumerate(s["records"]):
        path = a.source_root / row["artifact"]["path"]
        if (
            row["index"] != index
            or row["course"] != declaration["courses"][index]
            or path.is_symlink()
            or path.parent != a.source_root
            or digest(path) != row["artifact"]["hash"]
        ):
            raise ValueError("all24 exact confined raw streams required")
        pins[str(path)] = digest(path)
    source_pins(pins)
    check_pins(pins)
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            source_head=a.expected_head,
            producer_hash=a.source_hash,
            input_file_hashes=pins,
            source_actions_executed=False,
            shared_world_or_player_driver_used=False,
            capture_teacher_function_reused_on_own_states=capture_study,
            hardware_authorized=False,
        ),
    )
    counts = dict(native_steps=0, neural_calls=0, teacher_ik_iterations=0)
    records = []
    try:
        from types import SimpleNamespace

        import mujoco
        import numpy as np
        import torch

        from rosclaw_soccer.providers.g1.joint_contract import G1_DDS_JOINT_NAMES
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_foot_capture_teacher import (
            CaptureTeacherConfig,
            MeasuredFootCaptureTeacher,
        )
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
        from scripts.rsi_stroke_search_common import make_episode, open_motor

        base_hash = declaration["baseline_training_hash"]
        audits = [
            json.loads(Path(p).read_text())
            for p in pins
            if p.endswith("/native-audit/complete.json")
        ]
        audits = [r for r in audits if r.get("training_hash") == base_hash]
        if len(audits) != 1:
            raise ValueError("unique consumed baseline context proof required")
        native_path, native = receipt(pins, audits[0]["source_hash"])
        old_dp = native_path.parent / "declaration.json"
        if digest(old_dp) != native["declaration_hash"] or pins[str(old_dp)] != digest(old_dp):
            raise ValueError("exact original consumed TRAIN context data required")
        contexts = json.loads(old_dp.read_text())["contexts"]
        bench, baseline = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
        candidate = declaration["candidate"]
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        xmls = [Path(p) for p in pins if p.endswith("/cached-standard/declared-scene.xml")]
        xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        if (a.source_root / "scene.xml").read_text() != xml:
            raise ValueError("generated robot/ball scene differs from pinned original")
        m = mujoco.MjModel.from_xml_string(xml)
        bm = int(m.body("soccer_ball").id)
        bj = int(m.body_jntadr[bm])
        bq, bv = int(m.jnt_qposadr[bj]), int(m.jnt_dofadr[bj])
        addresses = {}
        for prefix in ("a/", "b/"):
            body = int(m.body(prefix + "pelvis").id)
            rj = int(m.body_jntadr[body])
            ji = [int(m.joint(prefix + n).id) for n in G1_DDS_JOINT_NAMES]
            addresses[prefix] = dict(
                body=body,
                rq=int(m.jnt_qposadr[rj]),
                rv=int(m.jnt_dofadr[rj]),
                qi=m.jnt_qposadr[ji],
                vi=m.jnt_dofadr[ji],
                ai=[int(m.actuator(prefix + n).id) for n in G1_DDS_JOINT_NAMES],
            )
        paths = [Path(p) for p in pins if p.endswith("/velocity/v0/exported/policy.onnx")]
        configs = [Path(p) for p in pins if p.endswith("/velocity/v0/params/deploy.yaml")]
        episodes = {i: make_episode(bench, contexts[i]) for i in indices}
        capture_configs = {
            key: CaptureTeacherConfig(**value)
            for key, value in declaration.get("capture_teacher_configs", {}).items()
        }
        expected_configs = (
            {"bilateral_default", "bilateral_early", "bilateral_early_lower"}
            if bilateral_study
            else {"priority_default", "priority_early", "priority_early_lower"}
            if priority_study
            else {"sweep_early", "sweep_lower", "sweep_early_lower"}
            if sweep_study
            else {"velocity_capture", "velocity_capture_clearance"}
        )
        if capture_study and set(capture_configs) != expected_configs:
            raise ValueError("exact explicit two capture-teacher variants required")
        for claim in s["records"]:
            ci, course = claim["index"], claim["course"]
            ep = episodes[course["context"]]
            with np.load(a.source_root / claim["artifact"]["path"], allow_pickle=False) as z:
                raw = {k: z[k].copy() for k in z.files}
            d = mujoco.MjData(m)
            q0, v0 = ep.layout.canonical(ep.env.data)
            forward = ep.goal[:2] - q0[36:38]
            forward /= np.linalg.norm(forward)
            home = (
                q0[36:38] + 4 * forward + course["lateral_m"] * np.array([-forward[1], forward[0]])
            )
            for prefix, adr in addresses.items():
                rq, rv = adr["rq"], adr["rv"]
                d.qpos[rq : rq + 7], d.qvel[rv : rv + 6] = q0[:7], v0[:6]
                d.qpos[adr["qi"]], d.qvel[adr["vi"]] = q0[7:36], v0[6:35]
                if prefix == "b/":
                    d.qpos[rq : rq + 2] = home
                    w, x, y, z = q0[3:7]
                    d.qpos[rq + 3 : rq + 7] = [-z, -y, x, w]
                    d.qvel[rv : rv + 2] *= -1
            d.qpos[bq : bq + 7], d.qvel[bv : bv + 6] = q0[36:], v0[35:]
            if not np.array_equal(d.qpos, raw["initial_qpos"]) or not np.array_equal(
                d.qvel, raw["initial_qvel"]
            ):
                raise ValueError("source paired initial states differ from own reconstruction")
            mujoco.mj_forward(m, d)
            velocity = (
                None
                if course["arm"] == "forced_kick"
                else G1VelocityPolicy(
                    paths[0],
                    configs[0],
                    expected_model_hash=pins[str(paths[0])],
                    expected_config_hash=pins[str(configs[0])],
                )
            )
            gains = {prefix: (ep.kp, ep.kd) for prefix in addresses}
            if velocity:
                gains["b/"] = (
                    np.array(velocity.params.stiffness),
                    np.array(velocity.params.damping),
                )
            teacher = teacher_world = None
            previous_receiver_forces = np.zeros(3)
            if course["arm"] in capture_configs:

                def canonical(data, *, adr=addresses["b/"]):
                    return (
                        np.r_[
                            data.qpos[adr["rq"] : adr["rq"] + 7],
                            data.qpos[adr["qi"]],
                            data.qpos[bq : bq + 7],
                        ],
                        np.r_[
                            data.qvel[adr["rv"] : adr["rv"] + 6],
                            data.qvel[adr["vi"]],
                            data.qvel[bv : bv + 6],
                        ],
                    )

                br = addresses["b/"]
                teacher_world = SimpleNamespace(
                    model=m,
                    data=d,
                    control_steps=0,
                    failed=False,
                    layouts={
                        "b/": SimpleNamespace(
                            pelvis=br["body"], ball=bm, qi=br["qi"], canonical=canonical
                        )
                    },
                )
                teacher = MeasuredFootCaptureTeacher(
                    teacher_world, "b/", velocity.params, config=capture_configs[course["arm"]]
                )
            states = {prefix: actor.initial_state() for prefix in addresses}
            last = {prefix: np.zeros(29, np.float32) for prefix in addresses}
            refs = {
                prefix: [
                    np.zeros(29, np.float32),
                    np.zeros(29, np.float32),
                    np.zeros(3, np.float32),
                ]
                for prefix in addresses
            }
            goals = {"a/": ep.goal, "b/": np.r_[ep.env.ball_pos[:2] - 3 * forward, 0.11]}
            launch = False
            ready_anchor = (
                (q0[36:38] + 4 * forward + 0.12 * np.array([-forward[1], forward[0]]))
                if (prealigned_study and course["arm"] != "reactive_intercept")
                else None
            )
            events = {}
            peak = np.zeros((2, 3))
            run = longest = 0
            minimum = float("inf")
            failed = np.zeros(2, bool)
            n = len(raw["qpos"])
            for step in range(n):
                mujoco.mj_forward(m, d)
                br = addresses["b/"]
                bquat = d.qpos[br["rq"] + 3 : br["rq"] + 7]
                w, x, y, z = bquat
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                intent = receiving_intent(
                    root_xy=d.qpos[br["rq"] : br["rq"] + 2],
                    yaw=yaw,
                    ball_xy=d.qpos[bq : bq + 2],
                    ball_velocity_xy=d.qvel[bv : bv + 2],
                    home_xy=home,
                    pass_direction=forward,
                    launch_seen=launch,
                    ready_anchor_xy=ready_anchor,
                )
                launch = intent.launch_seen
                if intent.role_state != raw["receiver_intent_state"][step]:
                    raise ValueError("measured receiver intent state differs")
                targets = {}
                for prefix, adr in addresses.items():
                    label = "A_" if prefix == "a/" else "B_"
                    if prefix == "b/" and velocity:
                        q = np.r_[
                            d.qpos[adr["rq"] : adr["rq"] + 7],
                            d.qpos[adr["qi"]],
                            d.qpos[bq : bq + 7],
                        ]
                        v = np.r_[
                            d.qvel[adr["rv"] : adr["rv"] + 6],
                            d.qvel[adr["vi"]],
                            d.qvel[bv : bv + 6],
                        ]
                        jac = np.zeros((3, m.nv))
                        mujoco.mj_jacBody(m, d, None, jac, adr["body"])
                        angular = d.xmat[adr["body"]].reshape(3, 3).T @ (jac @ d.qvel)
                        command = np.array(
                            intent.command_body
                            if course["arm"] != "velocity_wait"
                            else (0.0, 0.0, 0.0),
                            np.float32,
                        )
                        targets[prefix] = velocity.propose(q, v, angular, command, frame=step)
                        evidence = dict(
                            V_obs=velocity.last_observation,
                            V_action=velocity.last_action,
                            V_command=command,
                            V_phase=np.array(velocity.phase),
                        )
                        if teacher is not None:
                            targets[prefix], capture_plan = teacher.propose(
                                velocity.last_action.copy(),
                                previous_forces=previous_receiver_forces,
                            )
                            evidence.update(
                                {"CAP_" + key: value for key, value in capture_plan.items()}
                            )
                            counts["teacher_ik_iterations"] += capture_plan["teacher_ik_iterations"]
                    else:
                        if step == ep.clip.length:
                            refs[prefix][1:] = [np.zeros(29, np.float32), np.zeros(3, np.float32)]
                        view = SimpleNamespace(
                            joint_pos=d.qpos[adr["qi"]].copy(),
                            joint_vel=d.qvel[adr["vi"]].copy(),
                            pelvis_pos=d.xpos[adr["body"]].copy(),
                            pelvis_quat=d.xquat[adr["body"]].copy(),
                            base_ang_vel=d.qvel[adr["rv"] + 3 : adr["rv"] + 6].copy(),
                        )
                        obs = ep.build_observation(
                            view,
                            None,
                            0,
                            last[prefix],
                            *refs[prefix],
                            d.xpos[bm].copy(),
                            goals[prefix],
                            ep.offset,
                            obs_dim=None,
                        )
                        with torch.no_grad():
                            mean, after = actor(
                                torch.from_numpy(obs).reshape(1, 1, 160), states[prefix]
                            )
                        action = mean[0, 0].numpy().copy()
                        evidence = dict(
                            NN_obs=obs,
                            NN_action=action,
                            NN_h_before=states[prefix][0].numpy(),
                            NN_c_before=states[prefix][1].numpy(),
                            NN_h_after=after[0].numpy(),
                            NN_c_after=after[1].numpy(),
                        )
                        states[prefix], last[prefix] = after, action
                        targets[prefix] = ep.offset + action[ep.order] * ep.scale
                        if step < ep.clip.length:
                            ang = ep.clip.body_ang_vel_w[step, ep.torso_index].copy()
                            if prefix == "b/":
                                ang[:2] *= -1
                            refs[prefix] = [
                                ep.clip.joint_pos[step].copy(),
                                ep.clip.joint_vel[step].copy(),
                                ang,
                            ]
                    counts["neural_calls"] += 1
                    for key, value in evidence.items():
                        if not np.array_equal(value, raw[label + key][step]):
                            raise ValueError(
                                f"own neural stream differs {ci}:{prefix}:{step}:{key}"
                            )
                control_peak = np.zeros((2, 3))
                for sub in range(10):
                    torque = np.zeros(m.nu)
                    for prefix, adr in addresses.items():
                        kp, kd = gains[prefix]
                        torque[adr["ai"]] = np.clip(
                            kp * (targets[prefix] - d.qpos[adr["qi"]]) - kd * d.qvel[adr["vi"]],
                            -ep.effort,
                            ep.effort,
                        )
                    d.ctrl[:] = torque
                    mujoco.mj_step(m, d)
                    counts["native_steps"] += 1
                    forces = np.zeros((2, 3))
                    for contact_id in range(d.ncon):
                        contact = d.contact[contact_id]
                        b1, b2 = (int(m.geom_bodyid[g]) for g in contact.geom)
                        other = b2 if b1 == bm else b1 if b2 == bm else -1
                        name = m.body(other).name if other > 0 else ""
                        for pi, prefix in enumerate(addresses):
                            if name.startswith(prefix):
                                wrench = np.zeros(6)
                                mujoco.mj_contactForce(m, d, contact_id, wrench)
                                force = float(np.linalg.norm(wrench[:3]))
                                part = (
                                    0
                                    if name == prefix + "left_ankle_roll_link"
                                    else 1
                                    if name == prefix + "right_ankle_roll_link"
                                    else 2
                                )
                                forces[pi, part] = max(forces[pi, part], force)
                                if force > 1:
                                    e = events.setdefault(
                                        name, dict(first_substep=step * 10 + sub, peak_force_n=0.0)
                                    )
                                    e["peak_force_n"] = max(e["peak_force_n"], force)
                    peak = np.maximum(peak, forces)
                    control_peak = np.maximum(control_peak, forces)
                    for key, value in dict(
                        qpos=d.qpos, qvel=d.qvel, torque=torque, ball_forces=forces
                    ).items():
                        if not np.array_equal(value, raw[key][step, sub]):
                            raise ValueError(
                                f"own native trajectory differs {ci}:{step}:{sub}:{key}"
                            )
                    if any(int(w.number) for w in d.warning) or not np.array_equal(
                        d.actuator_force, torque
                    ):
                        raise ValueError("native warning or changed torque transmission")
                    for pi, adr in enumerate(addresses.values()):
                        root = d.qpos[adr["rq"] : adr["rq"] + 7]
                        failed[pi] |= root[2] < 0.65 or 1 - 2 * (root[4] ** 2 + root[5] ** 2) < 0.9
                if not np.array_equal(failed, raw["body_failed"][step]):
                    raise ValueError("source body failure trajectory differs")
                previous_receiver_forces = control_peak[1].copy()
                if teacher is not None:
                    velocity.last_action = capture_plan["teacher_applied_action"].copy()
                    teacher_world.control_steps = step + 1
                    teacher_world.failed = bool(failed.any())
                distance = float(
                    np.linalg.norm(d.qpos[br["rq"] : br["rq"] + 2] - d.qpos[bq : bq + 2])
                )
                minimum = min(minimum, distance)
                close = (
                    launch
                    and peak[1, :2].max() > 10
                    and distance <= 0.65
                    and np.linalg.norm(d.qvel[br["rv"] : br["rv"] + 2] - d.qvel[bv : bv + 2]) <= 0.6
                    and d.qpos[bq + 2] <= 0.3
                )
                run = run + 1 if close else 0
                longest = max(longest, run)
                if failed.any() and step != n - 1:
                    raise ValueError("source continued after failure")
            healthy = bool(not failed.any() and n == 300)
            capture = bool(healthy and longest >= 15 and peak[1, 2] <= 1)
            if (
                claim["healthy"] != healthy
                or claim["qualified_capture"] != capture
                or claim["consecutive_capture_controls"] != longest
                or claim["minimum_receiver_ball_distance_m"] != minimum
                or not np.array_equal(peak, np.array(claim["maximum_native_contact_force_n"]))
                or (not failed.any() and n != 300)
            ):
                raise ValueError("source outcome/capture grade differs from actual own trajectory")
            record = dict(
                index=ci,
                course=course,
                controls=n,
                healthy=healthy,
                qualified_capture=capture,
                contact_body_witnesses=events,
                consecutive_capture_controls=longest,
            )
            records.append(record)
            write_once(a.output / f"course{ci}.json", record)
            print(
                json.dumps(dict(index=ci, healthy=healthy, capture=capture, counts=counts)),
                flush=True,
            )
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "CAPTURE_BILATERAL_SWEEP_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                    if bilateral_study
                    else "CAPTURE_PRIORITY_SWEEP_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                    if priority_study
                    else "CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                    if sweep_study
                    else "PREALIGNED_FOOT_CAPTURE_DEVELOPMENT32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                    if development_study
                    else "PREALIGNED_FOOT_CAPTURE32_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                    if prealigned_study
                    else "ACTUAL_PASS_FOOT_CAPTURE24_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                    if capture_study
                    else "ACTUAL_PASS_ROLE24_ALL_OWN_NATIVE_NEURAL_AUDIT_PASS"
                ),
                source_head=a.expected_head,
                producer_hash=a.source_hash,
                records=records,
                counts=counts,
                input_file_hashes=pins,
                goal_complete=False,
                team_game_verified=False,
                hardware_authorized=False,
                promotion_authorized=False,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="ROLE_OWN_AUDIT_FAILED_PRESERVE_NO_RETRY",
                source_head=a.expected_head,
                error_type=type(error).__name__,
                error=str(error),
                counts=counts,
                completed_records=records,
                hardware_authorized=False,
            ),
        )
        raise


if __name__ == "__main__":
    main()
