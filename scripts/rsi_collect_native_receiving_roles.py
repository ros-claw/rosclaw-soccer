"""Paired SIM role study: actual neural passer and physically coupled receiver.

Twenty-four or32 TRAIN worlds: four consumed entries, two receiver offsets,
three or four declared motor/intent arms. One standard ball is kicked by
the real passer. No scripted ball flight, hidden resets or saved actions.
This qualifies a teacher/backend combination, NOT new RL or a team policy.
"""

import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for key in ("frontier-root", "output"):
        p.add_argument("--" + key, type=Path, required=True)
    for key in ("frontier-head", "frontier-hash", "expected-head"):
        p.add_argument("--" + key, required=True)
    p.add_argument("--teacher-screen", action="store_true")
    p.add_argument("--prealign", action="store_true")
    p.add_argument("--new-context-block", type=int, choices=(0, 1, 2))
    p.add_argument("--capture-sweep", action="store_true")
    p.add_argument("--priority-sweep", action="store_true")
    p.add_argument("--bilateral-sweep", action="store_true")
    a = p.parse_args()
    if a.prealign and not a.teacher_screen:
        p.error("prealignment is a separately declared foot-teacher protocol")
    if a.new_context_block is not None and not (a.prealign and a.teacher_screen):
        p.error("new-context development requires the explicit prealigned teacher protocol")
    if (a.capture_sweep or a.priority_sweep or a.bilateral_sweep) and a.new_context_block is None:
        p.error("capture sweep requires a declared new-context block")
    if sum((a.capture_sweep, a.priority_sweep, a.bilateral_sweep)) > 1:
        p.error("one capture sweep protocol at a time")
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import admit_frontier, clean, receipt, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused physically coupled receiving study output required")
    pins, base_hash, candidates, contexts = admit_frontier(
        a.frontier_root, a.frontier_head, a.frontier_hash
    )
    if [c["seed"] for c in candidates] != [65023]:
        raise ValueError("fixed loss-free neural passer identity required")
    paths = [Path(path) for path in pins if path.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(path) for path in pins if path.endswith("/velocity/v0/params/deploy.yaml")]
    if len(paths) != 1 or len(configs) != 1:
        raise ValueError("exact pinned existing learned velocity backend required")
    source_pins(pins)
    check_pins(pins)
    arms = (
        ("velocity_intercept", "velocity_capture", "velocity_capture_clearance")
        if a.teacher_screen
        else ("forced_kick", "velocity_wait", "velocity_intercept")
    )
    if a.prealign:
        arms = ("reactive_intercept",) + arms
    if a.capture_sweep:
        arms = ("velocity_intercept", "sweep_early", "sweep_lower", "sweep_early_lower")
    if a.priority_sweep:
        arms = ("velocity_intercept", "priority_default", "priority_early", "priority_early_lower")
    if a.bilateral_sweep:
        arms = (
            "velocity_intercept",
            "bilateral_default",
            "bilateral_early",
            "bilateral_early_lower",
        )
    context_blocks = ((2, 3, 5, 7), (8, 9, 11, 13), (15, 16, 17, 18))
    indices = (
        list(context_blocks[a.new_context_block])
        if a.new_context_block is not None
        else [1, 4, 20, 21]
    )
    if a.new_context_block is not None:
        frontier = json.loads((a.frontier_root / "complete.json").read_text())
        report = next(r for r in frontier["reports"] if r["seed"] == 65023)
        if not set(indices).issubset(report["achieved_contexts"]):
            raise ValueError("new receiver contexts must be in the acquired passer TRAIN frontier")
    courses = [
        dict(context=i, lateral_m=side, arm=arm)
        for i in indices
        for side in (-0.55, 0.55)
        for arm in arms
    ]
    a.output.mkdir(parents=True)
    from dataclasses import asdict

    from rosclaw_soccer.rsi.measured_foot_capture_teacher import CaptureTeacherConfig

    teacher_configs = (
        {
            "velocity_capture": CaptureTeacherConfig(shin_clearance=False),
            "velocity_capture_clearance": CaptureTeacherConfig(shin_clearance=True),
        }
        if a.teacher_screen
        else {}
    )
    if a.capture_sweep:
        teacher_configs = {
            "sweep_early": CaptureTeacherConfig(
                anticipation_s=0.40, contact_front_m=0.35, shin_clearance=True
            ),
            "sweep_lower": CaptureTeacherConfig(
                foot_height_below_ball_m=0.06, toe_up_rad=0.20, shin_clearance=True
            ),
            "sweep_early_lower": CaptureTeacherConfig(
                anticipation_s=0.40,
                contact_front_m=0.35,
                foot_height_below_ball_m=0.06,
                toe_up_rad=0.20,
                shin_clearance=True,
            ),
        }
    if a.priority_sweep:
        teacher_configs = {
            "priority_default": CaptureTeacherConfig(clearance_profile="anticipatory-priority-v1"),
            "priority_early": CaptureTeacherConfig(
                anticipation_s=0.40,
                contact_front_m=0.35,
                clearance_profile="anticipatory-priority-v1",
            ),
            "priority_early_lower": CaptureTeacherConfig(
                anticipation_s=0.40,
                contact_front_m=0.35,
                foot_height_below_ball_m=0.06,
                toe_up_rad=0.20,
                clearance_profile="anticipatory-priority-v1",
            ),
        }
    if a.bilateral_sweep:
        teacher_configs = {
            "bilateral_default": CaptureTeacherConfig(
                clearance_profile="anticipatory-priority-v1", bilateral_clearance=True
            ),
            "bilateral_early": CaptureTeacherConfig(
                anticipation_s=0.40,
                contact_front_m=0.35,
                clearance_profile="anticipatory-priority-v1",
                bilateral_clearance=True,
            ),
            "bilateral_early_lower": CaptureTeacherConfig(
                anticipation_s=0.40,
                contact_front_m=0.35,
                foot_height_below_ball_m=0.06,
                toe_up_rad=0.20,
                clearance_profile="anticipatory-priority-v1",
                bilateral_clearance=True,
            ),
        }
    declaration = dict(
        protocol=(
            "dual256.actual_pass.capture_bilateral_sweep.development.v7"
            if a.bilateral_sweep
            else "dual256.actual_pass.capture_priority_sweep.development.v6"
            if a.priority_sweep
            else "dual256.actual_pass.capture_geometry_sweep.development.v5"
            if a.capture_sweep
            else "dual256.actual_pass.prealigned_foot_teacher.development.v4"
            if a.new_context_block is not None
            else "dual256.actual_pass.prealigned_foot_teacher.v3"
            if a.prealign
            else "dual256.actual_pass.measured_foot_teacher.v2"
            if a.teacher_screen
            else "dual256.actual_pass.velocity_receiver.teacher_screen.v1"
        ),
        source_head=a.expected_head,
        input_file_hashes=pins,
        candidate=candidates[0],
        baseline_training_hash=base_hash,
        courses=courses,
        context_indices=indices,
        new_context_block=a.new_context_block,
        capture_sweep=a.capture_sweep,
        priority_sweep=a.priority_sweep,
        bilateral_sweep=a.bilateral_sweep,
        horizon_controls=300,
        receiver_forward_m=4.0,
        source_motion_world_rotation_rad=3.141592653589793,
        same_initial_qpos_qvel_for_each_paired_arm=True,
        shared_ball_count=1,
        maximum_native_steps=len(courses) * 3000,
        maximum_actor_calls=len(courses) * 600,
        capture_rule=dict(
            max_horizontal_distance_m=0.65,
            max_relative_speed_mps=0.6,
            max_ball_height_m=0.3,
            minimum_consecutive_controls=15,
            required_native_foot_force_n=10.0,
            maximum_nonfoot_force_n=1.0,
        ),
        goal_complete=False,
        promotion_authorized=False,
        hardware_authorized=False,
        optimizer_updates=0,
        geometry_intent_is_teacher_not_learned_policy=True,
        old100_standing_and_legacy52_not_claimed=True,
        capture_teacher_configs={key: asdict(value) for key, value in teacher_configs.items()},
        communicated_ready_anchor_before_launch=a.prealign,
        ready_anchor_lateral_m=0.12 if a.prealign else None,
        history_semantics=(
            "previous velocity action becomes ACTUAL executed target in original float32 units"
        ),
    )
    write_once(a.output / "declaration.json", declaration)
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        from types import SimpleNamespace

        import mujoco
        import numpy as np

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_foot_capture_teacher import MeasuredFootCaptureTeacher
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.shared_native_motor_world import (
            NativePDParameters,
            SharedNativeMotorWorld,
        )
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
        from rosclaw_soccer.sim.contracts import hash_json
        from rosclaw_soccer.training.native_pass_courses import NativePassCourse
        from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
        from scripts.rsi_stroke_search_common import make_episode, open_motor

        bench, baseline = open_motor(pins, base_hash, {i: contexts[i] for i in indices})
        candidate = candidates[0]
        _, training = receipt(pins, candidate["training_hash"])
        actor, _ = load_dual_candidate_for_sim_audit(
            baseline.parent,
            parent_lineage_hash=training["parent_lineage_hash"],
            root=Path(candidate["training_root"]),
            expected_receipt_hash=candidate["training_hash"],
            checkpoint_kind=candidate["kind"],
        )
        xmls = [Path(path) for path in pins if path.endswith("/cached-standard/declared-scene.xml")]
        xml = shared_native_scene(xmls[0].read_text(), prefixes=("a/", "b/"))
        m = mujoco.MjModel.from_xml_string(xml)
        ball_dimensions = inspect_native_ball_dimensions(
            m, geom_id=int(m.geom("soccer_ball_geom").id)
        )
        if not ball_dimensions.size_and_mass_in_ifab_range:
            raise ValueError("declared standard ball size and mass required")
        (a.output / "scene.xml").write_text(xml)
        layouts = {
            prefix: NativeVelocityLayout(
                m, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=2
            )
            for prefix in ("a/", "b/")
        }
        episodes = {i: make_episode(bench, contexts[i]) for i in indices}
        velocity_example = G1VelocityPolicy(
            paths[0],
            configs[0],
            expected_model_hash=pins[str(paths[0])],
            expected_config_hash=pins[str(configs[0])],
        )
        vp = velocity_example.params
        parameter_sets = {}
        for i, ep in episodes.items():
            parameter_sets[i] = {
                "a/": NativePDParameters(ep.kp, ep.kd, ep.effort),
                "b_kick": NativePDParameters(ep.kp, ep.kd, ep.effort),
                "b_velocity": NativePDParameters(
                    np.array(vp.stiffness), np.array(vp.damping), ep.effort
                ),
            }

        def prepare(course):
            ep = episodes[course["context"]]
            q, v = ep.layout.canonical(ep.env.data)
            forward = ep.goal[:2] - q[36:38]
            forward /= np.linalg.norm(forward)
            lateral = np.array([-forward[1], forward[0]])
            home = q[36:38] + 4 * forward + course["lateral_m"] * lateral
            d = mujoco.MjData(m)
            for prefix, layout in layouts.items():
                d.qpos[layout.rq : layout.rq + 7] = q[:7]
                d.qpos[layout.qi] = q[7:36]
                d.qvel[layout.rv : layout.rv + 6] = v[:6]
                d.qvel[layout.vi] = v[6:35]
                if prefix == "b/":
                    d.qpos[layout.rq : layout.rq + 2] = home
                    w, x, y, z = q[3:7]
                    d.qpos[layout.rq + 3 : layout.rq + 7] = [-z, -y, x, w]  # worldZ(pi)*q
                    d.qvel[layout.rv : layout.rv + 2] *= -1
            al = layouts["a/"]
            d.qpos[al.bq : al.bq + 7], d.qvel[al.bv : al.bv + 6] = q[36:], v[35:]
            mujoco.mj_forward(m, d)
            for c in d.contact:
                b1, b2 = (int(m.geom_bodyid[g]) for g in c.geom)
                if al.ball in (b1, b2) and (b2 if b1 == al.ball else b1) != 0:
                    raise ValueError("declared initial ball/body overlap; reject ALL courses")
            ps = parameter_sets[course["context"]]
            world = SharedNativeMotorWorld(
                m,
                d,
                players={
                    "a/": ps["a/"],
                    "b/": ps["b_kick" if course["arm"] == "forced_kick" else "b_velocity"],
                },
                ball_body_name="soccer_ball",
            )
            return ep, world, home, forward

        # Full paired-course declaration preflight, zero steps before first case.
        typed_courses = []
        identities = set()
        for course in courses:
            _, world, _, _ = prepare(course)
            if world.native_steps:
                raise ValueError("no partial course-bank physics before preflight")
            key = (course["context"], course["lateral_m"])
            if key not in identities:
                identities.add(key)
                typed_courses.append(
                    NativePassCourse(
                        course["context"],
                        4.0,
                        course["lateral_m"],
                        hash_json(
                            dict(qpos=world.data.qpos.tolist(), qvel=world.data.qvel.tolist())
                        ),
                    )
                )
        course_binding = preflight_receiving_courses(tuple(typed_courses))
        write_once(
            a.output / "preflight.json",
            dict(
                courses=len(courses),
                native_steps=0,
                all_declared_dimensions_and_initial_contacts_checked=True,
                compiled_ball_dimensions=asdict(ball_dimensions),
                typed_course_binding=course_binding,
                distinct_initial_courses=len(typed_courses),
            ),
        )

        for ci, course in enumerate(courses):
            ep, world, home, forward = prepare(course)
            d = world.data
            rotated_ang = ep.clip.body_ang_vel_w.copy()
            rotated_ang[..., :2] *= -1
            rotated_clip = SimpleNamespace(
                length=ep.clip.length,
                joint_pos=ep.clip.joint_pos,
                joint_vel=ep.clip.joint_vel,
                body_ang_vel_w=rotated_ang,
            )

            def neural(prefix, clip, goal, *, world=world, ep=ep):
                return SharedNativeNeuralPlayer(
                    world,
                    prefix,
                    actor,
                    clip=clip,
                    torso_index=ep.torso_index,
                    build_observation=ep.build_observation,
                    goal=goal,
                    offset=ep.offset,
                    scale=ep.scale,
                    order=ep.order,
                )

            passer = neural("a/", ep.clip, ep.goal)
            receiver = (
                neural("b/", rotated_clip, np.r_[ep.env.ball_pos[:2] - 3 * forward, 0.11])
                if course["arm"] == "forced_kick"
                else None
            )
            velocity = (
                None
                if receiver
                else G1VelocityPolicy(
                    paths[0],
                    configs[0],
                    expected_model_hash=pins[str(paths[0])],
                    expected_config_hash=pins[str(configs[0])],
                )
            )
            capture_teacher = (
                MeasuredFootCaptureTeacher(
                    world, "b/", velocity.params, config=teacher_configs[course["arm"]]
                )
                if course["arm"] in teacher_configs
                else None
            )
            previous_receiver_forces = np.zeros(3)
            ready_anchor = (
                (ep.env.ball_pos[:2] + 4 * forward + 0.12 * np.array([-forward[1], forward[0]]))
                if (a.prealign and course["arm"] != "reactive_intercept")
                else None
            )
            raw, states = [], []
            launch_seen, capture_run, longest_capture = False, 0, 0
            max_forces = np.zeros((2, 3))
            minimum_distance = float("inf")
            initial_q, initial_v = d.qpos.copy(), d.qvel.copy()
            for step in range(300):
                mujoco.mj_forward(m, d)
                for pl in (passer, receiver):
                    if pl is not None and pl.phase == "MOTION_COMPLETE":
                        pl.begin_standing()
                target_a, trace_a = passer.propose()
                counts["actor_calls"] += 1
                q, v = layouts["b/"].canonical(d)
                w, x, y, z = q[3:7]
                yaw = float(np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))
                intent = receiving_intent(
                    root_xy=q[:2],
                    yaw=yaw,
                    ball_xy=q[36:38],
                    ball_velocity_xy=v[35:37],
                    home_xy=home,
                    pass_direction=forward,
                    launch_seen=launch_seen,
                    ready_anchor_xy=ready_anchor,
                )
                launch_seen = intent.launch_seen
                if receiver:
                    target_b, trace_b = receiver.propose()
                else:
                    command = np.array(
                        intent.command_body
                        if course["arm"] != "velocity_wait"
                        else (0.0, 0.0, 0.0),
                        np.float32,
                    )
                    target_b = velocity.propose(
                        q, v, layouts["b/"].angular_velocity_body(d), command, frame=step
                    )
                    trace_b = dict(
                        V_obs=velocity.last_observation.copy(),
                        V_action=velocity.last_action.copy(),
                        V_command=command,
                        V_phase=np.array(velocity.phase),
                    )
                    if capture_teacher is not None:
                        target_b, capture_plan = capture_teacher.propose(
                            velocity.last_action.copy(), previous_forces=previous_receiver_forces
                        )
                        trace_b.update({"CAP_" + key: value for key, value in capture_plan.items()})
                counts["actor_calls"] += 1
                row = world.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                passer.commit()
                if receiver:
                    receiver.commit()
                if capture_teacher is not None:
                    velocity.last_action = capture_plan["teacher_applied_action"].copy()
                previous_receiver_forces = row["ball_forces"].max(axis=0)[1].copy()
                max_forces = np.maximum(max_forces, row["ball_forces"].max(axis=0))
                q, v = layouts["b/"].canonical(d)
                distance = float(np.linalg.norm(q[:2] - q[36:38]))
                minimum_distance = min(minimum_distance, distance)
                close = (
                    launch_seen
                    and max_forces[1, :2].max() > 10
                    and distance <= 0.65
                    and np.linalg.norm(v[:2] - v[35:37]) <= 0.6
                    and q[38] <= 0.3
                )
                capture_run = capture_run + 1 if close else 0
                longest_capture = max(longest_capture, capture_run)
                row.update({"A_" + k: value for k, value in trace_a.items()})
                row.update({"B_" + k: value for k, value in trace_b.items()})
                raw.append(row)
                states.append(intent.role_state)
                if any(row["body_failed"].values()):
                    break
            path = a.output / f"course{ci}.npz"
            np.savez_compressed(
                path,
                initial_qpos=initial_q,
                initial_qvel=initial_v,
                **{k: np.stack([r[k] for r in raw]) for k in raw[0] if k != "body_failed"},
                body_failed=np.array([[r["body_failed"][p] for p in ("a/", "b/")] for r in raw]),
                receiver_intent_state=np.array(states),
            )
            healthy = not world.failed and len(raw) == 300
            row = dict(
                course=course,
                index=ci,
                controls=len(raw),
                healthy=healthy,
                body_failed=raw[-1]["body_failed"],
                maximum_native_contact_force_n=max_forces.tolist(),
                passer_actual_foot_contact=bool(max_forces[0, :2].max() > 10),
                receiver_actual_foot_contact=bool(max_forces[1, :2].max() > 10),
                receiver_actual_any_contact=bool(max_forces[1].max() > 1),
                minimum_receiver_ball_distance_m=minimum_distance,
                consecutive_capture_controls=longest_capture,
                qualified_capture=bool(healthy and longest_capture >= 15 and max_forces[1, 2] <= 1),
                role_states=sorted(set(states)),
                artifact=dict(path=path.name, hash=digest(path)),
            )
            records.append(row)
            write_once(a.output / f"course{ci}.json", row)
            print(json.dumps(dict(record=row, counts=counts)), flush=True)
        check_pins(pins)
        clean(a.expected_head)
        write_once(
            a.output / "complete.json",
            dict(
                status=(
                    "CAPTURE_BILATERAL_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.bilateral_sweep
                    else "CAPTURE_PRIORITY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.priority_sweep
                    else "CAPTURE_GEOMETRY_SWEEP_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.capture_sweep
                    else "PREALIGNED_FOOT_CAPTURE_DEVELOPMENT32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.new_context_block is not None
                    else "PREALIGNED_FOOT_CAPTURE_TEACHER32_COLLECTED_PENDING_OWN_REPLAY"
                    if a.prealign
                    else "ACTUAL_PASS_FOOT_CAPTURE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY"
                    if a.teacher_screen
                    else "ACTUAL_PASS_ROLE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY"
                ),
                source_head=a.expected_head,
                declaration_hash=digest(a.output / "declaration.json"),
                records=records,
                counts=counts,
                input_file_hashes=pins,
                summary={
                    arm: dict(
                        healthy=sum(r["healthy"] for r in records if r["course"]["arm"] == arm),
                        captures=sum(
                            r["qualified_capture"] for r in records if r["course"]["arm"] == arm
                        ),
                        foot_receptions=sum(
                            r["receiver_actual_foot_contact"]
                            for r in records
                            if r["course"]["arm"] == arm
                        ),
                    )
                    for arm in arms
                },
                team_game_verified=False,
                promotion_authorized=False,
                hardware_authorized=False,
                goal_complete=False,
                optimizer_updates=0,
            ),
        )
    except Exception as error:
        write_once(
            a.output / "failed.json",
            dict(
                status="ROLE_STUDY_FAILED_PRESERVE_NO_RETRY",
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
