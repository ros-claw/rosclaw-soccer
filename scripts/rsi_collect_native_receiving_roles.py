"""Paired SIM role study: actual neural passer and physically coupled receiver.

Twenty-four TRAIN worlds: four consumed source entries, two receiver offsets,
three explicitly declared motor/intent arms. One standard ball is kicked by
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
    a = p.parse_args()
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
    arms = ("forced_kick", "velocity_wait", "velocity_intercept")
    indices = [1, 4, 20, 21]
    courses = [
        dict(context=i, lateral_m=side, arm=arm)
        for i in indices
        for side in (-0.55, 0.55)
        for arm in arms
    ]
    a.output.mkdir(parents=True)
    declaration = dict(
        protocol="dual256.actual_pass.velocity_receiver.teacher_screen.v1",
        source_head=a.expected_head,
        input_file_hashes=pins,
        candidate=candidates[0],
        baseline_training_hash=base_hash,
        courses=courses,
        horizon_controls=300,
        receiver_forward_m=4.0,
        source_motion_world_rotation_rad=3.141592653589793,
        same_initial_qpos_qvel_for_each_paired_arm=True,
        shared_ball_count=1,
        maximum_native_steps=72000,
        maximum_actor_calls=14400,
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
    )
    write_once(a.output / "declaration.json", declaration)
    counts = dict(native_steps=0, actor_calls=0)
    records = []
    try:
        from dataclasses import asdict
        from types import SimpleNamespace

        import mujoco
        import numpy as np

        from rosclaw_soccer.physics.native_ball_dimensions import inspect_native_ball_dimensions
        from rosclaw_soccer.providers.g1.velocity_policy import G1VelocityPolicy
        from rosclaw_soccer.rsi.dual_onpolicy_checkpoint import load_dual_candidate_for_sim_audit
        from rosclaw_soccer.rsi.measured_receiving_intent import receiving_intent
        from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
        from rosclaw_soccer.rsi.shared_native_motor_world import (
            NativePDParameters,
            SharedNativeMotorWorld,
        )
        from rosclaw_soccer.rsi.shared_native_neural_player import SharedNativeNeuralPlayer
        from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
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
        for course in courses:
            _, world, _, _ = prepare(course)
            if world.native_steps:
                raise ValueError("no partial course-bank physics before preflight")
        write_once(
            a.output / "preflight.json",
            dict(
                courses=24,
                native_steps=0,
                all_declared_dimensions_and_initial_contacts_checked=True,
                compiled_ball_dimensions=asdict(ball_dimensions),
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
                )
                launch_seen = intent.launch_seen
                if receiver:
                    target_b, trace_b = receiver.propose()
                else:
                    command = np.array(
                        intent.command_body
                        if course["arm"] == "velocity_intercept"
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
                counts["actor_calls"] += 1
                row = world.advance({"a/": target_a, "b/": target_b})
                counts["native_steps"] += 10
                passer.commit()
                if receiver:
                    receiver.commit()
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
                status="ACTUAL_PASS_ROLE_TEACHER24_COLLECTED_PENDING_OWN_REPLAY",
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
