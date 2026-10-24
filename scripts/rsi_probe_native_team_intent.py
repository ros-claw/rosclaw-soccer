"""Native8-body coached role reaction baseline, not learned autonomous4v4 football.

All8 private98->29 streams share one real500Hz MuJoCo clock. No kicks, goals,
root forces, body resets or ball rewrites are provided by this diagnostic.
Fresh NN re-execution is not an independent controller implementation.
"""

import argparse
import json
import math
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("parent", "own", "core-source-root", "output"):
        p.add_argument("--" + name, type=Path, required=True)
    for name in ("parent-hash", "own-hash", "core-expected-head", "expected-head"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--intercept-course", action="store_true")
    p.add_argument("--foot-track-course", action="store_true")
    p.add_argument("--clearance-course", action="store_true")
    p.add_argument("--foot-reference", type=Path)
    p.add_argument("--foot-reference-hash")
    p.add_argument("--previous", type=Path)
    p.add_argument("--previous-hash")
    p.add_argument(
        "--ball-course",
        choices=(
            "outbound-north",
            "incoming-north",
            "incoming-south",
            "incoming-east",
            "incoming-west",
        ),
        default="outbound-north",
    )
    a = p.parse_args()
    if a.foot_track_course and not a.intercept_course:
        p.error("foot tracking is an explicit extension of the interception course")
    if a.clearance_course:
        if not a.foot_track_course or a.foot_reference is None or a.foot_reference_hash is None:
            p.error("clearance contrast requires the entire preceding measured-foot course")
    elif a.foot_reference is not None or a.foot_reference_hash is not None:
        p.error("foot reference is specific to the explicit clearance contrast")
    if a.ball_course != "outbound-north" and not a.intercept_course:
        p.error("new moving-ball geometry requires explicitly versioned interception course")
    if (a.intercept_course and (a.previous is None or a.previous_hash is None)) or (
        not a.intercept_course and (a.previous is not None or a.previous_hash is not None)
    ):
        p.error("interception course requires entire exact preceding8-body baseline")
    from scripts.rsi_active_core_source import pin_active_core
    from scripts.rsi_atomic_artifacts import write_once
    from scripts.rsi_native_context_recovery_common import check_pins, digest
    from scripts.rsi_native_fresh_common import clean, source_pins

    clean(a.expected_head)
    if a.output.exists():
        raise ValueError("unused native team baseline output required")
    reports, pins = {}, {}
    for name in ("parent", "own"):
        path = getattr(a, name) / "complete.json"
        expected = getattr(a, name + "_hash")
        if path.is_symlink() or digest(path) != expected:
            raise ValueError("exact verified native predecessor required")
        report = json.loads(path.read_text())
        for key, value in report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting qualification ancestry")
            pins[key] = value
        pins[str(path)] = expected
        reports[name] = report
    if (
        not reports["own"]["all_exact"]
        or reports["own"]["source_hash"] != a.parent_hash
        or len(reports["parent"]["records"]) != 16
        or not all(x["healthy_full_horizon"] for x in reports["parent"]["records"])
    ):
        raise ValueError(
            "complete healthy independently replayed native asset predecessor required"
        )
    predecessor = None
    if a.intercept_course:
        path = a.previous / "complete.json"
        if path.is_symlink() or digest(path) != a.previous_hash:
            raise ValueError("exact completed baseline required")
        predecessor = json.loads(path.read_text())
        if (
            predecessor["status"] != "NATIVE8_COACHED_ROLE_BASELINE_MEASURED_NOT_A_MATCH"
            or len(predecessor["records"]) != 4
            or not predecessor["fresh_reexecution_exact"]
            or not all(r["healthy_full_horizon"] for r in predecessor["records"])
        ):
            raise ValueError("entire healthy baseline/reexecution required")
        for key, value in predecessor["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting baseline ancestry")
            pins[key] = value
        pins[str(path)] = a.previous_hash
        for row in predecessor["records"]:
            path = a.previous / row["artifact"]["path"]
            if (
                path.is_symlink()
                or path.parent.resolve() != a.previous.resolve()
                or digest(path) != row["artifact"]["hash"]
            ):
                raise ValueError("confined exact preceding baseline arrays required")
            pins[str(path)] = row["artifact"]["hash"]
    pin_active_core(a.core_source_root, a.core_expected_head, pins)
    foot_predecessor = None
    if a.clearance_course:
        path = a.foot_reference / "complete.json"
        if path.is_symlink() or digest(path) != a.foot_reference_hash:
            raise ValueError("exact preceding measured-foot course required")
        foot_predecessor = json.loads(path.read_text())
        declaration = json.loads((a.foot_reference / "declaration.json").read_text())
        if (
            len(foot_predecessor["records"]) != 10
            or not foot_predecessor["fresh_reexecution_exact"]
            or declaration["protocol"] != "native8_measured_foot_tracking.v4"
            or declaration["ball_course"] != a.ball_course
            or (a.foot_reference / "failed.json").exists()
        ):
            raise ValueError("complete same-course physical foot predecessor required")
        for key, value in foot_predecessor["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting foot-course ancestry")
            pins[key] = value
        for file, expected in [(path, a.foot_reference_hash)]:
            pins[str(file)] = expected
        path = a.foot_reference / "declaration.json"
        pins[str(path)] = digest(path)
        for row in foot_predecessor["records"]:
            path = a.foot_reference / row["artifact"]["path"]
            if (
                path.is_symlink()
                or path.parent.resolve() != a.foot_reference.resolve()
                or digest(path) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined predecessor raw arrays required")
            pins[str(path)] = row["artifact"]["hash"]
    source_pins(pins)
    check_pins(pins)
    from rosclaw_soccer.training.native_team_courses import native_team_course_bank
    from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses

    course_bank = native_team_course_bank()
    course_bank_hash = preflight_receiving_courses(course_bank)
    course = next(item for item in course_bank if item.course_id == a.ball_course)
    import mujoco
    import numpy as np
    from rosclaw.simforge.tasks.g1_goalforge.concepts import G1_HARD_TORQUE_LIMITS

    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.rsi.native_team_intent import (
        ROLES,
        clearance_command,
        command_toward,
        finisher_intercept_target,
        foot_tracking_command,
        role_target,
    )
    from rosclaw_soccer.rsi.native_velocity_transfer import NativeVelocityLayout
    from rosclaw_soccer.rsi.shared_native_motor_world import (
        NativePDParameters,
        SharedNativeMotorWorld,
    )
    from rosclaw_soccer.rsi.shared_native_scene import shared_native_scene
    from rosclaw_soccer.sim.contracts import hash_json

    scenes = [Path(k) for k in pins if k.endswith("/cached-standard/declared-scene.xml")]
    models = [Path(k) for k in pins if k.endswith("/velocity/v0/exported/policy.onnx")]
    configs = [Path(k) for k in pins if k.endswith("/velocity/v0/params/deploy.yaml")]
    motions = [Path(k) for k in pins if k.endswith("/policy/robonaldo/model/freekick_motion.npz")]
    if any(len(x) != 1 for x in (scenes, models, configs, motions)):
        raise ValueError("unique pinned qualified body/velocity assets required")
    with np.load(motions[0]) as motion:
        root_z = float(motion["body_pos_w"][0, 0, 2])
    agents = [(team, role, team + "_" + role + "/") for team in ("red", "blue") for role in ROLES]
    prefixes = tuple(x[2] for x in agents)
    xml = shared_native_scene(scenes[0].read_text(), prefixes=prefixes)
    horizon = 600
    starts = ((-8.0, 0.0), (-4.0, -1.5), (-2.0, 2.5), (-1.5, 0.0))
    ball_start, ball_velocity = course.ball_xy, course.ball_velocity_xy
    arms = (
        ("zero", "role-reactive", "role-fast", "role-intercept")
        if a.intercept_course
        else ("zero", "role-reactive")
    )
    if a.foot_track_course:
        arms = (*arms, "role-foot-track")
    if a.clearance_course:
        arms = ("role-foot-track", "role-foot-clearance")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="native8_measured_clearance_contrast.v5"
            if a.clearance_course
            else "native8_measured_foot_tracking.v4"
            if a.foot_track_course
            else "native8_coached_intercept_course.v3"
            if a.intercept_course
            else "native8_coached_ball_intent_baseline.v1",
            source_head=a.expected_head,
            input_file_hashes=pins,
            agents=agents,
            starts_red_m=starts,
            blue_starts_halfturn_of_red=True,
            ball_course=a.ball_course,
            entire_course_bank_preflight_hash=course_bank_hash,
            ball_initial_xy_m=ball_start,
            ball_initial_linear_velocity_mps=[*ball_velocity, 0.0],
            initial_angular_velocity_is_rolling=True,
            control_hz=50,
            native_hz=500,
            horizon_controls=horizon,
            warmup_zero_velocity_controls=50,
            maximum_world_executions=2 * len(arms),
            maximum_native_steps=2 * len(arms) * horizon * 10,
            arms=arms,
            interception_uses_existing_ground_intercept_model=a.intercept_course,
            preselected_slow_ball_root_speed_mps=0.95 if a.intercept_course else None,
            original8body_baseline_hash=a.previous_hash,
            trained_velocity_model_reused=True,
            independent_private_policy_instances=8,
            high_level_role_law_is_coached_not_learned=True,
            optimizer_updates=0,
            no_parent_torque_artifact_body_hash_rebinding=True,
            no_saved_action_replay=True,
            no_state_writes_after_initialization=True,
            no_applied_external_forces=True,
            independent_controller_replay=False,
            not_a_match_or_skill_qualification=True,
            postcontact_metrics_are_not_possession_qualification=True,
            metric_forward_kinematics_uses_separate_snapshot=True,
            foot_tracking_activation_root_gap_m=1.0 if a.foot_track_course else None,
            foot_tracking_gain_per_sec=1.5 if a.foot_track_course else None,
            foot_tracking_world_slew_mps_per_control=0.08 if a.foot_track_course else None,
            nonfoot_contact_counter_is_not_fifa_adjudication=True,
            actual_interplayer_contacts_measured=a.clearance_course,
            preceding_foot_course_hash=a.foot_reference_hash,
            clearance_m=0.6 if a.clearance_course else None,
            clearance_influence_m=1.2 if a.clearance_course else None,
            clearance_approach_gain=0.6 if a.clearance_course else None,
            engine_version=mujoco.__version__,
            engine_is_legacy_research_fixture_not_qualified_harness=True,
            publicity_rights_not_established=True,
            hardware_authorized=False,
            promotion_authorized=False,
        ),
    )
    records, charged = [], 0
    try:
        for arm in arms:
            reference = None
            for repeat in (0, 1):
                m = mujoco.MjModel.from_xml_string(xml)
                d = mujoco.MjData(m)
                streams = {
                    prefix: ContinuousVelocityStream(
                        models[0],
                        configs[0],
                        expected_model_hash=pins[str(models[0])],
                        expected_config_hash=pins[str(configs[0])],
                        horizon_controls=horizon,
                    )
                    for prefix in prefixes
                }
                if len({id(x.last_action) for x in streams.values()}) != 8:
                    raise ValueError("all8 private action memories required")
                layouts, params = {}, {}
                for i, (team, _role, prefix) in enumerate(agents):
                    layout = NativeVelocityLayout(
                        m, robot_prefix=prefix, ball_body_name="soccer_ball", player_count=8
                    )
                    layouts[prefix] = layout
                    sign = 1.0 if team == "red" else -1.0
                    x, y = starts[i % 4]
                    d.qpos[layout.rq : layout.rq + 3] = [sign * x, sign * y, root_z]
                    d.qpos[layout.rq + 3 : layout.rq + 7] = (
                        [1.0, 0.0, 0.0, 0.0] if team == "red" else [0.0, 0.0, 0.0, 1.0]
                    )
                    stream = streams[prefix]
                    d.qpos[layout.qi] = np.asarray(stream.params.defaults)
                    params[prefix] = NativePDParameters(
                        np.asarray(stream.params.stiffness),
                        np.asarray(stream.params.damping),
                        0.85 * np.asarray(G1_HARD_TORQUE_LIMITS),
                    )
                ball = m.body("soccer_ball").id
                gids = np.flatnonzero(
                    (m.geom_bodyid == ball) & ((m.geom_contype != 0) | (m.geom_conaffinity != 0))
                )
                if (
                    len(gids) != 1
                    or m.geom_type[gids[0]] != int(mujoco.mjtGeom.mjGEOM_SPHERE)
                    or abs(float(m.geom_size[gids[0], 0]) - 0.11) > 1e-12
                    or abs(float(m.body_mass[ball]) - 0.43) > 1e-12
                ):
                    raise ValueError("actual standard sphere .11m/.43kg required")
                d.qpos[layout.bq : layout.bq + 7] = [*ball_start, 0.11, 1.0, 0.0, 0.0, 0.0]
                d.qvel[layout.bv : layout.bv + 6] = [
                    *ball_velocity,
                    0.0,
                    -ball_velocity[1] / 0.11,
                    ball_velocity[0] / 0.11,
                    0.0,
                ]
                mujoco.mj_forward(m, d)
                w = SharedNativeMotorWorld(
                    m,
                    d,
                    players=params,
                    ball_body_name="soccer_ball",
                    measure_interplayer_contact=a.clearance_course,
                )
                metric = mujoco.MjData(m)
                feet = {
                    prefix: [
                        m.body(prefix + side + "_ankle_roll_link").id for side in ("left", "right")
                    ]
                    for prefix in prefixes
                }
                touched = np.zeros(8, np.bool_)
                previous_world_commands = np.zeros((8, 2))
                initial = {
                    prefix: d.qpos[lo.rq : lo.rq + 3].copy() for prefix, lo in layouts.items()
                }
                raw = {key: [] for key in ("qpos", "qvel", "torque", "ball_forces")}
                if a.clearance_course:
                    raw["interplayer_contact_forces"] = []
                controls = {
                    key: [] for key in ("commands", "role_targets", "observations98", "targets29")
                }
                if a.intercept_course:
                    controls["intercept_horizon_sec"], controls["kinematically_reachable"] = [], []
                (
                    controls["foot_gap_after_m"],
                    controls["postcontact_near_slow"],
                    controls["ball_speed_after_mps"],
                ) = [], [], []
                if a.foot_track_course:
                    controls["nearest_foot_relative_speed_mps"] = []
                if a.clearance_course:
                    controls["clearance_feasible"], controls["clearance_constrained"] = [], []
                for frame in range(horizon):
                    if a.foot_track_course:
                        metric.qpos[:] = d.qpos
                        mujoco.mj_kinematics(m, metric)
                    ball_xy = d.qpos[layout.bq : layout.bq + 2].copy()
                    commands, targets, goals, observations = [], {}, [], []
                    horizons, reachable_flags = [], []
                    clearance_feasible, clearance_constrained = [], []
                    for agent_index, (team, role, prefix) in enumerate(agents):
                        lo, stream = layouts[prefix], streams[prefix]
                        q, v = lo.canonical(d)
                        quat = q[3:7]
                        yaw = math.atan2(
                            2 * (quat[0] * quat[3] + quat[1] * quat[2]),
                            1 - 2 * (quat[2] ** 2 + quat[3] ** 2),
                        )
                        goal = role_target(team, role, ball_xy)
                        ih, ir = 0.0, False
                        if (
                            arm in ("role-intercept", "role-foot-track", "role-foot-clearance")
                            and role == "finisher"
                        ):
                            goal, ih, ir = finisher_intercept_target(
                                team, q[:2], ball_xy, d.qvel[lo.bv : lo.bv + 2].copy()
                            )
                        command = (
                            np.zeros(3, np.float32)
                            if arm == "zero" or frame < 50
                            else command_toward(
                                q[:2],
                                yaw,
                                goal,
                                ball_xy,
                                maximum_forward_speed=1.0
                                if role == "finisher"
                                and arm
                                in (
                                    "role-fast",
                                    "role-intercept",
                                    "role-foot-track",
                                    "role-foot-clearance",
                                )
                                else 0.8,
                            )
                        )
                        if (
                            arm in ("role-foot-track", "role-foot-clearance")
                            and role == "finisher"
                            and frame >= 50
                            and np.linalg.norm(ball_xy - q[:2]) < 1.0
                        ):
                            command, _ = foot_tracking_command(
                                q[:2],
                                yaw,
                                metric.xpos[feet[prefix], :2].copy(),
                                ball_xy,
                                d.qvel[lo.bv : lo.bv + 2].copy(),
                                previous_world_commands[agent_index],
                            )
                        feasible, constrained = True, False
                        if arm == "role-foot-clearance" and frame >= 50:
                            neighbors = np.array(
                                [
                                    d.qpos[other.rq : other.rq + 2] - q[:2]
                                    for other_prefix, other in layouts.items()
                                    if other_prefix != prefix
                                ]
                            )
                            command, feasible, constrained = clearance_command(
                                command,
                                yaw,
                                neighbors,
                                maximum_speed_mps=1.0 if role == "finisher" else 0.8,
                            )
                        clearance_feasible.append(feasible)
                        clearance_constrained.append(constrained)
                        c, s = math.cos(yaw), math.sin(yaw)
                        previous_world_commands[agent_index] = [
                            c * command[0] - s * command[1],
                            s * command[0] + c * command[1],
                        ]
                        targets[prefix] = stream.propose(
                            q, v, lo.angular_velocity_body(d), command, frame=frame
                        )
                        commands.append(command)
                        goals.append(goal)
                        observations.append(stream.last_observation.copy())
                        horizons.append(ih)
                        reachable_flags.append(ir)
                    for key, value in (
                        ("commands", commands),
                        ("role_targets", goals),
                        ("observations98", observations),
                        ("targets29", [targets[p] for p in prefixes]),
                    ):
                        controls[key].append(np.asarray(value).copy())
                    if a.intercept_course:
                        controls["intercept_horizon_sec"].append(horizons)
                        controls["kinematically_reachable"].append(reachable_flags)
                    if a.clearance_course:
                        controls["clearance_feasible"].append(clearance_feasible)
                        controls["clearance_constrained"].append(clearance_constrained)
                    step = w.advance(targets)
                    charged += 10
                    for key in raw:
                        raw[key].append(step[key])
                    metric.qpos[:] = d.qpos
                    mujoco.mj_kinematics(m, metric)
                    if a.foot_track_course:
                        metric.qvel[:] = d.qvel
                        mujoco.mj_comPos(m, metric)
                        relative_speeds = []
                        for prefix in prefixes:
                            nearest = min(
                                feet[prefix],
                                key=lambda f: np.linalg.norm(
                                    metric.xpos[f] - d.qpos[layout.bq : layout.bq + 3]
                                ),
                            )
                            jac = np.zeros((3, m.nv))
                            mujoco.mj_jacBody(m, metric, jac, None, nearest)
                            relative_speeds.append(
                                float(
                                    np.linalg.norm(
                                        jac @ metric.qvel - d.qvel[layout.bv : layout.bv + 3]
                                    )
                                )
                            )
                        controls["nearest_foot_relative_speed_mps"].append(relative_speeds)
                    foot_gaps = np.array(
                        [
                            min(
                                np.linalg.norm(
                                    metric.xpos[foot] - d.qpos[layout.bq : layout.bq + 3]
                                )
                                for foot in feet[prefix]
                            )
                            for prefix in prefixes
                        ]
                    )
                    touched |= np.max(step["ball_forces"][:, :, :2], axis=(0, 2)) > 1.0
                    ball_speed = float(np.linalg.norm(d.qvel[layout.bv : layout.bv + 3]))
                    controls["foot_gap_after_m"].append(foot_gaps)
                    controls["postcontact_near_slow"].append(
                        touched & (foot_gaps < 0.35) & (ball_speed < 0.5)
                    )
                    controls["ball_speed_after_mps"].append(ball_speed)
                    if w.failed:
                        break
                arrays = {k: np.concatenate(v) for k, v in raw.items()}
                arrays.update({k: np.asarray(v) for k, v in controls.items()})
                own_exact = None
                legacy_identity = None
                foot_identity = None
                if foot_predecessor is not None and arm == "role-foot-track":
                    old = next(
                        r
                        for r in foot_predecessor["records"]
                        if r["arm"] == arm and r["repeat"] == repeat
                    )
                    with np.load(a.foot_reference / old["artifact"]["path"]) as saved:
                        foot_identity = {
                            key: bool(np.array_equal(arrays[key], saved[key]))
                            for key in saved.files
                        }
                    if not all(foot_identity.values()):
                        raise ValueError(
                            "measuring actual player contacts changed preceding foot dynamics"
                        )
                if (
                    predecessor is not None
                    and a.ball_course == "outbound-north"
                    and arm in ("zero", "role-reactive")
                ):
                    old = next(
                        r
                        for r in predecessor["records"]
                        if r["arm"] == arm and r["repeat"] == repeat
                    )
                    with np.load(a.previous / old["artifact"]["path"]) as saved:
                        legacy_identity = {
                            key: bool(np.array_equal(arrays[key], saved[key]))
                            for key in saved.files
                        }
                    if not all(legacy_identity.values()):
                        raise ValueError("unchanged8-body baselines differ under course extension")
                if repeat:
                    with np.load(a.output / reference["path"]) as previous:
                        own_exact = {
                            key: bool(np.array_equal(value, previous[key]))
                            for key, value in arrays.items()
                        }
                    if not all(own_exact.values()):
                        raise ValueError("fresh8-stream source re-execution differs")
                path = a.output / f"{arm}-{repeat}.npz"
                np.savez_compressed(path, **arrays)
                artifact = dict(path=path.name, hash=digest(path))
                if not repeat:
                    reference = artifact
                players = []
                for i, (team, role, prefix) in enumerate(agents):
                    lo = layouts[prefix]
                    speeds = np.linalg.norm(arrays["qvel"][:, lo.rv : lo.rv + 2], axis=1)
                    players.append(
                        dict(
                            team=team,
                            role=role,
                            body_binding_hash=hash_json(lo.profile()),
                            displacement_m=float(
                                np.linalg.norm(d.qpos[lo.rq : lo.rq + 2] - initial[prefix][:2])
                            ),
                            actual_max_speed_mps=float(speeds.max()),
                            actual_mean_speed_mps=float(speeds.mean()),
                            neural_calls=streams[prefix].neural_calls,
                            foot_contact_steps=int(
                                np.count_nonzero(
                                    np.max(arrays["ball_forces"][:, i, :2], axis=1) > 1.0
                                )
                            ),
                            illegal_contact_steps=int(
                                np.count_nonzero(arrays["ball_forces"][:, i, 2] > 1.0)
                            ),
                            nonzero_command_controls=int(
                                np.count_nonzero(
                                    np.linalg.norm(arrays["commands"][:, i], axis=1) > 0
                                )
                            ),
                            minimum_measured_foot_gap_m=float(
                                arrays["foot_gap_after_m"][:, i].min()
                            ),
                            postcontact_near_slow_controls=int(
                                arrays["postcontact_near_slow"][:, i].sum()
                            ),
                        )
                    )
                row = dict(
                    arm=arm,
                    repeat=repeat,
                    players=players,
                    controls=w.control_steps,
                    native_steps=w.native_steps,
                    healthy_full_horizon=w.control_steps == horizon and not w.failed,
                    body_failed=step["body_failed"],
                    fresh_reexecution_exact=own_exact,
                    legacy_baseline_identity=legacy_identity,
                    preceding_foot_identity=foot_identity,
                    artifact=artifact,
                )
                records.append(row)
                write_once(a.output / f"record-{arm}-{repeat}.json", row)
                print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE8_COACHED_ROLE_BASELINE_MEASURED_NOT_A_MATCH",
                source_head=a.expected_head,
                input_file_hashes=pins,
                records=records,
                native_steps=charged,
                optimizer_updates=0,
                fresh_reexecution_exact=all(
                    all(r["fresh_reexecution_exact"].values()) for r in records if r["repeat"]
                ),
                independent_controller_replay=False,
                not_learned_role_autonomy=True,
                hardware_authorized=False,
                promotion_authorized=False,
                goal_complete=False,
            ),
        )
    except BaseException as error:
        write_once(
            a.output / "failed.json",
            dict(error=repr(error), completed=len(records), known_native_steps=charged),
        )
        raise


if __name__ == "__main__":
    main()
