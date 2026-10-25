"""Native8-body coached role reaction baseline, not learned autonomous4v4 football.

All8 private98->29 streams share one real500Hz MuJoCo clock. An explicit
optional one-leg contact probe changes joint targets, never root/ball forces.
No goals, body resets or ball rewrites are provided by this diagnostic.
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
    p.add_argument("--command-training", type=Path)
    p.add_argument("--command-training-hash")
    p.add_argument("--command-residual", type=Path)
    p.add_argument("--command-residual-hash")
    p.add_argument("--residual-exploration-seed", type=int)
    p.add_argument("--foot-contact-config", type=Path)
    p.add_argument("--foot-contact-config-hash")
    p.add_argument("--experimental-approach-config", type=Path)
    p.add_argument("--experimental-approach-config-hash")
    p.add_argument(
        "--residual-agent",
        choices=tuple(
            team + "_" + role
            for team in ("red", "blue")
            for role in ("goalkeeper", "defender", "playmaker", "finisher")
        ),
    )
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
    if (a.experimental_approach_config is None) != (
        a.experimental_approach_config_hash is None
    ) or (a.experimental_approach_config is not None and a.foot_contact_config is None):
        p.error("experimental approach requires exact config and paired contact-teacher contract")
    if (a.foot_contact_config is None) != (a.foot_contact_config_hash is None) or (
        a.foot_contact_config is not None
        and (not a.clearance_course or a.command_training is not None)
    ):
        p.error(
            "foot-contact teacher requires exact config, clearance, and no learned command model"
        )
    if a.residual_agent is not None and a.command_residual is None:
        p.error("single-player residual scope requires an explicit residual head")
    if (a.command_residual is None) != (a.command_residual_hash is None) or (
        a.command_residual is not None and a.command_training is None
    ):
        p.error("explicit residual head requires the exact retained encoder training")
    if a.residual_exploration_seed is not None and (
        a.command_residual is None or not 0 <= a.residual_exploration_seed < 2**32
    ):
        p.error("bounded explicit residual exploration seed and model required")
    if (a.command_training is None) != (a.command_training_hash is None) or (
        a.command_training is not None and not a.clearance_course
    ):
        p.error("learned command contrast requires exact training and clearance course")
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
    trained = None
    command_predecessor = None
    command_reference = None
    if a.command_training is not None:
        path = a.command_training / "complete.json"
        if path.is_symlink() or digest(path) != a.command_training_hash:
            raise ValueError("exact completed command training required")
        trained = json.loads(path.read_text())
        if (
            trained["status"]
            not in (
                "NATIVE_ROLE_COMMAND_BC_TRAINED_NOT_PHYSICALLY_EVALUATED",
                "NATIVE_ROLE_COMMAND_DAGGER_TRAINED_NOT_PHYSICALLY_EVALUATED",
            )
            or trained["actual_BC_actor_optimizer_updates"] != 512
            or trained["hardware_authorized"] is not False
        ):
            raise ValueError("explicit supervised command training required")
        for key, value in trained["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting training ancestry")
            pins[key] = value
        pins[str(path)] = a.command_training_hash
        plans = [Path(k) for k, v in pins.items() if v == trained["plan_hash"]]
        if len(plans) != 1:
            raise ValueError("exact unique training plan required")
        plan = json.loads(plans[0].read_text())
        reference = next(item for item in plan["sources"] if item["course"] == a.ball_course)
        command_reference = Path(reference["root"])
        path = command_reference / "complete.json"
        if pins.get(str(path)) != reference["hash"] or digest(path) != reference["hash"]:
            raise ValueError("same-course measured clearance baseline required")
        command_predecessor = json.loads(path.read_text())
        for row in command_predecessor["records"]:
            path = command_reference / row["artifact"]["path"]
            if path.is_symlink() or digest(path) != row["artifact"]["hash"]:
                raise ValueError("exact preceding clearance arrays required")
            pins[str(path)] = row["artifact"]["hash"]
        path = a.command_training / trained["policy"]["path"]
        if (
            path.parent.resolve() != a.command_training.resolve()
            or path.is_symlink()
            or digest(path) != trained["policy"]["hash"]
        ):
            raise ValueError("exact confined trained command model required")
        pins[str(path)] = trained["policy"]["hash"]
    foot_predecessor = None
    if a.command_residual is not None:
        if a.command_residual.is_symlink() or digest(a.command_residual) != a.command_residual_hash:
            raise ValueError("exact separate residual head required")
        pins[str(a.command_residual)] = a.command_residual_hash
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
    teacher_config = None
    teacher_parent_root = None
    teacher_parent_report = None
    if a.foot_contact_config is not None:
        if (
            a.foot_contact_config.is_symlink()
            or digest(a.foot_contact_config) != a.foot_contact_config_hash
        ):
            raise ValueError("exact ordinary foot-contact teacher declaration required")
        pins[str(a.foot_contact_config)] = a.foot_contact_config_hash
        teacher_config = json.loads(a.foot_contact_config.read_text())
        if (
            set(teacher_config) != {"schema", "agent", "parameters", "parent_root", "parent_hash"}
            or teacher_config["schema"] != "soccer.native_foot_contact_teacher.v1"
            or teacher_config["agent"] != "red_finisher"
        ):
            raise ValueError("explicit isolated red-finisher teacher contract required")
        teacher_parent_root = Path(teacher_config["parent_root"])
        path = teacher_parent_root / "complete.json"
        if path.is_symlink() or digest(path) != teacher_config["parent_hash"]:
            raise ValueError("exact physical parent comparison required")
        teacher_parent_report = json.loads(path.read_text())
        parent_declaration = json.loads((teacher_parent_root / "declaration.json").read_text())
        if (
            parent_declaration["ball_course"] != a.ball_course
            or not teacher_parent_report["fresh_reexecution_exact"]
            or len(teacher_parent_report["records"]) != 4
            or not all(r["healthy_full_horizon"] for r in teacher_parent_report["records"])
        ):
            raise ValueError("all healthy paired same-course parent records required")
        pins[str(path)] = teacher_config["parent_hash"]
        for key, value in teacher_parent_report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting teacher-parent ancestry")
            pins[key] = value
        for row in teacher_parent_report["records"]:
            path = teacher_parent_root / row["artifact"]["path"]
            if (
                path.parent.resolve() != teacher_parent_root.resolve()
                or path.is_symlink()
                or digest(path) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined physical teacher-parent arrays required")
            pins[str(path)] = row["artifact"]["hash"]
    approach_config = None
    surface_mode = False
    if a.experimental_approach_config is not None:
        if (
            a.experimental_approach_config.is_symlink()
            or digest(a.experimental_approach_config) != a.experimental_approach_config_hash
        ):
            raise ValueError("exact separately declared experimental approach config required")
        pins[str(a.experimental_approach_config)] = a.experimental_approach_config_hash
        approach_config = json.loads(a.experimental_approach_config.read_text())
        surface_mode = (
            approach_config.get("schema")
            == "soccer.native_single_finisher_experimental_approach.v2"
        )
        allowed_fields = {
            "schema",
            "point_root",
            "point_hash",
            "reference_source",
            "reference_hash",
        }
        if surface_mode:
            allowed_fields.add("surface_aware_contact_tracking")
        if (
            set(approach_config) != allowed_fields
            or approach_config["schema"]
            not in (
                "soccer.native_single_finisher_experimental_approach.v1",
                "soccer.native_single_finisher_experimental_approach.v2",
            )
            or (surface_mode and approach_config["surface_aware_contact_tracking"] is not True)
        ):
            raise ValueError("fixed single-finisher experimental approach contract required")
        point_root = Path(approach_config["point_root"])
        path = point_root / "complete.json"
        if path.is_symlink() or digest(path) != approach_config["point_hash"]:
            raise ValueError("exact completed experimental point evidence required")
        point_report = json.loads(path.read_text())
        declaration = json.loads((point_root / "declaration.json").read_text())
        if (
            point_report["status"]
            != "EXPERIMENTAL_VELOCITY_POINTS_MEASURED_NOT_A_DOMAIN_QUALIFICATION"
            or declaration["protocol"] != "native_one_body_experimental_velocity_heading_hold.v2"
            or len(point_report["records"]) != 16
            or not all(
                r["point_feasible"] and r["no_ball_contact"] for r in point_report["records"]
            )
            or not all(
                all(r["fresh_reexecution_exact"].values())
                for r in point_report["records"]
                if r["repeat"]
            )
        ):
            raise ValueError("all preregistered heading-held native points must have passed")
        pins[str(path)] = approach_config["point_hash"]
        for key, value in point_report["input_file_hashes"].items():
            if key in pins and pins[key] != value:
                raise ValueError("conflicting native locomotion ancestry")
            pins[key] = value
        for row in point_report["records"]:
            path = point_root / row["artifact"]["path"]
            if (
                path.parent.resolve() != point_root.resolve()
                or path.is_symlink()
                or digest(path) != row["artifact"]["hash"]
            ):
                raise ValueError("exact confined point-qualification arrays required")
            pins[str(path)] = row["artifact"]["hash"]
        path = Path(approach_config["reference_source"])
        if path.is_symlink() or digest(path) != approach_config["reference_hash"]:
            raise ValueError("exact upstream training reference required")
        pins[str(path)] = approach_config["reference_hash"]
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

    from rosclaw_soccer.providers.g1 import experimental_velocity_envelope as envelope_module
    from rosclaw_soccer.providers.g1.continuous_velocity_stream import ContinuousVelocityStream
    from rosclaw_soccer.providers.g1.experimental_velocity_envelope import (
        ExperimentalVelocityEnvelopeStream,
    )
    from rosclaw_soccer.rsi.experimental_approach_command import experimental_approach_command
    from rosclaw_soccer.rsi.native_foot_contact_teacher import (
        FootContactTeacherConfig,
        NativeFootContactTeacher,
    )
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
    from rosclaw_soccer.rsi.surface_foot_interception import (
        foot_surface_points,
        surface_foot_interception_command,
    )
    from rosclaw_soccer.sim.contracts import hash_json

    if approach_config is not None:
        verified_codec_hashes = {
            v
            for k, v in point_report["input_file_hashes"].items()
            if k.endswith("/providers/g1/experimental_velocity_envelope.py")
        }
        if verified_codec_hashes != {digest(Path(envelope_module.__file__))}:
            raise ValueError("exact experimentally measured observation/actor codec required")

    teacher_parameters = (
        None if teacher_config is None else FootContactTeacherConfig(**teacher_config["parameters"])
    )

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
    if trained is not None:
        arms = ("role-foot-clearance", "role-command-learned")
    if a.command_residual is not None:
        arms = ("role-foot-clearance", "role-command-residual")
    if teacher_config is not None:
        arms = ("role-foot-clearance", "role-foot-contact-teacher")
    if approach_config is not None:
        arms = ("role-foot-clearance", "role-experimental-approach", "role-foot-contact-teacher")
    a.output.mkdir(parents=True)
    write_once(
        a.output / "declaration.json",
        dict(
            protocol="native8_single_player_surface_aware_approach_contact.v11"
            if surface_mode
            else "native8_single_player_accelerated_approach_contact.v10"
            if approach_config is not None
            else "native8_single_player_foot_contact_teacher.v9"
            if teacher_config is not None
            else "native8_single_player_role_residual_contrast.v8"
            if a.residual_agent is not None
            else "native8_parent_role_residual_contrast.v7"
            if a.command_residual is not None
            else "native8_learned_role_command_contrast.v6"
            if trained is not None
            else "native8_measured_clearance_contrast.v5"
            if a.clearance_course
            else "native8_measured_foot_tracking.v4"
            if a.foot_track_course
            else "native8_coached_intercept_course.v3"
            if a.intercept_course
            else "native8_coached_ball_intent_baseline.v1",
            source_head=a.expected_head,
            residual_active_agent=a.residual_agent,
            foot_contact_teacher=teacher_config,
            foot_contact_config_hash=a.foot_contact_config_hash,
            experimental_approach_config=approach_config,
            experimental_approach_config_hash=a.experimental_approach_config_hash,
            approach_nominal_teacher_history_is_own_previous_proposal=approach_config is not None,
            surface_aware_contact_tracking=surface_mode,
            contact_body_force_trace_is_thresholded_above_one_newton=approach_config is not None,
            approach_experimental_speed_norm_mps=2.0 if approach_config is not None else None,
            approach_slew_per_body_component_per_control=0.08
            if approach_config is not None
            else None,
            foot_tracking_slew_describes_nominal_teacher=approach_config is not None,
            experimental_axis_points_do_not_qualify_the_full_command_domain=True
            if approach_config is not None
            else None,
            gait_previous_action_is_previous_base_network_output=teacher_config is not None,
            unselected_players_receive_no_learned_residual=a.residual_agent is not None,
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
            high_level_role_law_is_coached_not_learned=trained is None,
            high_level_role_model_is_shared_and_role_conditioned=trained is not None,
            coached_reference_arm_always_included=True,
            optimizer_updates=0,
            imported_command_training_hash=a.command_training_hash,
            imported_BC_updates=trained["optimizer_updates"] if trained is not None else 0,
            separate_residual_hash=a.command_residual_hash,
            residual_exploration_seed=a.residual_exploration_seed,
            residual_exploration_rho=0.95 if a.residual_exploration_seed is not None else None,
            residual_exploration_stationary_latent_std=0.35
            if a.residual_exploration_seed is not None
            else None,
            residual_legacy_role_actor_commands_are_nominal_pre_parent_clearance=a.command_residual
            is not None,
            residual_legacy_clearance_flags_describe_parent=a.command_residual is not None,
            physical_evaluation_performs_no_optimizer_updates=True,
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
                streams = {}
                for prefix in prefixes:
                    stream_kwargs = dict(
                        expected_model_hash=pins[str(models[0])],
                        expected_config_hash=pins[str(configs[0])],
                        horizon_controls=horizon,
                    )
                    if (
                        approach_config is not None
                        and prefix == "red_finisher/"
                        and arm != "role-foot-clearance"
                    ):
                        streams[prefix] = ExperimentalVelocityEnvelopeStream(
                            models[0],
                            configs[0],
                            reference_source=Path(approach_config["reference_source"]),
                            expected_reference_hash=approach_config["reference_hash"],
                            **stream_kwargs,
                        )
                    else:
                        streams[prefix] = ContinuousVelocityStream(
                            models[0],
                            configs[0],
                            **stream_kwargs,
                        )
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
                previous_nominal_world_commands = np.zeros((8, 2))
                previous_body_commands = np.zeros((8, 3), np.float32)
                command_policy = None
                residual_policy = None
                residual_calls = 0
                contact_teacher = (
                    None
                    if teacher_parameters is None
                    else NativeFootContactTeacher(teacher_parameters)
                )
                exploration_rng = np.random.default_rng(a.residual_exploration_seed)
                exploration_state = np.zeros((8, 3))
                if trained is not None:
                    from rosclaw_soccer.rsi.native_role_command import (
                        NativeRoleCommandPolicy,
                        role_command_observation,
                    )

                    gait = {
                        k: v
                        for k, v in pins.items()
                        if k.endswith(
                            ("/velocity/v0/exported/policy.onnx", "/velocity/v0/params/deploy.yaml")
                        )
                    }
                    binding = hash_json(
                        dict(
                            agents=[
                                dict(team=t, role=r, body_hash=hash_json(layouts[prefix].profile()))
                                for t, r, prefix in agents
                            ],
                            gait=gait,
                        )
                    )
                    command_policy = NativeRoleCommandPolicy(
                        a.command_training / trained["policy"]["path"],
                        expected_hash=trained["policy"]["hash"],
                        expected_binding=binding,
                    )
                    if a.command_residual is not None:
                        from rosclaw_soccer.rsi.native_role_residual import (
                            ROLE_COMMAND_LIMITS,
                            NativeRoleResidualPolicy,
                            guard_role_residual,
                            qualified_role_parent_hash,
                        )

                        residual_policy = NativeRoleResidualPolicy(
                            a.command_residual,
                            expected_hash=a.command_residual_hash,
                            encoder=command_policy,
                            expected_encoder_hash=trained["policy"]["hash"],
                            expected_binding=binding,
                            expected_parent_hash=qualified_role_parent_hash(),
                        )
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
                if trained is not None:
                    controls["role_observations118"], controls["role_actor_commands"] = [], []
                residual_keys = (
                    "residual_context121",
                    "residual_features131",
                    "residual_normalized_actions",
                    "residual_deltas",
                    "residual_parent_commands",
                    "residual_raw_commands",
                    "residual_model_valid",
                    "residual_final_guard_changed",
                    "residual_latent_noise",
                )
                if residual_policy is not None:
                    controls.update({key: [] for key in residual_keys})
                if contact_teacher is not None:
                    controls.update(
                        {
                            "teacher_" + key: []
                            for key in (
                                "nominal_targets29",
                                "requested_delta",
                                "applied_delta",
                                "active",
                                "fallback",
                                "foot_index",
                                "surface_gap",
                                "gap_censored_at_one_meter",
                                "point",
                                "point_local",
                                "foot_velocity",
                                "ball_velocity",
                                "jacobian_hg",
                                "estimated_local_foot_shift",
                                "point_after",
                                "relative_velocity_after",
                                "ball_position_after",
                            )
                        }
                    )
                approach_keys = (
                    "parent_commands",
                    "parent_history_world_commands",
                    "raw_commands",
                    "slewed_commands",
                    "scale",
                    "active",
                    "feasible",
                    "projection_changed",
                    "parent_fallback",
                )
                if approach_config is not None:
                    controls.update({"approach_" + k: [] for k in approach_keys})
                    controls["approach_contact_body_force_gt1"] = []
                if surface_mode:
                    controls.update(
                        {
                            "approach_surface_" + k: []
                            for k in (
                                "active",
                                "selected_foot",
                                "point",
                                "target",
                                "degenerate_ray",
                            )
                        }
                    )
                for frame in range(horizon):
                    if a.foot_track_course:
                        metric.qpos[:] = d.qpos
                        mujoco.mj_kinematics(m, metric)
                        if contact_teacher is not None:
                            metric.qvel[:] = d.qvel
                            mujoco.mj_comPos(m, metric)
                    ball_xy = d.qpos[layout.bq : layout.bq + 2].copy()
                    commands, targets, goals, observations = [], {}, [], []
                    horizons, reachable_flags = [], []
                    clearance_feasible, clearance_constrained = [], []
                    role_observations, role_actor_commands = [], []
                    residual_values = {key: [] for key in residual_keys}
                    approach_values = {key: [] for key in approach_keys}
                    surface_trace = dict(
                        active=False,
                        selected_foot=-1,
                        point=np.zeros(3),
                        target=np.zeros(3),
                        degenerate_ray=False,
                    )
                    if (
                        residual_policy is not None
                        and arm == "role-command-residual"
                        and frame >= 50
                        and a.residual_exploration_seed is not None
                    ):
                        exploration_state = 0.95 * exploration_state + math.sqrt(
                            1 - 0.95**2
                        ) * 0.35 * exploration_rng.normal(size=(8, 3))
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
                            arm
                            in (
                                "role-intercept",
                                "role-foot-track",
                                "role-foot-clearance",
                                "role-command-residual",
                                "role-foot-contact-teacher",
                                "role-experimental-approach",
                            )
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
                                    "role-command-residual",
                                    "role-foot-contact-teacher",
                                    "role-experimental-approach",
                                )
                                else 0.8,
                            )
                        )
                        if (
                            arm
                            in (
                                "role-foot-track",
                                "role-foot-clearance",
                                "role-command-residual",
                                "role-foot-contact-teacher",
                                "role-experimental-approach",
                            )
                            and role == "finisher"
                            and frame >= 50
                            and np.linalg.norm(ball_xy - q[:2]) < 1.0
                        ):
                            if (
                                surface_mode
                                and prefix == "red_finisher/"
                                and arm != "role-foot-clearance"
                            ):
                                command, surface_trace = surface_foot_interception_command(
                                    yaw=yaw,
                                    surfaces=foot_surface_points(m, metric, lo),
                                    ball_position=d.qpos[lo.bq : lo.bq + 3].copy(),
                                    ball_velocity_xy=d.qvel[lo.bv : lo.bv + 2].copy(),
                                    previous_nominal_world_command=previous_nominal_world_commands[
                                        agent_index
                                    ],
                                )
                            else:
                                command, _ = foot_tracking_command(
                                    q[:2],
                                    yaw,
                                    metric.xpos[feet[prefix], :2].copy(),
                                    ball_xy,
                                    d.qvel[lo.bv : lo.bv + 2].copy(),
                                    previous_nominal_world_commands[agent_index]
                                    if approach_config is not None
                                    else previous_world_commands[agent_index],
                                )
                        if command_policy is not None:
                            observation = role_command_observation(
                                d,
                                metric,
                                layouts,
                                team=team,
                                role=role,
                                previous_body_command=previous_body_commands[agent_index],
                            )
                            role_observations.append(observation)
                            if arm == "role-command-learned" and frame >= 50:
                                command = command_policy.propose(observation)
                            role_actor_commands.append(command.copy())
                        feasible, constrained = True, False
                        if (
                            arm
                            in (
                                "role-foot-clearance",
                                "role-command-learned",
                                "role-command-residual",
                                "role-foot-contact-teacher",
                                "role-experimental-approach",
                            )
                            and frame >= 50
                        ):
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
                        if residual_policy is not None:
                            from rosclaw.control.parent_residual import parent_context

                            parent_command = command.copy()
                            context = parent_context(
                                observation, parent_command, ROLE_COMMAND_LIMITS
                            )
                            features, action, delta = (
                                np.zeros(131, np.float32),
                                np.zeros(3),
                                np.zeros(3),
                            )
                            raw_command, changed, valid = (
                                parent_command.astype(np.float64),
                                False,
                                False,
                            )
                            noise = np.zeros(3)
                            if (
                                arm == "role-command-residual"
                                and frame >= 50
                                and (
                                    a.residual_agent is None
                                    or team + "_" + role == a.residual_agent
                                )
                            ):
                                noise = exploration_state[agent_index].copy()
                                proposal = residual_policy.propose(
                                    observation, parent_command, latent_noise=noise
                                )
                                residual_calls += 1
                                valid = True
                                features, action, delta, raw_command = (
                                    proposal.features131,
                                    proposal.normalized_action,
                                    proposal.delta,
                                    proposal.raw_command,
                                )
                                # Separate final guard: legacy flags above remain the
                                # parent's, never falsely called candidate projection.
                                command, _, changed = guard_role_residual(
                                    proposal,
                                    yaw,
                                    neighbors,
                                    maximum_speed_mps=1.0 if role == "finisher" else 0.8,
                                )
                            for key, value in zip(
                                residual_keys,
                                (
                                    context,
                                    features,
                                    action,
                                    delta,
                                    parent_command,
                                    raw_command,
                                    valid,
                                    changed,
                                    noise,
                                ),
                                strict=True,
                            ):
                                residual_values[key].append(value)
                        clearance_feasible.append(feasible)
                        clearance_constrained.append(constrained)
                        if approach_config is not None:
                            parent_command = command.copy()
                            parent_history = previous_nominal_world_commands[agent_index].copy()
                            parent_c, parent_s = math.cos(yaw), math.sin(yaw)
                            previous_nominal_world_commands[agent_index] = [
                                parent_c * command[0] - parent_s * command[1],
                                parent_s * command[0] + parent_c * command[1],
                            ]
                            effect_active = (
                                prefix == "red_finisher/"
                                and arm != "role-foot-clearance"
                                and frame >= 50
                            )
                            effect = dict(
                                raw_command=command.astype(np.float64),
                                slewed_command=command.astype(np.float64),
                                scale=1.0,
                                feasible=True,
                                projection_changed=False,
                                parent_fallback=False,
                            )
                            if effect_active:
                                effect = experimental_approach_command(
                                    command,
                                    previous_body_commands[agent_index],
                                    root_xy=q[:2].copy(),
                                    ball_xy=ball_xy.copy(),
                                    yaw=yaw,
                                    neighbor_offsets=neighbors,
                                )
                                command = effect["command"]
                            for key, value in zip(
                                approach_keys,
                                (
                                    parent_command,
                                    parent_history,
                                    effect["raw_command"],
                                    effect["slewed_command"],
                                    effect["scale"],
                                    effect_active,
                                    effect["feasible"],
                                    effect["projection_changed"],
                                    effect["parent_fallback"],
                                ),
                                strict=True,
                            ):
                                approach_values[key].append(value)
                        previous_body_commands[agent_index] = command
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
                    if contact_teacher is not None:
                        selected = "red_finisher/"
                        nominal_targets = np.stack([targets[p].copy() for p in prefixes])
                        effect = contact_teacher.propose(
                            model=m,
                            snapshot=metric,
                            layout=layouts[selected],
                            nominal=targets[selected],
                            ball_geom=int(gids[0]),
                            foot_bodies=feet[selected],
                            enabled=arm == "role-foot-contact-teacher" and frame >= 50,
                            frame=frame,
                        )
                        targets[selected] = effect["target"]
                        controls["teacher_nominal_targets29"].append(nominal_targets)
                        for key, value in effect.items():
                            if "teacher_" + key in controls:
                                controls["teacher_" + key].append(np.asarray(value).copy())
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
                    if trained is not None:
                        controls["role_observations118"].append(role_observations)
                        controls["role_actor_commands"].append(role_actor_commands)
                    if residual_policy is not None:
                        for key in residual_keys:
                            controls[key].append(residual_values[key])
                    if approach_config is not None:
                        for key in approach_keys:
                            controls["approach_" + key].append(approach_values[key])
                    if surface_mode:
                        for key, value in surface_trace.items():
                            controls["approach_surface_" + key].append(np.asarray(value).copy())
                    step = w.advance(targets)
                    charged += 10
                    if approach_config is not None:
                        by_body = np.zeros(m.nbody)
                        for body_forces in w.last_ball_contact_bodies.values():
                            for name, force in body_forces.items():
                                by_body[m.body(name).id] = force
                        reconstructed = np.zeros((8, 3))
                        for body, (owner, part) in w.contact_owners.items():
                            reconstructed[owner, part] = max(
                                reconstructed[owner, part], by_body[body]
                            )
                        actual = step["ball_forces"].max(axis=0)
                        if not np.array_equal(reconstructed, np.where(actual > 1, actual, 0)):
                            raise ValueError(
                                "thresholded per-body physical contact provenance differs"
                            )
                        controls["approach_contact_body_force_gt1"].append(by_body)
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
                    if contact_teacher is not None:
                        body = feet["red_finisher/"][max(0, effect["foot_index"])]
                        point_after = (
                            metric.xpos[body]
                            + metric.xmat[body].reshape(3, 3) @ effect["point_local"]
                        )
                        jac = np.zeros((3, m.nv))
                        mujoco.mj_jac(m, metric, jac, None, point_after, body)
                        controls["teacher_point_after"].append(point_after.copy())
                        controls["teacher_relative_velocity_after"].append(
                            jac @ metric.qvel - d.qvel[layout.bv : layout.bv + 3]
                        )
                        controls["teacher_ball_position_after"].append(
                            d.qpos[layout.bq : layout.bq + 3].copy()
                        )
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
                teacher_identity = None
                teacher_parent_identity = None
                if contact_teacher is not None:
                    selected_index = prefixes.index("red_finisher/")
                    other_indices = [i for i in range(8) if i != selected_index]
                    teacher_identity = dict(
                        unselected_seven_targets_exact_base=bool(
                            np.array_equal(
                                arrays["targets29"][:, other_indices].view(np.uint64),
                                arrays["teacher_nominal_targets29"][:, other_indices].view(
                                    np.uint64
                                ),
                            )
                        ),
                        selected_applied_delta_exact=bool(
                            np.array_equal(
                                arrays["targets29"][:, selected_index]
                                - arrays["teacher_nominal_targets29"][:, selected_index],
                                arrays["teacher_applied_delta"],
                            )
                        ),
                        nonleg_targets_unchanged=bool(
                            not np.any(arrays["teacher_applied_delta"][:, 12:])
                        ),
                        delta_bound_pass=bool(
                            np.max(abs(arrays["teacher_applied_delta"]))
                            <= teacher_parameters.maximum_delta_rad + 1e-12
                        ),
                    )
                    if not all(teacher_identity.values()):
                        raise ValueError("single-player foot-contact target authority violated")
                    if arm == "role-foot-clearance":
                        old = next(
                            r
                            for r in teacher_parent_report["records"]
                            if r["arm"] == arm and r["repeat"] == repeat
                        )
                        with np.load(teacher_parent_root / old["artifact"]["path"]) as original:
                            teacher_parent_identity = {
                                k: bool(np.array_equal(v, original[k]))
                                for k, v in arrays.items()
                                if not k.startswith(("teacher_", "approach_"))
                            }
                    elif teacher_parameters.strength == 0:
                        with np.load(a.output / f"role-foot-clearance-{repeat}.npz") as original:
                            teacher_parent_identity = {
                                k: bool(np.array_equal(v, original[k]))
                                for k, v in arrays.items()
                                if not k.startswith(("teacher_", "approach_"))
                            }
                    if teacher_parent_identity is not None and not all(
                        teacher_parent_identity.values()
                    ):
                        raise ValueError(
                            "teacher measurement or zero strength changed parent physics"
                        )
                approach_authority = None
                if approach_config is not None:
                    selected_index = prefixes.index("red_finisher/")
                    mask = np.zeros((w.control_steps, 8), np.bool_)
                    if arm != "role-foot-clearance":
                        mask[50:, selected_index] = True
                    others = [i for i in range(8) if i != selected_index]
                    actual = arrays["commands"]
                    nonfallback = mask & ~arrays["approach_parent_fallback"]
                    slew = actual - np.concatenate((np.zeros((1, 8, 3), np.float32), actual[:-1]))
                    approach_authority = dict(
                        only_selected_controller_active=bool(
                            np.array_equal(arrays["approach_active"], mask)
                        ),
                        unselected_seven_commands_exact_parent=bool(
                            np.array_equal(
                                actual[:, others].view(np.uint32),
                                arrays["approach_parent_commands"][:, others].view(np.uint32),
                            )
                        ),
                        actual_experimental_domain_pass=bool(
                            np.all(
                                (actual[:, selected_index, 0] >= -1)
                                & (actual[:, selected_index, 0] <= 2)
                            )
                            and np.all(abs(actual[:, selected_index, 1:]) <= 1)
                            and np.all(
                                np.linalg.norm(actual[:, selected_index, :2], axis=1) <= 2 + 1e-6
                            )
                        ),
                        nonfallback_slew_pass=bool(
                            not nonfallback.any() or np.max(abs(slew[nonfallback])) <= 0.08 + 1e-6
                        ),
                        explicit_parent_fallback=bool(
                            np.array_equal(
                                actual[arrays["approach_parent_fallback"]],
                                arrays["approach_parent_commands"][
                                    arrays["approach_parent_fallback"]
                                ],
                            )
                        ),
                    )
                    if not all(approach_authority.values()):
                        raise ValueError("actual experimental approach authority or scope changed")
                isolated_identity = None
                if a.residual_agent is not None and residual_policy is not None:
                    selected = np.array(
                        [team + "_" + role == a.residual_agent for team, role, _ in agents]
                    )
                    expected_valid = np.zeros((w.control_steps, 8), np.bool_)
                    if arm == "role-command-residual":
                        expected_valid[50:, selected] = True
                    unselected = ~selected
                    isolated_identity = dict(
                        model_called_only_for_declared_player=bool(
                            np.array_equal(arrays["residual_model_valid"], expected_valid)
                            and residual_calls == int(expected_valid.sum())
                        ),
                        unselected_commands_exact_parent=bool(
                            np.array_equal(
                                arrays["commands"][:, unselected].view(np.uint32),
                                arrays["residual_parent_commands"][:, unselected].view(np.uint32),
                            )
                        ),
                        unselected_residuals_noise_and_features_zero=all(
                            not np.any(arrays[key][:, unselected])
                            for key in (
                                "residual_features131",
                                "residual_normalized_actions",
                                "residual_deltas",
                                "residual_latent_noise",
                            )
                        ),
                    )
                    if not all(isolated_identity.values()):
                        raise ValueError("single-player residual isolation violated")
                own_exact = None
                legacy_identity = None
                foot_identity = None
                command_identity = None
                residual_identity = None
                if (
                    arm == "role-command-residual"
                    and residual_policy.metadata["update_index"] == 0
                    and a.residual_exploration_seed is None
                ):
                    with np.load(a.output / f"role-foot-clearance-{repeat}.npz") as original:
                        residual_identity = {
                            key: bool(np.array_equal(arrays[key], original[key]))
                            for key in original.files
                            if not key.startswith("residual_")
                        }
                    if not all(residual_identity.values()):
                        raise ValueError("zero residual changed the qualified native parent")
                if command_predecessor is not None and arm == "role-foot-clearance":
                    old = next(
                        r
                        for r in command_predecessor["records"]
                        if r["arm"] == arm and r["repeat"] == repeat
                    )
                    with np.load(command_reference / old["artifact"]["path"]) as saved:
                        command_identity = {
                            key: bool(np.array_equal(arrays[key], saved[key]))
                            for key in saved.files
                        }
                    if not all(command_identity.values()):
                        raise ValueError("command observations changed preceding clearance physics")
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
                    preceding_clearance_identity=command_identity,
                    zero_residual_parent_identity=residual_identity,
                    actual_residual_model_calls=residual_calls,
                    residual_active_agent=a.residual_agent,
                    isolated_player_identity=isolated_identity,
                    teacher_target_authority=teacher_identity,
                    teacher_parent_identity=teacher_parent_identity,
                    experimental_approach_authority=approach_authority,
                    gait_profile_hashes={p: str(s.contract_hash) for p, s in streams.items()},
                    contact_body_names=[m.body(i).name for i in range(m.nbody)]
                    if approach_config is not None
                    else None,
                    teacher_active_controls=int(arrays["teacher_active"].sum())
                    if contact_teacher is not None
                    else 0,
                    teacher_fallback_controls=int(arrays["teacher_fallback"].sum())
                    if contact_teacher is not None
                    else 0,
                    residual_update_index=residual_policy.metadata["update_index"]
                    if residual_policy is not None
                    else None,
                    actual_command_model_calls=8 * max(0, w.control_steps - 50)
                    if arm == "role-command-learned"
                    else 0,
                    artifact=artifact,
                )
                records.append(row)
                write_once(a.output / f"record-{arm}-{repeat}.json", row)
                print(json.dumps(row), flush=True)
        check_pins(pins)
        write_once(
            a.output / "complete.json",
            dict(
                status="NATIVE8_ACCELERATED_APPROACH_CONTACT_MEASURED_NOT_LEARNED"
                if approach_config is not None
                else "NATIVE8_CONTACT_TEACHER_AUTHORITY_MEASURED_NOT_LEARNED"
                if teacher_config is not None
                else "NATIVE8_LEARNED_ROLE_COMMANDS_MEASURED_NOT_A_MATCH"
                if trained is not None
                else "NATIVE8_COACHED_ROLE_BASELINE_MEASURED_NOT_A_MATCH",
                source_head=a.expected_head,
                input_file_hashes=pins,
                records=records,
                native_steps=charged,
                optimizer_updates=0,
                fresh_reexecution_exact=all(
                    all(r["fresh_reexecution_exact"].values()) for r in records if r["repeat"]
                ),
                independent_controller_replay=False,
                role_commands_include_trained_model=trained is not None,
                not_an_autonomous_match_policy=True,
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
