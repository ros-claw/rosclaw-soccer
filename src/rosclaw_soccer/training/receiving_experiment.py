"""Reusable, simulation-only execution of the frozen R0 receiving classroom.

Experiment drivers supply external paths and persist evidence. This adapter
does not import historical experiment scripts, patch agent methods, update
weights, write artifacts, or create a runtime/hardware execution path.
"""

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.providers.g1.qualified_receiving_student import QualifiedReceivingStudent
from rosclaw_soccer.providers.g1.receiving_foot_capture import ReceivingFootCaptureTeacher
from rosclaw_soccer.providers.g1.receiving_sonic import (
    ReceivingSonicBallFollowOption,
    ReceivingSonicFeedbackOption,
    ReceivingSonicOption,
    RecordingReceivingSonicBallFollowOption,
    RecordingReceivingSonicOption,
)
from rosclaw_soccer.providers.g1.sonic_command_scale import SonicCommandScaleSchedule
from rosclaw_soccer.providers.g1.sonic_latent import SonicLatentSchedule
from rosclaw_soccer.providers.g1.sonic_pose_reference import SonicPoseReference
from rosclaw_soccer.skills.team.independent_team_world import (
    IndependentTeamWorldResult,
    IndependentTeamWorldScenario,
    simulate_independent_team_world,
)
from rosclaw_soccer.training.contact_teacher_ablation import ContactTeacherSuppression
from rosclaw_soccer.training.contact_teacher_evidence import inspect_teacher_suppression
from rosclaw_soccer.training.continuous_match_residual_ppo import collection_fixture
from rosclaw_soccer.training.receiving_classroom import (
    coached_receiving_cells,
    r0_receiving_configuration,
)
from rosclaw_soccer.training.receiving_feedback import (
    ReceivingFeedbackProvider,
    ReceivingFeedbackSlot,
)
from rosclaw_soccer.training.receiving_oracle_schedule import ReceivingOracleSchedule
from rosclaw_soccer.training.receiving_phase_feedback import ReceivingPhaseReference
from rosclaw_soccer.training.role_receiving_courses import (
    ROSTER,
    ReceivingCourse,
    receiving_ball_launch,
)


def simulate_r0_receiving_course(
    *,
    asset_root: Path,
    reference_policy_path: Path,
    course: ReceivingCourse,
    scenario_id: str,
    checkpoint_frame: int = 0,
    capture_support: bool = False,
    capture_oracle_authority: bool = False,
    capture_locomotion_memory: bool = False,
    oracle: ReceivingOracleSchedule | None = None,
    phase_reference: ReceivingPhaseReference | None = None,
    feedback_provider: ReceivingFeedbackProvider | None = None,
    suppression: ContactTeacherSuppression | None = None,
    sonic_model_root: Path | None = None,
    sonic_start_frame: int = 0,
    sonic_velocity_scale: float = 1.0,
    sonic_planner_seed: int = 920101,
    sonic_latent_schedule: SonicLatentSchedule | None = None,
    sonic_command_scale_schedule: SonicCommandScaleSchedule | None = None,
    sonic_pose_reference: SonicPoseReference | None = None,
    sonic_command_replanning: bool = False,
    feedback_actor_path: Path | None = None,
    capture_sonic_targets: bool = False,
    sonic_ball_follow_gain: float | None = None,
    sonic_ball_follow_fast_replan: bool = False,
    sonic_ball_follow_post_touch_chase: bool = False,
    sonic_ball_follow_brake_distance_m: float | None = None,
    sonic_ball_follow_brake_axis: str = "xy",
    sonic_ball_follow_post_touch_target_distance_m: float | None = None,
    sonic_ball_follow_post_touch_speed_limit_mps: float | None = None,
    research_student_handoff: bool = False,
    capture_ball_follow_targets: bool = False,
    capture_live_motor_observations: bool = False,
    receiving_student: QualifiedReceivingStudent | None = None,
    receiving_student_probe_torque_nm: float = 0.0,
    receiving_student_hip_roll_offset_rad: float = 0.0,
    receiving_student_contact_impedance_scale: float = 1.0,
    receiving_student_posttouch_brake_nm: float = 0.0,
    receiving_foot_capture_teacher: ReceivingFootCaptureTeacher | None = None,
    research_coupled_teacher: bool = False,
    capture_team_motor_targets: bool = False,
    research_control_frame_limit: int | None = None,
) -> tuple[IndependentTeamWorldResult, dict[str, NDArray[Any]]]:
    """Run one frozen course with private controller state and unchanged guards.

    A query checkpoint can be after contact for feedback reconstruction; the
    separate M0 bank audit is responsible for its pre-contact entry requirement.
    Only the focal player can receive an experimental oracle or motor option.
    """
    if not isinstance(course, ReceivingCourse) or course.agent_id not in ROSTER:
        raise ValueError("typed focal receiving course required")
    if type(sonic_command_replanning) is not bool or (
        sonic_command_replanning
        and (sonic_latent_schedule is not None or sonic_pose_reference is not None)
    ):
        raise ValueError("command-event probe requires unmixed SONIC navigation")
    if feedback_provider is not None:
        if oracle is None or phase_reference is not None:
            raise ValueError("feedback must bind one schedule without competing phase feedback")
        feedback_slot = ReceivingFeedbackSlot(feedback_provider, oracle)
        if feedback_slot.requires_locomotion_memory and capture_locomotion_memory is not True:
            raise ValueError("recurrent feedback requires explicit recorded locomotion memory")
    if phase_reference is not None:
        if not isinstance(phase_reference, ReceivingPhaseReference):
            raise ValueError("typed phase reference required")
        phase_reference.__post_init__()
        if (
            oracle is None
            or not isinstance(oracle, ReceivingOracleSchedule)
            or oracle.substrate != "A0_leg12"
            or phase_reference.schedule_hash != oracle.contract_hash
            or phase_reference.start_frame != oracle.start_frame
        ):
            raise ValueError("phase reference must bind this A0 schedule and entry")
    if type(capture_oracle_authority) is not bool or (capture_oracle_authority and oracle is None):
        raise ValueError("authority capture requires a receiving oracle")
    if type(capture_locomotion_memory) is not bool or (
        capture_locomotion_memory and oracle is None
    ):
        raise ValueError("locomotion memory capture requires a receiving oracle")
    if (
        type(checkpoint_frame) is not int
        or not 0 <= checkpoint_frame < 300
        or type(capture_support) is not bool
    ):
        raise ValueError("checkpoint must occur inside the six-second reference course")
    if oracle is not None:
        if not isinstance(oracle, ReceivingOracleSchedule):
            raise ValueError("typed oracle schedule required")
        oracle.__post_init__()
        if oracle.agent_id != course.agent_id or oracle.start_frame >= 300:
            raise ValueError("oracle must execute for this focal player inside the course")
        if (oracle.substrate == "A3_sonic_residual") != (sonic_model_root is not None):
            raise ValueError("oracle and frozen SONIC ownership must match")
        if oracle.substrate == "A3_sonic_residual" and oracle.start_frame != sonic_start_frame:
            raise ValueError("SONIC and residual must share their entry frame")
    if suppression is not None:
        if not isinstance(suppression, ContactTeacherSuppression):
            raise ValueError("typed suppression contract required")
        suppression.__post_init__()
        if suppression.agent_id != course.agent_id or suppression.start_frame >= 300:
            raise ValueError("suppression must bind this focal player inside the course")
    if sonic_model_root is None and (
        sonic_start_frame != 0
        or sonic_velocity_scale != 1.0
        or sonic_planner_seed != 920101
        or sonic_latent_schedule is not None
        or sonic_command_scale_schedule is not None
        or sonic_pose_reference is not None
        or sonic_command_replanning
        or feedback_actor_path is not None
        or capture_sonic_targets
        or sonic_ball_follow_gain is not None
        or sonic_ball_follow_fast_replan
        or sonic_ball_follow_post_touch_chase
        or sonic_ball_follow_brake_distance_m is not None
        or sonic_ball_follow_post_touch_target_distance_m is not None
        or sonic_ball_follow_post_touch_speed_limit_mps is not None
        or capture_ball_follow_targets
        or capture_live_motor_observations
    ):
        raise ValueError("SONIC parameters without a frozen model are invalid")
    if sonic_model_root is not None and (
        type(sonic_start_frame) is not int or not 0 <= sonic_start_frame < 300
    ):
        raise ValueError("SONIC must enter inside the reference course")
    if type(capture_sonic_targets) is not bool or (
        capture_sonic_targets and (feedback_actor_path is not None or sonic_start_frame != 0)
    ):
        raise ValueError("pure frame-zero SONIC motor recording required")
    if sonic_ball_follow_gain is not None and (
        type(sonic_ball_follow_gain) not in (int, float)
        or not np.isfinite(sonic_ball_follow_gain)
        or not 0 < sonic_ball_follow_gain <= 1
        or feedback_actor_path is not None
        or capture_sonic_targets
        or sonic_start_frame != 0
        or sonic_velocity_scale != 1.0
        or sonic_command_scale_schedule is not None
        or sonic_latent_schedule is not None
        or sonic_pose_reference is not None
    ):
        raise ValueError("unmixed frame-zero ball-follow probe required")
    if type(sonic_ball_follow_fast_replan) is not bool or (
        sonic_ball_follow_fast_replan and sonic_ball_follow_gain is None
    ):
        raise ValueError("fast replan requires an explicit ball-follow probe")
    if type(sonic_ball_follow_post_touch_chase) is not bool or (
        sonic_ball_follow_post_touch_chase and not sonic_ball_follow_fast_replan
    ):
        raise ValueError("post-touch chase requires an explicit fast-replan follow probe")
    if sonic_ball_follow_brake_distance_m is not None and (
        type(sonic_ball_follow_brake_distance_m) not in (int, float)
        or not np.isfinite(sonic_ball_follow_brake_distance_m)
        or not 0.45 <= sonic_ball_follow_brake_distance_m <= 0.90
        or not sonic_ball_follow_fast_replan
    ):
        raise ValueError("early receiving brake requires bounded fast-replan follow probe")
    if sonic_ball_follow_brake_axis not in ("xy", "x") or (
        sonic_ball_follow_brake_axis == "x"
        and (
            sonic_ball_follow_brake_distance_m != 0.65
            or receiving_foot_capture_teacher is None
            or receiving_student is None
            or research_control_frame_limit != 130
            or sonic_ball_follow_post_touch_chase
        )
    ):
        raise ValueError("longitudinal-only brake requires fixed SIM_ONLY foot-capture course")
    if (
        sonic_ball_follow_post_touch_target_distance_m is not None
        or sonic_ball_follow_post_touch_speed_limit_mps is not None
    ) and (
        not research_student_handoff
        or sonic_ball_follow_post_touch_target_distance_m not in (0.42, 0.45)
        or sonic_ball_follow_post_touch_speed_limit_mps != 0.35
    ):
        raise ValueError("predeclared SIM_ONLY guarded post-touch follow parameters required")
    if type(capture_ball_follow_targets) is not bool or (
        capture_ball_follow_targets
        and (
            sonic_ball_follow_gain is None
            or sonic_ball_follow_fast_replan
            or sonic_ball_follow_post_touch_chase
            or sonic_ball_follow_brake_distance_m is not None
            or capture_sonic_targets
        )
    ):
        raise ValueError("pure measured-ball follow teacher capture required")
    if (
        type(capture_live_motor_observations) is not bool
        or capture_live_motor_observations
        and (
            receiving_student is None
            or research_control_frame_limit != 130
            or sonic_ball_follow_gain != 0.75
            or sonic_ball_follow_fast_replan
            or sonic_ball_follow_post_touch_chase
            or sonic_ball_follow_brake_distance_m is not None
            or capture_ball_follow_targets
            or capture_team_motor_targets
        )
    ):
        raise ValueError("read-only live SONIC tape requires fixed 130-frame student course")
    if (
        type(research_student_handoff) is not bool
        or research_student_handoff
        and (
            receiving_student is None
            or research_control_frame_limit != 130
            or sonic_ball_follow_gain != 0.75
            or not sonic_ball_follow_fast_replan
            or not sonic_ball_follow_post_touch_chase
            or sonic_ball_follow_brake_distance_m != 0.65
            or receiving_student_hip_roll_offset_rad != -0.06
            or receiving_student_contact_impedance_scale != 1.0
            or receiving_student_posttouch_brake_nm != 0.0
            or (sonic_ball_follow_post_touch_target_distance_m is None)
            != (sonic_ball_follow_post_touch_speed_limit_mps is None)
        )
    ):
        raise ValueError("bounded SIM_ONLY student handoff requires fixed live short course")
    if receiving_student is not None and (
        not isinstance(receiving_student, QualifiedReceivingStudent)
        or sonic_model_root is None
        or sonic_ball_follow_gain != 0.75
        or sonic_start_frame != 0
        or sonic_ball_follow_fast_replan
        and not (
            research_student_handoff
            or receiving_foot_capture_teacher is not None
            and sonic_ball_follow_brake_distance_m == 0.65
            and not sonic_ball_follow_post_touch_chase
        )
        or sonic_ball_follow_post_touch_chase
        and not research_student_handoff
        or sonic_ball_follow_brake_distance_m is not None
        and not (
            research_student_handoff
            or receiving_foot_capture_teacher is not None
            and sonic_ball_follow_fast_replan
            and sonic_ball_follow_brake_distance_m == 0.65
            and not sonic_ball_follow_post_touch_chase
        )
        or capture_ball_follow_targets
        or feedback_actor_path is not None
        or oracle is not None
        or feedback_provider is not None
    ):
        raise ValueError("qualified student requires unmixed 0.75 SONIC receiving foundation")
    if receiving_foot_capture_teacher is not None and (
        not isinstance(receiving_foot_capture_teacher, ReceivingFootCaptureTeacher)
        or receiving_foot_capture_teacher.activation_ceiling != "SIM_ONLY"
        or receiving_student is None
        or research_control_frame_limit != 130
        or receiving_student_probe_torque_nm != 0.0
        or receiving_student_hip_roll_offset_rad != 0.0
        or receiving_student_contact_impedance_scale != 1.0
        or receiving_student_posttouch_brake_nm != 0.0
        or research_student_handoff
        or sonic_ball_follow_brake_distance_m is not None
        and (
            sonic_ball_follow_brake_distance_m != 0.65
            or not sonic_ball_follow_fast_replan
            or sonic_ball_follow_post_touch_chase
        )
    ):
        raise ValueError("bounded unmixed SIM_ONLY live foot-capture probe required")
    if (
        type(receiving_student_probe_torque_nm) not in (int, float)
        or not np.isfinite(receiving_student_probe_torque_nm)
        or abs(receiving_student_probe_torque_nm) > 1.0
        or receiving_student_probe_torque_nm != 0.0
        and receiving_student is None
    ):
        raise ValueError("bounded student torque probe requires a qualified receiving student")
    if (
        type(receiving_student_hip_roll_offset_rad) not in (int, float)
        or not np.isfinite(receiving_student_hip_roll_offset_rad)
        or abs(receiving_student_hip_roll_offset_rad) > 0.08
        or receiving_student_hip_roll_offset_rad != 0.0
        and (receiving_student is None or receiving_student_probe_torque_nm != 0.0)
    ):
        raise ValueError("bounded student hip probe requires an unmixed qualified actor")
    if research_control_frame_limit is not None and (
        type(research_control_frame_limit) is not int
        or research_control_frame_limit != 130
        or receiving_student is None
        or capture_team_motor_targets
    ):
        raise ValueError("SIM_ONLY live short receiving course requires one student and 130 frames")
    if (
        type(receiving_student_contact_impedance_scale) is not float
        or receiving_student_contact_impedance_scale not in (0.70, 0.85, 1.0)
        or receiving_student_contact_impedance_scale != 1.0
        and (
            receiving_student is None
            or receiving_student_hip_roll_offset_rad != -0.06
            or research_control_frame_limit != 130
        )
    ):
        raise ValueError("bounded SIM_ONLY impedance course requires fixed live short parent")
    if (
        type(receiving_student_posttouch_brake_nm) is not float
        or receiving_student_posttouch_brake_nm not in (-2.0, 0.0, 2.0)
        or receiving_student_posttouch_brake_nm != 0.0
        and (
            receiving_student is None
            or receiving_student_hip_roll_offset_rad != -0.06
            or receiving_student_contact_impedance_scale != 1.0
            or research_control_frame_limit != 130
        )
    ):
        raise ValueError("bounded SIM_ONLY post-touch brake course requires live short parent")
    if (
        type(research_coupled_teacher) is not bool
        or research_coupled_teacher
        and (
            receiving_student is not None
            or sonic_model_root is None
            or sonic_ball_follow_gain != 0.75
            or sonic_start_frame != 0
            or sonic_ball_follow_fast_replan
            or sonic_ball_follow_post_touch_chase
            or sonic_ball_follow_brake_distance_m is not None
            or capture_ball_follow_targets
            or feedback_actor_path is not None
            or oracle is not None
            or feedback_provider is not None
        )
    ):
        raise ValueError("privileged coupled teacher requires unmixed 0.75 SONIC training world")
    if (
        type(capture_team_motor_targets) is not bool
        or capture_team_motor_targets
        and (
            checkpoint_frame != 45
            or research_coupled_teacher
            or sonic_model_root is None
            or sonic_ball_follow_gain != 0.75
        )
    ):
        raise ValueError("read-only team motor capture requires frame-45 SONIC physics")
    # Validate physical launch values before allocating/loading the simulator.
    receiving_ball_launch(course, origin=(0.0, 0.0, 0.0), radius_m=0.115)
    fixture = collection_fixture(asset_root, keeper_preview=True)
    policy = NearBallResidualPolicy.load(reference_policy_path)
    cells = coached_receiving_cells(fixture.cells, focal_agent_id=course.agent_id)
    player = next(player for player in fixture.players if player.agent_id == course.agent_id)
    position, velocity = receiving_ball_launch(
        course, origin=player.origin_m, radius_m=fixture.goal.ball_radius_m
    )
    world, teacher = r0_receiving_configuration()
    motors: dict[str, ReceivingSonicOption] = {}
    if sonic_model_root is not None:
        world = replace(world, motor_idle_residual_fallback=True)
        # Instantiate a fresh backend for every simulation, never reuse its history.
        if sonic_ball_follow_gain is not None:
            ball_option_type = (
                RecordingReceivingSonicBallFollowOption
                if capture_ball_follow_targets or capture_live_motor_observations
                else ReceivingSonicBallFollowOption
            )
            motors[course.agent_id] = ball_option_type(
                sonic_model_root,
                course.agent_id,
                start_frame=sonic_start_frame,
                response_gain=sonic_ball_follow_gain,
                fast_replan=sonic_ball_follow_fast_replan,
                post_touch_chase=sonic_ball_follow_post_touch_chase,
                brake_distance_m=sonic_ball_follow_brake_distance_m,
                brake_axis=sonic_ball_follow_brake_axis,
                post_touch_target_distance_m=(
                    sonic_ball_follow_post_touch_target_distance_m
                    if sonic_ball_follow_post_touch_target_distance_m is not None
                    else 0.30
                ),
                post_touch_speed_limit_mps=sonic_ball_follow_post_touch_speed_limit_mps,
            )
        else:
            option_type = (
                ReceivingSonicFeedbackOption
                if feedback_actor_path is not None
                else RecordingReceivingSonicOption
                if capture_sonic_targets
                else ReceivingSonicOption
            )
            extra = (
                {"feedback_actor_path": feedback_actor_path}
                if feedback_actor_path is not None
                else {}
            )
            motors[course.agent_id] = option_type(
                sonic_model_root,
                course.agent_id,
                start_frame=sonic_start_frame,
                velocity_scale=sonic_velocity_scale,
                planner_seed=sonic_planner_seed,
                latent_schedule=sonic_latent_schedule,
                command_scale_schedule=sonic_command_scale_schedule,
                pose_reference=sonic_pose_reference,
                experimental_command_replanning=sonic_command_replanning,
                **extra,
            )
    result, trace = simulate_independent_team_world(
        asset_root=asset_root,
        roster=fixture.roster,
        cells=cells,
        players=fixture.players,
        scenario=IndependentTeamWorldScenario(scenario_id, position, velocity, course.seed),
        goal=fixture.goal,
        config=world,
        near_ball_policy=policy,
        contact_teacher_config=teacher,
        near_ball_seed=course.seed,
        near_ball_explore=False,
        motor_options=motors,
        receiving_students=(
            {course.agent_id: receiving_student} if receiving_student is not None else None
        ),
        receiving_student_probe_torque_nm=receiving_student_probe_torque_nm,
        receiving_student_hip_roll_offset_rad=receiving_student_hip_roll_offset_rad,
        receiving_student_contact_impedance_scale=receiving_student_contact_impedance_scale,
        receiving_student_posttouch_brake_nm=receiving_student_posttouch_brake_nm,
        receiving_foot_capture_teacher=receiving_foot_capture_teacher,
        research_coupled_teacher_agent_id=(course.agent_id if research_coupled_teacher else None),
        receiving_oracle=oracle,
        receiving_phase_reference=phase_reference,
        receiving_feedback=feedback_provider,
        capture_oracle_authority=capture_oracle_authority,
        capture_locomotion_memory=capture_locomotion_memory,
        contact_teacher_suppression=suppression,
        capture_initial_physics=True,
        capture_initial_support=capture_support,
        physics_checkpoint_frame=checkpoint_frame,
        capture_team_motor_targets=capture_team_motor_targets,
        research_control_frame_limit=research_control_frame_limit,
    )
    if suppression is not None:
        inspect_teacher_suppression(trace, contract=suppression, agent_ids=policy.agent_ids)
    if sonic_model_root is not None and sonic_latent_schedule is not None:
        records = motors[course.agent_id].navigation.backend.latent_records
        trace["sonic_latent_local_frames"] = np.asarray([row[0] for row in records], dtype=np.int64)
        trace["sonic_encoded_tokens"] = (
            np.concatenate([row[1] for row in records], axis=0)
            if records
            else np.empty((0, 64), dtype=np.float32)
        )
        trace["sonic_executed_tokens"] = (
            np.concatenate([row[2] for row in records], axis=0)
            if records
            else np.empty((0, 64), dtype=np.float32)
        )
        trace["sonic_latent_schedule_hash"] = np.asarray([sonic_latent_schedule.contract_hash])
    if feedback_actor_path is not None:
        feedback_motor = motors[course.agent_id]
        assert isinstance(feedback_motor, ReceivingSonicFeedbackOption)
        actor = feedback_motor.feedback_actor
        trace["feedback_actor_hash"] = np.asarray([actor.artifact_hash])
        trace["feedback_actor_foot_seen"] = np.asarray([feedback_motor.feedback_foot_seen])
        trace["feedback_actor_nonfoot_seen"] = np.asarray([feedback_motor.feedback_nonfoot_seen])
    if capture_sonic_targets or capture_ball_follow_targets or capture_live_motor_observations:
        recording_motor = motors[course.agent_id]
        assert isinstance(
            recording_motor,
            (RecordingReceivingSonicOption, RecordingReceivingSonicBallFollowOption),
        )
        motor_records = recording_motor.recorded
        expected_records = 130 if capture_live_motor_observations else 300
        if len(motor_records) != expected_records or any(
            target is None for _, target in motor_records
        ):
            raise ValueError("complete measured SONIC training trace required")
        trace["sonic_recorded_qpos"] = np.asarray(
            [observation.qpos for observation, _ in motor_records]
        )
        trace["sonic_recorded_qvel"] = np.asarray(
            [observation.qvel for observation, _ in motor_records]
        )
        trace["sonic_recorded_target"] = np.asarray(
            [target.target_rad for _, target in motor_records if target is not None]
        )
        trace["sonic_recorded_kp"] = np.asarray(
            [target.kp for _, target in motor_records if target is not None]
        )
        trace["sonic_recorded_kd"] = np.asarray(
            [target.kd for _, target in motor_records if target is not None]
        )
        if capture_live_motor_observations:
            if any(observation.navigation_command is None for observation, _ in motor_records):
                raise ValueError("complete post-clearance live navigation tape required")
            trace["sonic_recorded_navigation_command"] = np.asarray(
                [observation.navigation_command for observation, _ in motor_records],
                dtype=np.float64,
            )
            trace["sonic_recorded_target_position"] = np.asarray(
                [observation.target_position_m for observation, _ in motor_records],
                dtype=np.float64,
            )
            trace["sonic_recorded_intent"] = np.asarray(
                [observation.intent for observation, _ in motor_records]
            )
            trace["sonic_recorded_prospective_owner"] = np.asarray(
                [observation.prospective_owner for observation, _ in motor_records]
            )
            trace["sonic_recorded_committed_receiver"] = np.asarray(
                [observation.committed_receiver for observation, _ in motor_records]
            )
    if sonic_model_root is not None and sonic_command_scale_schedule is not None:
        scale_records = motors[course.agent_id].command_scale_records
        trace["sonic_command_scale_local_frames"] = np.asarray(
            [row[0] for row in scale_records], dtype=np.int64
        )
        trace["sonic_command_requested_scale"] = np.asarray([row[1] for row in scale_records])
        trace["sonic_command_scale_schedule_hash"] = np.asarray(
            [sonic_command_scale_schedule.contract_hash]
        )
    if sonic_command_replanning:
        navigation = motors[course.agent_id].navigation
        trace["sonic_command_replanning_contract_hash"] = np.asarray([navigation.contract_hash])
        trace["sonic_planner_local_frames"] = np.asarray(
            [event["frame"] for event in navigation.backend.events], dtype=np.int64
        )
        trace["sonic_planner_commands"] = np.asarray(
            [event["command"] for event in navigation.backend.events], dtype=np.float64
        ).reshape(-1, 3)
        trace["sonic_planner_reference_hashes"] = np.asarray(
            [event["reference_hash"] for event in navigation.backend.events], dtype=str
        )
    if sonic_pose_reference is not None:
        trace["sonic_pose_reference_hash"] = np.asarray([sonic_pose_reference.contract_hash])
        trace["sonic_pose_reference_source_hash"] = np.asarray(
            [sonic_pose_reference.source_evidence_hash]
        )
        trace["sonic_planner_calls"] = np.asarray(
            [motors[course.agent_id].navigation.backend.planner_calls]
        )
    return result, trace
