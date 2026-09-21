"""Reusable, simulation-only execution of the frozen R0 receiving classroom.

Experiment drivers supply external paths and persist evidence. This adapter
does not import historical experiment scripts, patch agent methods, update
weights, write artifacts, or create a runtime/hardware execution path.
"""

from dataclasses import replace
from pathlib import Path

import numpy as np

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.providers.g1.receiving_sonic import ReceivingSonicOption
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
) -> tuple[IndependentTeamWorldResult, dict[str, np.ndarray]]:
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
    ):
        raise ValueError("SONIC parameters without a frozen model are invalid")
    if sonic_model_root is not None and (
        type(sonic_start_frame) is not int or not 0 <= sonic_start_frame < 300
    ):
        raise ValueError("SONIC must enter inside the reference course")
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
    motors = {}
    if sonic_model_root is not None:
        world = replace(world, motor_idle_residual_fallback=True)
        # Instantiate a fresh backend for every simulation, never reuse its history.
        motors[course.agent_id] = ReceivingSonicOption(
            sonic_model_root,
            course.agent_id,
            start_frame=sonic_start_frame,
            velocity_scale=sonic_velocity_scale,
            planner_seed=sonic_planner_seed,
            latent_schedule=sonic_latent_schedule,
            command_scale_schedule=sonic_command_scale_schedule,
            pose_reference=sonic_pose_reference,
            experimental_command_replanning=sonic_command_replanning,
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
        receiving_oracle=oracle,
        receiving_phase_reference=phase_reference,
        receiving_feedback=feedback_provider,
        capture_oracle_authority=capture_oracle_authority,
        capture_locomotion_memory=capture_locomotion_memory,
        contact_teacher_suppression=suppression,
        capture_initial_physics=True,
        capture_initial_support=capture_support,
        physics_checkpoint_frame=checkpoint_frame,
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
