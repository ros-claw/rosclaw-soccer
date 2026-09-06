from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.growth.phase_conditioned_strike_assessment import (
    PhaseConditionedStrikeThresholds,
    assess_phase_conditioned_strike,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.growth.strike_phase_controller import (
    StrikePhase,
    StrikePhaseConfig,
    StrikePhaseState,
)
from rosclaw_soccer.media.phase_conditioned_strike_video import _timeline
from rosclaw_soccer.skills.team.independent_team_world import (
    _run_locomotion,
    _strike_tracking_yaw_error,
)
from rosclaw_soccer.training.phase_conditioned_strike_growth import (
    default_phase_strike_controller,
    default_phase_strike_option,
    run_phase_conditioned_strike_growth,
    validate_phase_conditioned_strike_growth,
)
from rosclaw_soccer.world.field import G1TrainingGoalSpec


def _advance(state: StrikePhaseState, time_sec: float, **overrides: object) -> StrikePhase:
    values: dict[str, object] = {
        "time_sec": time_sec,
        "stable": True,
        "stance_depth_m": 0.45,
        "stance_lateral_error_m": 0.20,
        "stance_yaw_error_rad": 0.10,
        "approach_yaw_error_rad": 0.10,
        "ball_speed_mps": 0.20,
        "ball_distance_m": 0.40,
        "option_active": False,
        "option_contact_observed": False,
        "option_completed": False,
        "config": StrikePhaseConfig(),
    }
    values.update(overrides)
    return state.advance(**values)  # type: ignore[arg-type]


def test_strike_phase_is_monotonic_and_recovery_accepts_ball_escape() -> None:
    state = StrikePhaseState()
    state.begin_capture(1.0)

    assert _advance(state, 1.23) is StrikePhase.ORIENT
    assert _advance(state, 1.24) is StrikePhase.PLANT
    assert _advance(state, 1.25) is StrikePhase.STRIKE
    assert (
        _advance(
            state,
            1.50,
            option_active=True,
            option_contact_observed=True,
        )
        is StrikePhase.RECOVER
    )
    assert _advance(state, 1.90, stable=False, ball_distance_m=9.0) is StrikePhase.RECOVER
    assert _advance(state, 2.21, stable=True, ball_distance_m=12.0) is StrikePhase.COMPLETE
    assert state.transition_count == 6
    assert state.abort_reason is None


def test_strike_phase_recovery_times_out_fail_closed() -> None:
    config = StrikePhaseConfig(recover_duration_sec=0.30, recover_timeout_sec=0.60)
    state = StrikePhaseState(phase=StrikePhase.RECOVER, phase_enter_time_sec=2.0)

    assert _advance(state, 2.61, stable=False, config=config) is StrikePhase.ABORTED
    assert state.abort_reason == "RECOVERY_TIMEOUT"


def test_strike_phase_clock_rollback_fails_closed_and_terminal_is_sticky() -> None:
    state = StrikePhaseState()
    state.begin_capture(2.0)

    assert _advance(state, 1.99) is StrikePhase.ABORTED
    assert state.abort_reason == "CLOCK_ROLLBACK"
    assert _advance(state, float("nan")) is StrikePhase.ABORTED
    assert state.abort_reason == "CLOCK_ROLLBACK"


def test_strike_phase_rejects_hardware_authority() -> None:
    with pytest.raises(ValueError, match="SIM-only envelope"):
        StrikePhaseConfig(hardware_authorized=True)


def test_strike_yaw_prediction_cannot_mutate_ball_qpos() -> None:
    qpos = np.zeros(16, dtype=np.float64)
    qpos[3] = 1.0
    qpos[7:9] = (2.0, -0.5)
    data = SimpleNamespace(qpos=qpos, qvel=np.asarray((0.7, -0.2), dtype=np.float64))
    before = qpos.copy()

    value = _strike_tracking_yaw_error(
        controller=SimpleNamespace(qpos_base=0),
        data=data,
        ball_qpos=7,
        ball_qvel=0,
        goal=G1TrainingGoalSpec(),
        config=StrikePhaseConfig(),
    )

    assert np.isfinite(value)
    np.testing.assert_array_equal(qpos, before)


class _RecordingPolicy:
    def __init__(self, state: SimpleNamespace, output: SimpleNamespace) -> None:
        self.state = state
        self.output = output
        self.seen_command: np.ndarray | None = None

    def run(self) -> None:
        self.seen_command = np.asarray(self.state.vel_cmd, dtype=np.float64).copy()
        self.output.actions = np.arange(29, dtype=np.float64)
        self.output.kps = np.arange(29, dtype=np.float64) + 1.0
        self.output.kds = np.arange(29, dtype=np.float64) + 2.0


def test_phase_mirror_reverses_yaw_without_mutating_policy_state() -> None:
    state = SimpleNamespace(
        q=np.zeros(29),
        dq=np.zeros(29),
        gravity_ori=np.asarray((0.0, 0.0, -1.0)),
        ang_vel=np.asarray((0.1, 0.2, 0.3)),
        vel_cmd=np.asarray((0.4, -0.2, 0.3)),
    )
    output = SimpleNamespace(actions=np.zeros(29), kps=np.ones(29), kds=np.ones(29))
    policy = _RecordingPolicy(state, output)
    controller = SimpleNamespace(state=state, output=output, policy=policy)
    original = state.vel_cmd.copy()

    _run_locomotion(controller, mirror=True, correct_mirrored_yaw=True)

    assert policy.seen_command is not None
    np.testing.assert_allclose(policy.seen_command, (0.4, 0.2, -0.3))
    np.testing.assert_array_equal(state.vel_cmd, original)


def _qualifying_trajectory() -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    count = 180
    time = 0.02 * (np.arange(count, dtype=np.float64) + 1.0)
    ball_pose = np.zeros((count, 7), dtype=np.float64)
    ball_pose[:, 2] = 0.115
    ball_pose[:, 3] = 1.0
    ball_pose[10, :3] = (2.0, -0.8, 0.115)
    ball_pose[50, :3] = (2.8, -0.7, 0.115)
    ball_pose[110, :3] = (4.5, -1.3, 0.16)
    ball_pose[129, :3] = (6.8, -1.2, 0.8)
    ball_pose[130, :3] = (6.9, -1.2, 0.8)
    ball_velocity = np.zeros((count, 6), dtype=np.float64)
    ball_velocity[110:130, :3] = (6.0, 0.0, 1.0)
    ball_velocity[115, :3] = (5.0, 0.0, 1.0)
    ball_velocity[131:, :3] = (1.0, 2.0, 0.0)
    contact_agent = np.zeros(count, dtype=np.int64)
    contact_effector = np.zeros(count, dtype=np.int64)
    contact_agent[[10, 50, 110, 130]] = (6, 4, 4, 2)
    contact_effector[[10, 50, 110, 130]] = (2, 2, 2, 3)
    contact_force = np.zeros(count, dtype=np.float64)
    contact_force[[10, 50, 110, 130]] = (20.0, 30.0, 180.0, 220.0)
    phase = np.zeros(count, dtype=np.int64)
    phase[50:60] = 1
    phase[60:80] = 2
    phase[80:100] = 3
    phase[100:120] = 4
    phase[120:150] = 5
    phase[150:] = 6
    phase_agent = np.zeros(count, dtype=np.int64)
    phase_agent[50:] = 4
    pass_source = np.zeros(count, dtype=np.int64)
    pass_target = np.zeros(count, dtype=np.int64)
    pass_source[10:20] = 6
    pass_target[10:20] = 4
    nonfoot = np.zeros(count, dtype=np.int64)
    nonfoot[130] = 2
    trajectory = {
        "time": time,
        "ball_pose": ball_pose,
        "ball_velocity": ball_velocity,
        "ball_contact_agent_code": contact_agent,
        "ball_contact_effector_code": contact_effector,
        "ball_contact_force_n": contact_force,
        "ball_nonfoot_contact_agent_code": nonfoot,
        "ball_nonfoot_contact_force_n": np.zeros(count, dtype=np.float64),
        "strike_phase_agent_code": phase_agent,
        "strike_phase_code": phase,
        "strike_phase_abort_code": np.zeros(count, dtype=np.int64),
        "robot_robot_contact_count": np.zeros(count, dtype=np.int64),
        "pass_source_agent_code": pass_source,
        "pass_target_agent_code": pass_target,
    }
    return trajectory, (
        "blue.finisher",
        "blue.goalkeeper",
        "blue.playmaker",
        "red.finisher",
        "red.goalkeeper",
        "red.playmaker",
    )


def test_phase_assessment_requires_complete_physics_chain() -> None:
    trajectory, agent_ids = _qualifying_trajectory()
    roles = {
        agent_id: MatchRole.GOALKEEPER
        if "goalkeeper" in agent_id
        else MatchRole.FINISHER
        if "finisher" in agent_id
        else MatchRole.PLAYMAKER
        for agent_id in agent_ids
    }

    assessment = assess_phase_conditioned_strike(
        trajectory=trajectory,
        trajectory_hash="sha256:" + "a" * 64,
        agent_ids=agent_ids,
        roles=roles,
        goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0),
        strict_replay=True,
        world_safe=True,
        thresholds=PhaseConditionedStrikeThresholds(),
    )

    assert assessment.passed
    assert assessment.phase_sequence == (1, 2, 3, 4, 5, 6)
    assert assessment.gates["on_target_before_save"]
    assert assessment.goalkeeper_agent_id == "blue.goalkeeper"


def test_phase_assessment_rejects_shooter_body_contact() -> None:
    trajectory, agent_ids = _qualifying_trajectory()
    trajectory["ball_nonfoot_contact_agent_code"][111] = 4
    roles = {
        agent_id: MatchRole.GOALKEEPER
        if "goalkeeper" in agent_id
        else MatchRole.FINISHER
        if "finisher" in agent_id
        else MatchRole.PLAYMAKER
        for agent_id in agent_ids
    }

    assessment = assess_phase_conditioned_strike(
        trajectory=trajectory,
        trajectory_hash="sha256:" + "b" * 64,
        agent_ids=agent_ids,
        roles=roles,
        goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0),
        strict_replay=True,
        world_safe=True,
    )

    assert not assessment.passed
    assert not assessment.gates["legal_nonfoot_defence_only"]


def test_phase_assessment_rejects_zero_force_glove_label() -> None:
    trajectory, agent_ids = _qualifying_trajectory()
    trajectory["ball_contact_force_n"][130] = 0.0
    roles = {
        agent_id: MatchRole.GOALKEEPER
        if "goalkeeper" in agent_id
        else MatchRole.FINISHER
        if "finisher" in agent_id
        else MatchRole.PLAYMAKER
        for agent_id in agent_ids
    }

    assessment = assess_phase_conditioned_strike(
        trajectory=trajectory,
        trajectory_hash="sha256:" + "c" * 64,
        agent_ids=agent_ids,
        roles=roles,
        goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0),
        strict_replay=True,
        world_safe=True,
    )

    assert not assessment.passed
    assert not assessment.gates["opponent_goalkeeper_glove_save"]
    assert not assessment.gates["forceful_glove_dominant_contact"]


def test_phase_assessment_rejects_fractional_codes() -> None:
    trajectory, agent_ids = _qualifying_trajectory()
    trajectory["strike_phase_code"] = trajectory["strike_phase_code"].astype(np.float64)
    trajectory["strike_phase_code"][70] = 2.5
    roles = {
        agent_id: MatchRole.GOALKEEPER
        if "goalkeeper" in agent_id
        else MatchRole.FINISHER
        if "finisher" in agent_id
        else MatchRole.PLAYMAKER
        for agent_id in agent_ids
    }

    with pytest.raises(ValueError, match="non-integer strike_phase_code"):
        assess_phase_conditioned_strike(
            trajectory=trajectory,
            trajectory_hash="sha256:" + "d" * 64,
            agent_ids=agent_ids,
            roles=roles,
            goal=G1TrainingGoalSpec(plane_x_m=7.5, width_m=3.0, height_m=2.0),
            strict_replay=True,
            world_safe=True,
        )


def test_phase_growth_defaults_bind_the_selected_physical_candidate() -> None:
    phase = default_phase_strike_controller()
    option = default_phase_strike_option()

    assert phase.strike_aim_lateral_bias_m == 0.65
    assert phase.target_stance_lateral_m == -0.30
    assert option.shoot_parameters.policy_type == "parameter"
    assert option.shoot_parameters.loft_synergy == 0.15


def test_phase_growth_refuses_to_overwrite_evidence(tmp_path: Path) -> None:
    evidence = tmp_path / "evidence"
    evidence.mkdir()
    (evidence / "operator.txt").write_text("preserve", encoding="utf-8")

    with pytest.raises(ValueError, match="must be empty"):
        run_phase_conditioned_strike_growth(
            evidence_dir=evidence,
            asset_root=evidence / "missing",
        )

    assert (evidence / "operator.txt").read_text(encoding="utf-8") == "preserve"


def test_phase_growth_validator_fails_closed_on_unbound_report(
    tmp_path: Path,
) -> None:
    report = tmp_path / "report.json"
    report.write_text('{"passed": false}\n', encoding="utf-8")

    with pytest.raises(ValueError, match="trajectory binding is absent"):
        validate_phase_conditioned_strike_growth(report)


def test_phase_video_timeline_has_four_views_and_slow_motion() -> None:
    trajectory = {"time": np.linspace(0.02, 8.60, 430)}
    assessment = {
        "events": {
            "strike_time_sec": 7.72,
            "save_time_sec": 8.08,
        }
    }

    clips = _timeline(trajectory, assessment=assessment, fps=30)

    assert {frame.camera for clip in clips for frame in clip.frames} == {
        "broadcast",
        "counter",
        "touchline",
        "wide",
    }
    assert sum(len(clip.frames) for clip in clips) > 12 * 30
