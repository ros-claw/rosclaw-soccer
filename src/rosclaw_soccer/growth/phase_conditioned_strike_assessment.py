"""Conservative assessment for a physical receive-strike-save chain.

Intent labels and peak velocity are insufficient.  This assessor requires a
teammate foot pass, a finisher foot receive, the complete monotonic strike
phase sequence, a sustained physical shot, an on-target pre-save trajectory,
and an opponent goalkeeper glove intervention in one immutable trajectory.
It is simulator-only and grants no actuator or hardware authority.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.growth.strike_phase_controller import StrikePhaseCode
from rosclaw_soccer.sim.contracts import hash_json
from rosclaw_soccer.world.field import G1TrainingGoalSpec

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")
_EXPECTED_PHASES = tuple(
    int(value)
    for value in (
        StrikePhaseCode.CAPTURE,
        StrikePhaseCode.ORIENT,
        StrikePhaseCode.PLANT,
        StrikePhaseCode.STRIKE,
        StrikePhaseCode.RECOVER,
        StrikePhaseCode.COMPLETE,
    )
)


@dataclass(frozen=True)
class PhaseConditionedStrikeThresholds:
    maximum_pass_receive_gap_sec: float = 3.0
    minimum_pass_progress_m: float = 0.50
    maximum_receive_strike_gap_sec: float = 5.0
    minimum_peak_shot_speed_mps: float = 5.0
    minimum_sustained_shot_speed_mps: float = 3.0
    sustained_speed_horizon_sec: float = 0.10
    maximum_strike_save_gap_sec: float = 0.80
    minimum_glove_contact_force_n: float = 5.0
    minimum_save_velocity_change_mps: float = 0.50
    maximum_recovery_gap_sec: float = 1.20
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.phase_conditioned_strike_thresholds.v1"

    def __post_init__(self) -> None:
        values = (
            self.maximum_pass_receive_gap_sec,
            self.minimum_pass_progress_m,
            self.maximum_receive_strike_gap_sec,
            self.minimum_peak_shot_speed_mps,
            self.minimum_sustained_shot_speed_mps,
            self.sustained_speed_horizon_sec,
            self.maximum_strike_save_gap_sec,
            self.minimum_glove_contact_force_n,
            self.minimum_save_velocity_change_mps,
            self.maximum_recovery_gap_sec,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.5 <= self.maximum_pass_receive_gap_sec <= 5.0
            or not 0.20 <= self.minimum_pass_progress_m <= 2.0
            or not 1.0 <= self.maximum_receive_strike_gap_sec <= 6.0
            or not 2.0 <= self.minimum_peak_shot_speed_mps <= 12.0
            or not 1.0 <= self.minimum_sustained_shot_speed_mps <= self.minimum_peak_shot_speed_mps
            or not 0.04 <= self.sustained_speed_horizon_sec <= 0.30
            or not 0.20 <= self.maximum_strike_save_gap_sec <= 2.0
            or not 0.10 <= self.minimum_glove_contact_force_n <= 100.0
            or not 0.20 <= self.minimum_save_velocity_change_mps <= 5.0
            or not 0.30 <= self.maximum_recovery_gap_sec <= 3.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("phase-conditioned strike thresholds violate SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class PhaseConditionedStrikeAssessment:
    trajectory_hash: str
    thresholds_hash: str
    shooter_agent_id: str
    goalkeeper_agent_id: str | None
    phase_sequence: tuple[int, ...]
    events: Mapping[str, Any]
    metrics: Mapping[str, float | int | bool | None]
    gates: Mapping[str, bool]
    strict_replay: bool
    world_safe: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.phase_conditioned_strike_assessment.v1"

    def __post_init__(self) -> None:
        if (
            not self.trajectory_hash.startswith("sha256:")
            or not self.thresholds_hash.startswith("sha256:")
            or not _IDENTIFIER.fullmatch(self.shooter_agent_id)
            or (
                self.goalkeeper_agent_id is not None
                and not _IDENTIFIER.fullmatch(self.goalkeeper_agent_id)
            )
            or any(not isinstance(value, bool) for value in self.gates.values())
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("phase-conditioned strike assessment is invalid")

    @property
    def passed(self) -> bool:
        return bool(all(self.gates.values()))

    def to_dict(self) -> dict[str, Any]:
        value = {
            "schema_version": self.schema_version,
            "trajectory_hash": self.trajectory_hash,
            "thresholds_hash": self.thresholds_hash,
            "shooter_agent_id": self.shooter_agent_id,
            "goalkeeper_agent_id": self.goalkeeper_agent_id,
            "phase_sequence": list(self.phase_sequence),
            "events": dict(self.events),
            "metrics": dict(self.metrics),
            "gates": dict(self.gates),
            "strict_replay": self.strict_replay,
            "world_safe": self.world_safe,
            "passed": self.passed,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }
        value["assessment_hash"] = hash_json(value)
        return value


def assess_phase_conditioned_strike(
    *,
    trajectory: Mapping[str, NDArray[Any]],
    trajectory_hash: str,
    agent_ids: tuple[str, ...],
    roles: Mapping[str, MatchRole],
    goal: G1TrainingGoalSpec,
    strict_replay: bool,
    world_safe: bool,
    thresholds: PhaseConditionedStrikeThresholds | None = None,
) -> PhaseConditionedStrikeAssessment:
    """Assess one physical pass, phase-conditioned shot, and glove save."""

    active = thresholds or PhaseConditionedStrikeThresholds()
    if (
        not trajectory_hash.startswith("sha256:")
        or len(agent_ids) < 4
        or len(agent_ids) != len(set(agent_ids))
        or set(agent_ids) != set(roles)
        or any(not _IDENTIFIER.fullmatch(value) for value in agent_ids)
    ):
        raise ValueError("phase-conditioned strike roster is invalid")
    time = _array(trajectory, "time", 1)
    ball_pose = _array(trajectory, "ball_pose", 2)
    ball_velocity = _array(trajectory, "ball_velocity", 2)
    contact_agent = _integer_array(trajectory, "ball_contact_agent_code")
    contact_effector = _integer_array(trajectory, "ball_contact_effector_code")
    contact_force = _array(trajectory, "ball_contact_force_n", 1)
    nonfoot_agent = _integer_array(trajectory, "ball_nonfoot_contact_agent_code")
    nonfoot_force = _array(trajectory, "ball_nonfoot_contact_force_n", 1)
    phase_agent = _integer_array(trajectory, "strike_phase_agent_code")
    phase_code = _integer_array(trajectory, "strike_phase_code")
    phase_abort = _integer_array(trajectory, "strike_phase_abort_code")
    robot_contacts = _integer_array(trajectory, "robot_robot_contact_count")
    pass_source = _integer_array(trajectory, "pass_source_agent_code")
    pass_target = _integer_array(trajectory, "pass_target_agent_code")
    count = len(time)
    one_dimensional = (
        contact_agent,
        contact_effector,
        contact_force,
        nonfoot_agent,
        nonfoot_force,
        phase_agent,
        phase_code,
        phase_abort,
        robot_contacts,
        pass_source,
        pass_target,
    )
    if (
        count < 3
        or ball_pose.shape != (count, 7)
        or ball_velocity.shape != (count, 6)
        or any(value.shape != (count,) for value in one_dimensional)
        or time[0] < 0.0
        or np.any(np.diff(time) <= 0.0)
        or any(
            np.any(value < 0) or np.any(value > len(agent_ids))
            for value in (contact_agent, phase_agent, nonfoot_agent, pass_source, pass_target)
        )
        or np.any(contact_effector < 0)
        or np.any(contact_effector > 4)
        or np.any(phase_code < 0)
        or np.any(phase_code > int(StrikePhaseCode.ABORTED))
        or np.any((contact_agent == 0) != (contact_effector == 0))
        or np.any(contact_force < 0.0)
        or np.any(nonfoot_force < 0.0)
        or np.any(phase_abort < 0)
        or np.any(phase_abort > 255)
        or np.any(robot_contacts < 0)
        or np.any((phase_agent == 0) != (phase_code == 0))
    ):
        raise ValueError("phase-conditioned strike trajectory is invalid")

    phase_owners = {int(value) for value in phase_agent if value > 0}
    shooter_code = next(iter(phase_owners)) if len(phase_owners) == 1 else 0
    shooter_id = agent_ids[shooter_code - 1] if shooter_code else agent_ids[0]
    compressed_phases = tuple(
        int(value)
        for index, value in enumerate(phase_code)
        if value > 0 and (index == 0 or value != phase_code[index - 1])
    )
    shooter_feet = np.flatnonzero(
        (contact_agent == shooter_code) & np.isin(contact_effector, (1, 2))
    )
    capture_frames = np.flatnonzero(phase_code == int(StrikePhaseCode.CAPTURE))
    receive_frame = (
        int(next((frame for frame in shooter_feet if frame in set(capture_frames)), -1))
        if shooter_code
        else -1
    )
    strike_candidates = [
        int(frame) for frame in shooter_feet if phase_code[frame] == int(StrikePhaseCode.STRIKE)
    ]
    strike_frame = strike_candidates[0] if strike_candidates else -1
    teammate_pass_frame = _find_teammate_pass_frame(
        time=time,
        ball_pose=ball_pose,
        contact_agent=contact_agent,
        contact_effector=contact_effector,
        pass_source=pass_source,
        pass_target=pass_target,
        agent_ids=agent_ids,
        shooter_code=shooter_code,
        receive_frame=receive_frame,
        active=active,
    )
    peak_speed = 0.0
    sustained_speed = 0.0
    if strike_frame >= 0:
        peak_end = int(
            np.searchsorted(time, time[strike_frame] + active.sustained_speed_horizon_sec + 0.20)
        )
        peak_speed = float(
            np.max(
                np.linalg.norm(
                    ball_velocity[strike_frame : max(strike_frame + 1, peak_end), :3], axis=1
                )
            )
        )
        sustained_index = min(
            count - 1,
            int(
                np.searchsorted(
                    time,
                    time[strike_frame] + active.sustained_speed_horizon_sec,
                    side="left",
                )
            ),
        )
        sustained_speed = float(np.linalg.norm(ball_velocity[sustained_index, :3]))

    save_frame, goalkeeper_code = _find_glove_save(
        time=time,
        ball_velocity=ball_velocity,
        contact_agent=contact_agent,
        contact_effector=contact_effector,
        contact_force=contact_force,
        roles=roles,
        agent_ids=agent_ids,
        shooter_code=shooter_code,
        strike_frame=strike_frame,
        active=active,
    )
    goalkeeper_id = agent_ids[goalkeeper_code - 1] if goalkeeper_code else None
    glove_force = 0.0 if save_frame < 0 else float(contact_force[save_frame])
    competing_body_force = 0.0 if save_frame < 0 else float(nonfoot_force[save_frame])
    save_delta = (
        0.0
        if save_frame < 0
        else float(
            np.linalg.norm(
                ball_velocity[min(count - 1, save_frame + 1), :3]
                - ball_velocity[max(0, save_frame - 1), :3]
            )
        )
    )
    projected_y, projected_z = _project_goal_crossing(
        ball_pose=ball_pose,
        ball_velocity=ball_velocity,
        frame=max(0, save_frame - 1),
        goal=goal,
    )
    completed_frames = np.flatnonzero(phase_code == int(StrikePhaseCode.COMPLETE))
    completed_frame = int(completed_frames[0]) if len(completed_frames) else -1
    legal_nonfoot = bool(
        goalkeeper_code
        and np.all(
            (nonfoot_agent == 0)
            | ((nonfoot_agent == goalkeeper_code) & (np.arange(count) >= max(0, save_frame - 1)))
        )
    )
    whole_ball_y = goal.width_m / 2.0 - goal.ball_radius_m
    on_target = bool(
        math.isfinite(projected_y)
        and math.isfinite(projected_z)
        and abs(projected_y) <= whole_ball_y
        and goal.ball_radius_m <= projected_z <= goal.height_m - goal.ball_radius_m
    )
    receive_strike_gap = (
        math.inf
        if receive_frame < 0 or strike_frame < 0
        else float(time[strike_frame] - time[receive_frame])
    )
    recovery_gap = (
        math.inf
        if strike_frame < 0 or completed_frame < 0
        else float(time[completed_frame] - time[strike_frame])
    )
    gates = {
        "strict_replay": strict_replay,
        "world_safe": bool(world_safe and not np.any(robot_contacts > 0)),
        "single_finisher_phase_owner": bool(
            len(phase_owners) == 1 and roles.get(shooter_id) is MatchRole.FINISHER
        ),
        "physical_teammate_pass_received": teammate_pass_frame >= 0 and receive_frame >= 0,
        "monotonic_complete_phase_chain": compressed_phases == _EXPECTED_PHASES,
        "phase_never_aborted": bool(not np.any(phase_abort > 0)),
        "physical_foot_strike_in_strike_phase": strike_frame >= 0,
        "bounded_receive_to_strike_time": receive_strike_gap
        <= active.maximum_receive_strike_gap_sec,
        "peak_shot_speed": peak_speed >= active.minimum_peak_shot_speed_mps,
        "sustained_post_contact_speed": sustained_speed >= active.minimum_sustained_shot_speed_mps,
        "opponent_goalkeeper_glove_save": save_frame >= 0,
        "forceful_glove_dominant_contact": bool(
            glove_force >= active.minimum_glove_contact_force_n
            and glove_force > competing_body_force
        ),
        "on_target_before_save": on_target,
        "save_changed_ball_velocity": save_delta >= active.minimum_save_velocity_change_mps,
        "legal_nonfoot_defence_only": legal_nonfoot,
        "stable_recovery_completed": recovery_gap <= active.maximum_recovery_gap_sec,
    }
    events = {
        "pass_frame": teammate_pass_frame,
        "receive_frame": receive_frame,
        "strike_frame": strike_frame,
        "save_frame": save_frame,
        "complete_frame": completed_frame,
        "pass_time_sec": None if teammate_pass_frame < 0 else float(time[teammate_pass_frame]),
        "receive_time_sec": None if receive_frame < 0 else float(time[receive_frame]),
        "strike_time_sec": None if strike_frame < 0 else float(time[strike_frame]),
        "save_time_sec": None if save_frame < 0 else float(time[save_frame]),
        "complete_time_sec": None if completed_frame < 0 else float(time[completed_frame]),
    }
    metrics: dict[str, float | int | bool | None] = {
        "peak_shot_speed_mps": peak_speed,
        "sustained_shot_speed_mps": sustained_speed,
        "save_velocity_change_mps": save_delta,
        "glove_contact_force_n": glove_force,
        "same_frame_nonfoot_contact_force_n": competing_body_force,
        "projected_goal_y_m": projected_y if math.isfinite(projected_y) else None,
        "projected_goal_z_m": projected_z if math.isfinite(projected_z) else None,
        "receive_to_strike_sec": (
            receive_strike_gap if math.isfinite(receive_strike_gap) else None
        ),
        "strike_to_recovery_complete_sec": (recovery_gap if math.isfinite(recovery_gap) else None),
        "robot_robot_contact_count": int(np.sum(robot_contacts)),
        "legal_nonfoot_defence_only": legal_nonfoot,
    }
    return PhaseConditionedStrikeAssessment(
        trajectory_hash=trajectory_hash,
        thresholds_hash=active.config_hash,
        shooter_agent_id=shooter_id,
        goalkeeper_agent_id=goalkeeper_id,
        phase_sequence=compressed_phases,
        events=events,
        metrics=metrics,
        gates=gates,
        strict_replay=strict_replay,
        world_safe=bool(world_safe),
    )


def _find_teammate_pass_frame(
    *,
    time: NDArray[np.float64],
    ball_pose: NDArray[np.float64],
    contact_agent: NDArray[np.int64],
    contact_effector: NDArray[np.int64],
    pass_source: NDArray[np.int64],
    pass_target: NDArray[np.int64],
    agent_ids: tuple[str, ...],
    shooter_code: int,
    receive_frame: int,
    active: PhaseConditionedStrikeThresholds,
) -> int:
    if shooter_code <= 0 or receive_frame < 0:
        return -1
    shooter_team = agent_ids[shooter_code - 1].split(".", 1)[0]
    for frame in reversed(range(receive_frame)):
        source_code = int(contact_agent[frame])
        if (
            source_code <= 0
            or source_code == shooter_code
            or contact_effector[frame] not in {1, 2}
            or agent_ids[source_code - 1].split(".", 1)[0] != shooter_team
            or time[receive_frame] - time[frame] > active.maximum_pass_receive_gap_sec
            or float(np.linalg.norm(ball_pose[receive_frame, :2] - ball_pose[frame, :2]))
            < active.minimum_pass_progress_m
        ):
            continue
        commitment_end = min(
            len(time), int(np.searchsorted(time, time[frame] + 0.30, side="right"))
        )
        if np.any(
            (pass_source[frame:commitment_end] == source_code)
            & (pass_target[frame:commitment_end] == shooter_code)
        ):
            return frame
    return -1


def _find_glove_save(
    *,
    time: NDArray[np.float64],
    ball_velocity: NDArray[np.float64],
    contact_agent: NDArray[np.int64],
    contact_effector: NDArray[np.int64],
    contact_force: NDArray[np.float64],
    roles: Mapping[str, MatchRole],
    agent_ids: tuple[str, ...],
    shooter_code: int,
    strike_frame: int,
    active: PhaseConditionedStrikeThresholds,
) -> tuple[int, int]:
    if shooter_code <= 0 or strike_frame < 0:
        return (-1, 0)
    shooter_team = agent_ids[shooter_code - 1].split(".", 1)[0]
    for raw_frame in np.flatnonzero(
        (np.arange(len(time)) > strike_frame) & np.isin(contact_effector, (3, 4))
    ):
        frame = int(raw_frame)
        code = int(contact_agent[frame])
        if (
            code > 0
            and roles[agent_ids[code - 1]] is MatchRole.GOALKEEPER
            and agent_ids[code - 1].split(".", 1)[0] != shooter_team
            and time[frame] - time[strike_frame] <= active.maximum_strike_save_gap_sec
            and contact_force[frame] >= active.minimum_glove_contact_force_n
            and float(
                np.linalg.norm(
                    ball_velocity[min(len(time) - 1, frame + 1), :3]
                    - ball_velocity[max(0, frame - 1), :3]
                )
            )
            >= active.minimum_save_velocity_change_mps
        ):
            return (int(frame), code)
    return (-1, 0)


def _project_goal_crossing(
    *,
    ball_pose: NDArray[np.float64],
    ball_velocity: NDArray[np.float64],
    frame: int,
    goal: G1TrainingGoalSpec,
) -> tuple[float, float]:
    position = ball_pose[frame, :3]
    velocity = ball_velocity[frame, :3]
    if velocity[0] <= 1.0e-6 or position[0] >= goal.plane_x_m:
        return (math.inf, math.inf)
    horizon = float((goal.plane_x_m - position[0]) / velocity[0])
    if not 0.0 < horizon <= 2.0:
        return (math.inf, math.inf)
    return (
        float(position[1] + horizon * velocity[1]),
        float(position[2] + horizon * velocity[2] - 0.5 * 9.81 * horizon * horizon),
    )


def _array(
    trajectory: Mapping[str, NDArray[Any]], name: str, dimensions: int
) -> NDArray[np.float64]:
    if name not in trajectory:
        raise ValueError(f"phase-conditioned strike trajectory lacks {name}")
    value = np.asarray(trajectory[name], dtype=np.float64)
    if value.ndim != dimensions or not np.all(np.isfinite(value)):
        raise ValueError(f"phase-conditioned strike trajectory has invalid {name}")
    return value


def _integer_array(trajectory: Mapping[str, NDArray[Any]], name: str) -> NDArray[np.int64]:
    value = _array(trajectory, name, 1)
    rounded = np.rint(value)
    if not np.array_equal(value, rounded):
        raise ValueError(f"phase-conditioned strike trajectory has non-integer {name}")
    return rounded.astype(np.int64)


__all__ = [
    "PhaseConditionedStrikeAssessment",
    "PhaseConditionedStrikeThresholds",
    "assess_phase_conditioned_strike",
]
