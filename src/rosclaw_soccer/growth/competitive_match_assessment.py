"""Physics-first assessment for continuous competitive soccer rollouts.

The assessor converts one immutable multi-G1 trajectory into conservative
skill evidence.  Intent labels are insufficient: a pass, interception, shot,
or save is credited only when the corresponding foot contact and ball-state
transition occur in the same monotonic MuJoCo trace.  The module is simulator
agnostic and grants no robot, ROS, actuator, or hardware authority.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from enum import StrEnum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.sim.contracts import hash_json

_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_.:-]{0,127}$")


class CompetitiveSkill(StrEnum):
    BALL_WIN = "ball_win"
    DRIBBLE = "dribble"
    PASS = "pass"
    SPRINT = "sprint"
    SHOT = "shot"
    SAVE = "save"


@dataclass(frozen=True)
class CompetitiveMatchThresholds:
    contact_onset_gap_sec: float = 0.12
    maximum_commitment_lag_sec: float = 0.30
    maximum_link_gap_sec: float = 6.0
    minimum_dribble_progress_m: float = 0.30
    minimum_pass_progress_m: float = 0.50
    minimum_sprint_speed_mps: float = 0.42
    minimum_sprint_duration_sec: float = 0.35
    minimum_shot_speed_mps: float = 3.0
    minimum_shot_goalward_speed_mps: float = 1.5
    minimum_save_velocity_change_mps: float = 0.50
    activation_ceiling: str = "SIM_ONLY"
    hardware_authorized: bool = False
    schema_version: str = "rosclaw_soccer.competitive_match_thresholds.v1"

    def __post_init__(self) -> None:
        values = (
            self.contact_onset_gap_sec,
            self.maximum_commitment_lag_sec,
            self.maximum_link_gap_sec,
            self.minimum_dribble_progress_m,
            self.minimum_pass_progress_m,
            self.minimum_sprint_speed_mps,
            self.minimum_sprint_duration_sec,
            self.minimum_shot_speed_mps,
            self.minimum_shot_goalward_speed_mps,
            self.minimum_save_velocity_change_mps,
        )
        if (
            any(not math.isfinite(value) for value in values)
            or not 0.04 <= self.contact_onset_gap_sec <= 0.30
            or not 0.05 <= self.maximum_commitment_lag_sec <= 0.50
            or not 0.5 <= self.maximum_link_gap_sec <= 6.0
            or not 0.10 <= self.minimum_dribble_progress_m <= 1.50
            or not 0.20 <= self.minimum_pass_progress_m <= 3.0
            or not 0.20 <= self.minimum_sprint_speed_mps <= 1.50
            or not 0.10 <= self.minimum_sprint_duration_sec <= 2.0
            or not 1.0 <= self.minimum_shot_speed_mps <= 12.0
            or not 0.5 <= self.minimum_shot_goalward_speed_mps <= self.minimum_shot_speed_mps
            or not 0.20 <= self.minimum_save_velocity_change_mps <= 5.0
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_authorized
        ):
            raise ValueError("competitive match thresholds violate the SIM-only envelope")

    @property
    def config_hash(self) -> str:
        return str(hash_json(asdict(self)))


@dataclass(frozen=True)
class CompetitiveSkillEvent:
    skill: CompetitiveSkill
    time_sec: float
    agent_id: str
    team_id: str
    frame_index: int
    ball_position_m: tuple[float, float, float]
    ball_speed_mps: float
    source_contact_force_n: float
    target_agent_id: str | None = None
    physics_derived: bool = True
    pixels_used_for_scoring: bool = False
    schema_version: str = "rosclaw_soccer.competitive_skill_event.v1"

    def __post_init__(self) -> None:
        if (
            not isinstance(self.skill, CompetitiveSkill)
            or not _IDENTIFIER.fullmatch(self.agent_id)
            or not _IDENTIFIER.fullmatch(self.team_id)
            or (
                self.target_agent_id is not None and not _IDENTIFIER.fullmatch(self.target_agent_id)
            )
            or not math.isfinite(self.time_sec)
            or self.time_sec < 0.0
            or isinstance(self.frame_index, bool)
            or self.frame_index < 0
            or len(self.ball_position_m) != 3
            or any(not math.isfinite(value) for value in self.ball_position_m)
            or not math.isfinite(self.ball_speed_mps)
            or self.ball_speed_mps < 0.0
            or not math.isfinite(self.source_contact_force_n)
            or self.source_contact_force_n < 0.0
            or not self.physics_derived
            or self.pixels_used_for_scoring
        ):
            raise ValueError("competitive skill event is invalid")

    @property
    def event_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        value = asdict(self)
        value["skill"] = self.skill.value
        return value


@dataclass(frozen=True)
class CompetitiveMatchAssessment:
    trajectory_hash: str
    thresholds_hash: str
    events: tuple[CompetitiveSkillEvent, ...]
    gates: Mapping[str, bool]
    first_failed_skill: CompetitiveSkill | None
    strict_replay: bool
    safe: bool
    activation_ceiling: str = "SIM_ONLY"
    hardware_command_sent: bool = False
    schema_version: str = "rosclaw_soccer.competitive_match_assessment.v1"

    def __post_init__(self) -> None:
        gates = dict(self.gates)
        if (
            not self.trajectory_hash.startswith("sha256:")
            or not self.thresholds_hash.startswith("sha256:")
            or not gates
            or any(not isinstance(value, bool) for value in gates.values())
            or not isinstance(self.strict_replay, bool)
            or not isinstance(self.safe, bool)
            or self.activation_ceiling != "SIM_ONLY"
            or self.hardware_command_sent
        ):
            raise ValueError("competitive match assessment is invalid")
        object.__setattr__(self, "events", tuple(self.events))
        object.__setattr__(self, "gates", gates)

    @property
    def passed(self) -> bool:
        return bool(all(self.gates.values()) and self.first_failed_skill is None)

    @property
    def assessment_hash(self) -> str:
        return str(hash_json(self.to_dict()))

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_hash": self.trajectory_hash,
            "thresholds_hash": self.thresholds_hash,
            "events": [event.to_dict() for event in self.events],
            "event_hashes": [event.event_hash for event in self.events],
            "gates": dict(self.gates),
            "first_failed_skill": (
                None if self.first_failed_skill is None else self.first_failed_skill.value
            ),
            "strict_replay": self.strict_replay,
            "safe": self.safe,
            "passed": self.passed,
            "activation_ceiling": self.activation_ceiling,
            "hardware_command_sent": self.hardware_command_sent,
        }


@dataclass(frozen=True)
class _AthleteContact:
    frame: int
    time_sec: float
    agent_id: str
    team_id: str
    ball_position: NDArray[np.float64]
    ball_velocity: NDArray[np.float64]
    force_n: float
    effector_code: int

    @property
    def is_foot(self) -> bool:
        return self.effector_code in {1, 2}

    @property
    def is_glove(self) -> bool:
        return self.effector_code in {3, 4}


def assess_competitive_match_trajectory(
    *,
    trajectory: Mapping[str, NDArray[Any]],
    trajectory_hash: str,
    agent_ids: tuple[str, ...],
    roles: Mapping[str, MatchRole],
    strict_replay: bool,
    world_safe: bool,
    thresholds: CompetitiveMatchThresholds | None = None,
) -> CompetitiveMatchAssessment:
    """Extract conservative match skills from one physical trajectory."""

    active = thresholds or CompetitiveMatchThresholds()
    if (
        not trajectory_hash.startswith("sha256:")
        or len(agent_ids) < 4
        or len(agent_ids) != len(set(agent_ids))
        or set(agent_ids) != set(roles)
        or any(not _IDENTIFIER.fullmatch(value) for value in agent_ids)
        or any(not isinstance(value, MatchRole) for value in roles.values())
    ):
        raise ValueError("competitive match roster or trajectory identity is invalid")
    time = _array(trajectory, "time", 1)
    ball_pose = _array(trajectory, "ball_pose", 2)
    ball_velocity = _array(trajectory, "ball_velocity", 2)
    contact_code = _array(trajectory, "ball_contact_agent_code", 1)
    effector_code = _array(trajectory, "ball_contact_effector_code", 1)
    contact_force = _array(trajectory, "ball_contact_force_n", 1)
    nonfoot_code = _array(trajectory, "ball_nonfoot_contact_agent_code", 1)
    robot_contacts = _array(trajectory, "robot_robot_contact_count", 1)
    pass_source_code = _array(trajectory, "pass_source_agent_code", 1)
    pass_target_code = _array(trajectory, "pass_target_agent_code", 1)
    strike_lease_code = _array(trajectory, "strike_lease_agent_code", 1)
    count = len(time)
    if (
        count < 3
        or ball_pose.shape != (count, 7)
        or ball_velocity.shape != (count, 6)
        or any(
            value.shape != (count,)
            for value in (
                contact_code,
                effector_code,
                contact_force,
                nonfoot_code,
                robot_contacts,
                pass_source_code,
                pass_target_code,
                strike_lease_code,
            )
        )
        or np.any(np.diff(time) <= 0.0)
        or np.any(contact_code < 0)
        or np.any(contact_code > len(agent_ids))
        or np.any(effector_code < 0)
        or np.any(effector_code > 4)
        or np.any((contact_code == 0) != (effector_code == 0))
        or np.any(pass_source_code < 0)
        or np.any(pass_source_code > len(agent_ids))
        or np.any(pass_target_code < 0)
        or np.any(pass_target_code > len(agent_ids))
        or np.any(strike_lease_code < 0)
        or np.any(strike_lease_code > len(agent_ids))
        or np.any((pass_source_code == 0) != (pass_target_code == 0))
    ):
        raise ValueError("competitive match trajectory shape or clock is invalid")
    contacts = _contact_onsets(
        time=time,
        ball_pose=ball_pose,
        ball_velocity=ball_velocity,
        contact_code=contact_code,
        effector_code=effector_code,
        contact_force=contact_force,
        agent_ids=agent_ids,
        onset_gap_sec=active.contact_onset_gap_sec,
    )
    events: list[CompetitiveSkillEvent] = []

    dribble = _find_dribble(contacts, active)
    if dribble is not None:
        events.append(_event(CompetitiveSkill.DRIBBLE, dribble[1], target=dribble[0].agent_id))
    pass_link = _find_pass(
        contacts,
        time=time,
        source_code=pass_source_code,
        target_code=pass_target_code,
        agent_ids=agent_ids,
        active=active,
    )
    if pass_link is not None:
        events.append(_event(CompetitiveSkill.PASS, pass_link[0], target=pass_link[1].agent_id))
    ball_win = _find_ball_win(contacts, active)
    if ball_win is not None:
        events.append(_event(CompetitiveSkill.BALL_WIN, ball_win[1], target=ball_win[0].agent_id))
    sprint = _find_sprint(trajectory, time, agent_ids, active)
    if sprint is not None:
        agent_id, frame, speed = sprint
        events.append(
            CompetitiveSkillEvent(
                skill=CompetitiveSkill.SPRINT,
                time_sec=float(time[frame]),
                agent_id=agent_id,
                team_id=_team(agent_id),
                frame_index=frame,
                ball_position_m=(
                    float(ball_pose[frame, 0]),
                    float(ball_pose[frame, 1]),
                    float(ball_pose[frame, 2]),
                ),
                ball_speed_mps=speed,
                source_contact_force_n=0.0,
            )
        )
    shot = _find_shot(
        contacts,
        time=time,
        ball_velocity=ball_velocity,
        strike_lease_code=strike_lease_code,
        agent_ids=agent_ids,
        roles=roles,
        active=active,
    )
    if shot is not None:
        events.append(_event(CompetitiveSkill.SHOT, shot))
    save = _find_save(contacts, ball_velocity, roles, shot, active)
    if save is not None:
        events.append(
            _event(CompetitiveSkill.SAVE, save, target=None if shot is None else shot.agent_id)
        )
    events.sort(key=lambda value: (value.time_sec, value.skill.value))

    skills = {event.skill for event in events}
    contact_agents = {contact.agent_id for contact in contacts}
    contact_teams = {contact.team_id for contact in contacts}
    telemetry_safe = bool(not np.any(robot_contacts > 0) and world_safe)
    foot_only_ball_control = bool(not np.any(nonfoot_code > 0))
    gates = {
        "strict_replay": strict_replay,
        "world_safe": telemetry_safe,
        "foot_only_ball_control": foot_only_ball_control,
        "three_physical_participants": len(contact_agents) >= 3,
        "both_teams_touch_ball": len(contact_teams) >= 2,
        "ball_win_observed": CompetitiveSkill.BALL_WIN in skills,
        "dribble_observed": CompetitiveSkill.DRIBBLE in skills,
        "pass_observed": CompetitiveSkill.PASS in skills,
        "sprint_observed": CompetitiveSkill.SPRINT in skills,
        "shot_observed": CompetitiveSkill.SHOT in skills,
        "save_observed": CompetitiveSkill.SAVE in skills,
    }
    ordered = (
        CompetitiveSkill.BALL_WIN,
        CompetitiveSkill.DRIBBLE,
        CompetitiveSkill.PASS,
        CompetitiveSkill.SPRINT,
        CompetitiveSkill.SHOT,
        CompetitiveSkill.SAVE,
    )
    first_failed = next((skill for skill in ordered if skill not in skills), None)
    return CompetitiveMatchAssessment(
        trajectory_hash=trajectory_hash,
        thresholds_hash=active.config_hash,
        events=tuple(events),
        gates=gates,
        first_failed_skill=first_failed,
        strict_replay=strict_replay,
        safe=telemetry_safe,
    )


def _array(
    trajectory: Mapping[str, NDArray[Any]], name: str, dimensions: int
) -> NDArray[np.float64]:
    if name not in trajectory:
        raise ValueError(f"competitive match trajectory lacks {name}")
    value = np.asarray(trajectory[name], dtype=np.float64)
    if value.ndim != dimensions or not np.all(np.isfinite(value)):
        raise ValueError(f"competitive match trajectory has invalid {name}")
    return value


def _contact_onsets(
    *,
    time: NDArray[np.float64],
    ball_pose: NDArray[np.float64],
    ball_velocity: NDArray[np.float64],
    contact_code: NDArray[np.float64],
    effector_code: NDArray[np.float64],
    contact_force: NDArray[np.float64],
    agent_ids: tuple[str, ...],
    onset_gap_sec: float,
) -> tuple[_AthleteContact, ...]:
    contacts: list[_AthleteContact] = []
    last_frame_by_agent: dict[str, int] = {}
    for frame in np.flatnonzero(contact_code > 0):
        code = int(contact_code[frame])
        agent_id = agent_ids[code - 1]
        previous = last_frame_by_agent.get(agent_id)
        last_frame_by_agent[agent_id] = int(frame)
        if previous is not None and float(time[frame] - time[previous]) <= onset_gap_sec:
            continue
        contacts.append(
            _AthleteContact(
                frame=int(frame),
                time_sec=float(time[frame]),
                agent_id=agent_id,
                team_id=_team(agent_id),
                ball_position=np.asarray(ball_pose[frame, :3], dtype=np.float64),
                ball_velocity=np.asarray(ball_velocity[frame, :3], dtype=np.float64),
                force_n=float(contact_force[frame]),
                effector_code=int(effector_code[frame]),
            )
        )
    return tuple(contacts)


def _find_dribble(
    contacts: tuple[_AthleteContact, ...], active: CompetitiveMatchThresholds
) -> tuple[_AthleteContact, _AthleteContact] | None:
    for first, second in zip(contacts, contacts[1:], strict=False):
        if (
            first.agent_id == second.agent_id
            and first.is_foot
            and second.is_foot
            and second.time_sec - first.time_sec <= active.maximum_link_gap_sec
            and float(np.linalg.norm(second.ball_position[:2] - first.ball_position[:2]))
            >= active.minimum_dribble_progress_m
        ):
            return first, second
    return None


def _find_pass(
    contacts: tuple[_AthleteContact, ...],
    *,
    time: NDArray[np.float64],
    source_code: NDArray[np.float64],
    target_code: NDArray[np.float64],
    agent_ids: tuple[str, ...],
    active: CompetitiveMatchThresholds,
) -> tuple[_AthleteContact, _AthleteContact] | None:
    for first, second in zip(contacts, contacts[1:], strict=False):
        if (
            first.team_id == second.team_id
            and first.is_foot
            and second.is_foot
            and first.agent_id != second.agent_id
            and second.time_sec - first.time_sec <= active.maximum_link_gap_sec
            and float(np.linalg.norm(second.ball_position[:2] - first.ball_position[:2]))
            >= active.minimum_pass_progress_m
            and _commitment_observed(
                time=time,
                source_code=source_code,
                target_code=target_code,
                source_frame=first.frame,
                source_agent_code=agent_ids.index(first.agent_id) + 1,
                target_agent_code=agent_ids.index(second.agent_id) + 1,
                maximum_lag_sec=active.maximum_commitment_lag_sec,
            )
        ):
            return first, second
    return None


def _find_ball_win(
    contacts: tuple[_AthleteContact, ...], active: CompetitiveMatchThresholds
) -> tuple[_AthleteContact, _AthleteContact] | None:
    for first, second in zip(contacts, contacts[1:], strict=False):
        if (
            first.team_id != second.team_id
            and first.is_foot
            and second.is_foot
            and second.time_sec - first.time_sec <= active.maximum_link_gap_sec
        ):
            return first, second
    return None


def _find_sprint(
    trajectory: Mapping[str, NDArray[Any]],
    time: NDArray[np.float64],
    agent_ids: tuple[str, ...],
    active: CompetitiveMatchThresholds,
) -> tuple[str, int, float] | None:
    best: tuple[str, int, float] | None = None
    for agent_id in agent_ids:
        pose = _array(trajectory, f"{_agent_key(agent_id)}_pelvis_pose", 2)
        if pose.shape != (len(time), 7):
            raise ValueError("competitive match pelvis trajectory shape is invalid")
        dt = np.diff(time)
        speed = np.linalg.norm(np.diff(pose[:, :2], axis=0), axis=1) / dt
        required = max(1, int(math.ceil(active.minimum_sprint_duration_sec / float(np.median(dt)))))
        mask = speed >= active.minimum_sprint_speed_mps
        for start in range(0, max(0, len(mask) - required + 1)):
            if bool(np.all(mask[start : start + required])):
                peak_offset = int(np.argmax(speed[start : start + required]))
                frame = start + peak_offset + 1
                candidate = (agent_id, frame, float(speed[frame - 1]))
                if best is None or candidate[2] > best[2]:
                    best = candidate
    return best


def _find_shot(
    contacts: tuple[_AthleteContact, ...],
    *,
    time: NDArray[np.float64],
    ball_velocity: NDArray[np.float64],
    strike_lease_code: NDArray[np.float64],
    agent_ids: tuple[str, ...],
    roles: Mapping[str, MatchRole],
    active: CompetitiveMatchThresholds,
) -> _AthleteContact | None:
    for contact in contacts:
        if roles[contact.agent_id] is not MatchRole.FINISHER or not contact.is_foot:
            continue
        agent_code = agent_ids.index(contact.agent_id) + 1
        commitment_end = int(
            np.searchsorted(
                time,
                contact.time_sec + active.maximum_commitment_lag_sec,
                side="right",
            )
        )
        if not np.any(strike_lease_code[contact.frame : commitment_end] == agent_code):
            continue
        end = min(len(ball_velocity), contact.frame + 21)
        window = ball_velocity[contact.frame : end, :3]
        attacking_sign = 1.0 if contact.team_id == "red" else -1.0
        if (
            float(np.max(np.linalg.norm(window, axis=1))) >= active.minimum_shot_speed_mps
            and float(np.max(attacking_sign * window[:, 0]))
            >= active.minimum_shot_goalward_speed_mps
        ):
            return contact
    return None


def _commitment_observed(
    *,
    time: NDArray[np.float64],
    source_code: NDArray[np.float64],
    target_code: NDArray[np.float64],
    source_frame: int,
    source_agent_code: int,
    target_agent_code: int,
    maximum_lag_sec: float,
) -> bool:
    """Bind a physical source touch to the declared pass recipient.

    The tactical controller only observes the contact on its next 10 Hz tick,
    so the commitment may trail the contact by a bounded fraction of a second.
    """

    end = int(
        np.searchsorted(
            time,
            float(time[source_frame]) + maximum_lag_sec,
            side="right",
        )
    )
    return bool(
        np.any(
            (source_code[source_frame:end] == source_agent_code)
            & (target_code[source_frame:end] == target_agent_code)
        )
    )


def _find_save(
    contacts: tuple[_AthleteContact, ...],
    ball_velocity: NDArray[np.float64],
    roles: Mapping[str, MatchRole],
    shot: _AthleteContact | None,
    active: CompetitiveMatchThresholds,
) -> _AthleteContact | None:
    if shot is None:
        return None
    for contact in contacts:
        if (
            contact.time_sec <= shot.time_sec
            or contact.time_sec - shot.time_sec > active.maximum_link_gap_sec
            or roles[contact.agent_id] is not MatchRole.GOALKEEPER
            or contact.team_id == shot.team_id
            or not contact.is_glove
        ):
            continue
        before = ball_velocity[max(0, contact.frame - 1), :3]
        after = ball_velocity[min(len(ball_velocity) - 1, contact.frame + 1), :3]
        if float(np.linalg.norm(after - before)) >= active.minimum_save_velocity_change_mps:
            return contact
    return None


def _event(
    skill: CompetitiveSkill,
    contact: _AthleteContact,
    *,
    target: str | None = None,
) -> CompetitiveSkillEvent:
    return CompetitiveSkillEvent(
        skill=skill,
        time_sec=contact.time_sec,
        agent_id=contact.agent_id,
        team_id=contact.team_id,
        frame_index=contact.frame,
        ball_position_m=(
            float(contact.ball_position[0]),
            float(contact.ball_position[1]),
            float(contact.ball_position[2]),
        ),
        ball_speed_mps=float(np.linalg.norm(contact.ball_velocity)),
        source_contact_force_n=contact.force_n,
        target_agent_id=target,
    )


def _team(agent_id: str) -> str:
    team = agent_id.split(".", 1)[0]
    if not _IDENTIFIER.fullmatch(team):
        raise ValueError("competitive match agent lacks a normalized team prefix")
    return team


def _agent_key(agent_id: str) -> str:
    return agent_id.replace(".", "_").replace(":", "_").replace("-", "_")


__all__ = [
    "CompetitiveMatchAssessment",
    "CompetitiveMatchThresholds",
    "CompetitiveSkill",
    "CompetitiveSkillEvent",
    "assess_competitive_match_trajectory",
]
