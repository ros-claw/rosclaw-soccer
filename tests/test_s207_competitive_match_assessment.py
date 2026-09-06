from __future__ import annotations

import numpy as np
import pytest

from rosclaw_soccer.growth.competitive_match_assessment import (
    CompetitiveSkill,
    assess_competitive_match_trajectory,
)
from rosclaw_soccer.growth.role_self_model import MatchRole
from rosclaw_soccer.sim.contracts import hash_json

_AGENTS = (
    "blue.finisher",
    "blue.goalkeeper",
    "red.finisher",
    "red.playmaker",
)
_ROLES = {
    "blue.finisher": MatchRole.FINISHER,
    "blue.goalkeeper": MatchRole.GOALKEEPER,
    "red.finisher": MatchRole.FINISHER,
    "red.playmaker": MatchRole.PLAYMAKER,
}


def _trajectory() -> dict[str, np.ndarray]:
    count = 120
    time = 0.02 * (np.arange(count) + 1)
    ball_pose = np.zeros((count, 7), dtype=np.float64)
    ball_pose[:, 3] = 1.0
    ball_pose[:, 0] = np.linspace(0.0, 4.0, count)
    ball_velocity = np.zeros((count, 6), dtype=np.float64)
    contact = np.zeros(count, dtype=np.int64)
    effector = np.zeros(count, dtype=np.int64)
    force = np.zeros(count, dtype=np.float64)
    pass_source = np.zeros(count, dtype=np.int64)
    pass_target = np.zeros(count, dtype=np.int64)
    strike_lease = np.zeros(count, dtype=np.int64)
    # Blue initially touches the ball, red wins it, the playmaker dribbles and
    # passes, the finisher shoots, then the opposing goalkeeper uses a glove.
    for frame, code in ((5, 1), (15, 4), (30, 4), (50, 3), (60, 3), (75, 2)):
        contact[frame] = code
        effector[frame] = 3 if code == 2 else 2
        force[frame] = 40.0
    ball_velocity[60:75, 0] = 4.2
    ball_velocity[75:, 0] = -1.0
    pass_source[30:36] = 4
    pass_target[30:36] = 3
    strike_lease[60:71] = 3
    value: dict[str, np.ndarray] = {
        "time": time,
        "ball_pose": ball_pose,
        "ball_velocity": ball_velocity,
        "ball_contact_agent_code": contact,
        "ball_contact_effector_code": effector,
        "ball_contact_force_n": force,
        "ball_nonfoot_contact_agent_code": np.zeros(count, dtype=np.int64),
        "robot_robot_contact_count": np.zeros(count, dtype=np.int64),
        "pass_source_agent_code": pass_source,
        "pass_target_agent_code": pass_target,
        "strike_lease_agent_code": strike_lease,
    }
    for index, agent_id in enumerate(_AGENTS):
        pose = np.zeros((count, 7), dtype=np.float64)
        pose[:, 2] = 0.78
        pose[:, 3] = 1.0
        if agent_id == "red.finisher":
            pose[:, 0] = np.arange(count) * 0.012
        else:
            pose[:, 0] = index
        value[agent_id.replace(".", "_") + "_pelvis_pose"] = pose
    return value


def test_competitive_match_requires_physical_full_chain() -> None:
    assessment = assess_competitive_match_trajectory(
        trajectory=_trajectory(),
        trajectory_hash=hash_json({"trajectory": "synthetic"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )

    assert assessment.passed
    assert {event.skill for event in assessment.events} == set(CompetitiveSkill)
    assert assessment.first_failed_skill is None


def test_competitive_match_fails_closed_on_body_contact() -> None:
    trajectory = _trajectory()
    trajectory["ball_nonfoot_contact_agent_code"][70] = 3
    assessment = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "body-contact"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )

    assert not assessment.passed
    assert assessment.gates["world_safe"]
    assert not assessment.gates["foot_only_ball_control"]


def test_competitive_match_does_not_infer_a_pass_without_target_commitment() -> None:
    trajectory = _trajectory()
    trajectory["pass_source_agent_code"][:] = 0
    trajectory["pass_target_agent_code"][:] = 0
    assessment = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "no-pass-commitment"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )

    assert not assessment.passed
    assert not assessment.gates["pass_observed"]


def test_competitive_match_does_not_infer_a_shot_without_strike_lease() -> None:
    trajectory = _trajectory()
    trajectory["strike_lease_agent_code"][:] = 0
    assessment = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "no-shot-commitment"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )

    assert not assessment.passed
    assert not assessment.gates["shot_observed"]


def test_competitive_match_does_not_call_a_lateral_blast_a_shot() -> None:
    trajectory = _trajectory()
    trajectory["ball_velocity"][60:75, 0] = 0.0
    trajectory["ball_velocity"][60:75, 1] = 5.0
    assessment = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "lateral-blast"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )

    assert not assessment.passed
    assert not assessment.gates["shot_observed"]


def test_competitive_match_rejects_nonfinite_telemetry() -> None:
    trajectory = _trajectory()
    trajectory["ball_pose"][3, 0] = np.nan

    with pytest.raises(ValueError, match="invalid ball_pose"):
        assess_competitive_match_trajectory(
            trajectory=trajectory,
            trajectory_hash=hash_json({"trajectory": "nan"}),
            agent_ids=_AGENTS,
            roles=_ROLES,
            strict_replay=True,
            world_safe=True,
        )
