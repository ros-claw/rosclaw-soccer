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


def test_sprint_records_ball_and_agent_speeds_separately() -> None:
    trajectory = _trajectory()
    trajectory["ball_velocity"][:, :3] = (0.2, 0.1, 0.0)
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "distinct-sprint-speeds"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    sprint = next(event for event in result.events if event.skill is CompetitiveSkill.SPRINT)
    assert sprint.ball_speed_mps == pytest.approx(np.sqrt(0.05))
    assert sprint.agent_speed_mps == pytest.approx(0.6)
    assert sprint.schema_version == "rosclaw_soccer.competitive_skill_event.v2"


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


def test_zero_force_contact_cannot_create_physical_events() -> None:
    trajectory = _trajectory()
    trajectory["ball_contact_force_n"][:] = 0.0
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "zero-force"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert {event.skill for event in result.events} <= {CompetitiveSkill.SPRINT}


@pytest.mark.parametrize("interruption_frame", [30, 40, 50])
def test_body_collision_interrupts_individual_pass_evidence(interruption_frame: int) -> None:
    trajectory = _trajectory()
    trajectory["ball_nonfoot_contact_agent_code"][interruption_frame] = 1
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "interrupted-pass"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert not result.gates["pass_observed"]
    assert all(event.skill is not CompetitiveSkill.PASS for event in result.events)


def test_body_collision_interrupts_dribble_and_win_evidence() -> None:
    trajectory = _trajectory()
    trajectory["ball_nonfoot_contact_agent_code"][[10, 20]] = 1
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "interrupted-dribble-win"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert all(
        event.agent_id != "red.playmaker"
        for event in result.events
        if event.skill is CompetitiveSkill.DRIBBLE
    )
    assert not result.gates["ball_win_observed"]


def test_simultaneous_body_collision_cannot_create_clean_shot() -> None:
    trajectory = _trajectory()
    trajectory["ball_nonfoot_contact_agent_code"][60] = 3
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "mixed-shot-contact"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert not result.gates["shot_observed"]
    assert not result.gates["save_observed"]


def test_intermediate_player_contact_cannot_be_claimed_as_direct_save() -> None:
    trajectory = _trajectory()
    trajectory["ball_contact_agent_code"][70] = 1
    trajectory["ball_contact_effector_code"][70] = 2
    trajectory["ball_contact_force_n"][70] = 40.0
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "deflected-before-save"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert result.gates["shot_observed"]
    assert not result.gates["save_observed"]


def test_existing_fast_ball_is_not_automatically_a_shot() -> None:
    trajectory = _trajectory()
    trajectory["ball_velocity"][:75, 0] = 4.2
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "already-fast-ball"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert not result.gates["shot_observed"]


def test_nonpositive_contact_force_is_not_physical_evidence() -> None:
    trajectory = _trajectory()
    trajectory["ball_contact_force_n"][:] = 1.0
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "exact-threshold"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert {event.skill for event in result.events} <= {CompetitiveSkill.SPRINT}


def test_later_other_player_impact_cannot_create_a_shot() -> None:
    trajectory = _trajectory()
    trajectory["ball_velocity"][:, :] = 0.0
    trajectory["ball_velocity"][65:75, 0] = 4.2
    trajectory["ball_contact_agent_code"][65] = 1
    trajectory["ball_contact_effector_code"][65] = 2
    trajectory["ball_contact_force_n"][65] = 40.0
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "later-unrelated-impact"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert not result.gates["shot_observed"]


def test_later_impact_cannot_be_attributed_to_goalkeeper_save() -> None:
    trajectory = _trajectory()
    trajectory["ball_velocity"][75, 0] = 4.2
    trajectory["ball_contact_agent_code"][76] = 1
    trajectory["ball_contact_effector_code"][76] = 2
    trajectory["ball_contact_force_n"][76] = 40.0
    result = assess_competitive_match_trajectory(
        trajectory=trajectory,
        trajectory_hash=hash_json({"trajectory": "late-save-impact"}),
        agent_ids=_AGENTS,
        roles=_ROLES,
        strict_replay=True,
        world_safe=True,
    )
    assert result.gates["shot_observed"]
    assert not result.gates["save_observed"]


def test_negative_force_and_invalid_nonfoot_identity_are_rejected() -> None:
    for key, value in (("ball_contact_force_n", -0.1), ("ball_nonfoot_contact_agent_code", 5)):
        trajectory = _trajectory()
        trajectory[key][40] = value
        with pytest.raises(ValueError, match="trajectory shape or clock is invalid"):
            assess_competitive_match_trajectory(
                trajectory=trajectory,
                trajectory_hash=hash_json({"invalid": key}),
                agent_ids=_AGENTS,
                roles=_ROLES,
                strict_replay=True,
                world_safe=True,
            )
