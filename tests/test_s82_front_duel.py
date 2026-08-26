from __future__ import annotations

import pytest

from rosclaw_soccer.growth.ballistic_contact_impulse_actor import (
    g1_ballistic_contact_impulse_context_hash,
)
from rosclaw_soccer.skills.shoot.free_kick import G1FreeKickFlowConfig
from rosclaw_soccer.skills.team.front_duel import G1FrontDuelConfig, G1FrontDuelSummary
from rosclaw_soccer.training.front_duel_actor import G1FrontDuelCurriculum


def test_front_duel_layout_keeps_teammate_out_of_striker_lane() -> None:
    assert G1FrontDuelConfig().teammate_origin_m == (2.0, -2.45, 0.0)
    with pytest.raises(ValueError, match="outside the striker lane"):
        G1FrontDuelConfig(teammate_origin_m=(2.0, 1.49, 0.0))


def test_front_duel_summary_requires_stable_independent_agents() -> None:
    passing = G1FrontDuelSummary(
        finite_state=True,
        teammate_minimum_pelvis_height_m=0.75,
        goalkeeper_minimum_pelvis_height_m=0.75,
        goalkeeper_peak_tilt_rad=0.05,
        goalkeeper_peak_lateral_speed_mps=0.62,
        goalkeeper_reaction_frames=46,
        goalkeeper_ball_contact_observed=False,
        goalkeeper_ball_contact_time_sec=None,
        robot_robot_contact_count=0,
        actuator_saturation=False,
        actuator_saturation_steps=0,
        actuator_saturation_fraction=0.0,
        actuator_peak_demand_ratio=0.99,
        joint_limit_violation=False,
        torque_authority_projection_steps=2,
        torque_authority_projection_fraction=0.0004,
        torque_authority_projection_peak_correction_nm=0.42,
        torque_authority_preprojection_peak_demand_ratio=1.02,
        torque_authority_projection_qualified=True,
    )

    assert passing.passed
    unsafe = G1FrontDuelSummary(
        **{
            key: value
            for key, value in passing.to_dict().items()
            if key not in {
                "passed",
                "schema_version",
                "actuator_saturation",
                "actuator_saturation_steps",
                "actuator_saturation_fraction",
            }
        },
        actuator_saturation=True,
        actuator_saturation_steps=2,
        actuator_saturation_fraction=0.0004,
    )
    assert not unsafe.passed


def test_contact_actor_context_is_bound_to_three_player_scene() -> None:
    common = {
        "flow_config": {"foo": 1, "shot_loft_teacher_max_force_n": 250.0},
        "goal_spec": {"target_y_m": 1.8, "target_z_m": 0.86},
        "runup_config": {"speed": 1.0},
        "sonic_runup_config": None,
        "approach_strike_candidate_hash": "sha256:candidate",
    }

    single_world = g1_ballistic_contact_impulse_context_hash(**common)
    shared_world = g1_ballistic_contact_impulse_context_hash(
        **common,
        front_duel_config=G1FrontDuelConfig().to_dict(),
    )
    changed_layout = g1_ballistic_contact_impulse_context_hash(
        **common,
        front_duel_config=G1FrontDuelConfig(
            teammate_origin_m=(2.0, -3.0, 0.0)
        ).to_dict(),
    )

    assert single_world != shared_world
    assert shared_world != changed_layout


def test_front_duel_curriculum_contains_success_and_failure_support() -> None:
    curriculum = G1FrontDuelCurriculum()

    assert len(curriculum.teacher_force_pairs_n) == 8
    assert len(set(curriculum.teacher_force_pairs_n)) == 8
    assert any(
        lateral >= 160.0 and vertical >= 250.0
        for lateral, vertical in curriculum.teacher_force_pairs_n
    )
    assert any(
        lateral <= 30.0 and vertical <= 30.0
        for lateral, vertical in curriculum.teacher_force_pairs_n
    )
    assert curriculum.target_y_m >= 1.5
    assert curriculum.target_z_m >= 0.60
    assert curriculum.ball_mass_kg == pytest.approx(0.41)


def test_front_duel_curriculum_rejects_undercovered_rehearsal() -> None:
    with pytest.raises(ValueError, match="eight unique probes"):
        G1FrontDuelCurriculum(teacher_force_pairs_n=((180.0, 250.0),) * 8)


def test_front_duel_projects_auxiliary_agents_below_saturation() -> None:
    config = G1FrontDuelConfig()
    assert config.torque_authority_projection_ratio == pytest.approx(0.99)
    assert config.torque_authority_projection_max_fraction == pytest.approx(0.01)


def test_jointwise_contact_task_projection_requires_final_authority_bound() -> None:
    with pytest.raises(ValueError, match="final authority bound"):
        G1FreeKickFlowConfig(contact_task_direction_projection_enabled=False)

    configured = G1FreeKickFlowConfig(
        torque_authority_projection_ratio=0.98,
        contact_task_direction_projection_enabled=False,
    )
    assert configured.torque_authority_projection_ratio == pytest.approx(0.98)
    assert not configured.contact_task_direction_projection_enabled
