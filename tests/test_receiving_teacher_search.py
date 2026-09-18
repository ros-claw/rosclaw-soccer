from dataclasses import asdict, replace

import numpy as np
import pytest

from rosclaw_soccer.growth.locomotion_contact_teacher import G1LocomotionContactTeacherConfig
from rosclaw_soccer.skills.team.independent_team_world import IndependentTeamWorldConfig
from rosclaw_soccer.training.receiving_teacher_search import (
    PARAMETERS,
    apply_receiving_teacher,
    normalized_teacher,
    receiving_feedback_rank,
    sample_teacher_population,
    teacher_parameters,
)


def test_bounded_composition_does_not_change_physical_or_force_limits():
    world = IndependentTeamWorldConfig(
        strict_receive_handoff=True,
        loose_ball_capture_hold=True,
        loose_ball_capture_control=True,
        loose_ball_capture_live_foundation=True,
        loose_ball_capture_follow_navigation=True,
    )
    teacher = G1LocomotionContactTeacherConfig(one_touch_finish_enabled=False)
    w, t = apply_receiving_teacher(np.zeros(7), world, teacher)
    assert {k for k, v in asdict(world).items() if v != asdict(w)[k]} <= {"receive_pacing_ratio"}
    assert {k for k, v in asdict(teacher).items() if v != asdict(t)[k]} <= set(PARAMETERS)
    assert t.maximum_task_force_n == teacher.maximum_task_force_n
    assert t.maximum_joint_residual_nm == teacher.maximum_joint_residual_nm
    assert (
        t.committed_receive_maximum_joint_residual_nm
        == teacher.committed_receive_maximum_joint_residual_nm
    )
    assert w.minimum_pelvis_height_m == world.minimum_pelvis_height_m
    with pytest.raises(ValueError):
        apply_receiving_teacher(np.zeros(7), IndependentTeamWorldConfig(), teacher)


def test_sampling_is_reproducible_bounded_and_retains_incumbent():
    mean = normalized_teacher((0.0, -1.0, -0.06, 0.12, 15.0, 1.0, 0.2))
    kwargs = dict(mean=mean, sigma=np.full(7, 0.6), incumbent=mean, seed=2734, count=8)
    a = sample_teacher_population(**kwargs)
    assert np.array_equal(a, sample_teacher_population(**kwargs))
    assert np.array_equal(a[0], mean) and (abs(a) <= 1).all()
    assert list(teacher_parameters(mean)) == list(PARAMETERS)
    for value in (np.full(7, np.nan), np.ones(8), np.full(7, 1.01)):
        with pytest.raises(ValueError):
            teacher_parameters(value)


def rows():
    return [
        dict(
            course_id=str(i),
            agent_id=f"player{i}",
            safe=True,
            controlled_reception=False,
            shaped_return=0.0,
        )
        for i in range(2)
    ]


def rank(data):
    return receiving_feedback_rank(data, expected_courses=("0", "1"))


def test_reward_cannot_compensate_for_fall_or_lost_role():
    a = rows()
    b = rows()
    b[0].update(safe=False, controlled_reception=True, shaped_return=1e10)
    assert rank(a) > rank(b)
    a[0]["controlled_reception"] = True
    b = rows()
    b[0]["shaped_return"] = 1e10
    assert rank(a) > rank(b)


def test_failures_cannot_be_omitted_or_duplicated():
    for values in (rows()[:1], rows() + [rows()[0]]):
        with pytest.raises(ValueError):
            rank(values)
    a = rows()
    a[0]["shaped_return"] = float("nan")
    with pytest.raises(ValueError):
        rank(a)


@pytest.mark.parametrize(
    "field,value",
    [
        ("committed_receive_velocity_damping_n_per_mps", 15.01),
        ("committed_receive_maximum_task_force_n", 120.01),
        ("committed_receive_maximum_joint_residual_nm", 20.01),
    ],
)
def test_receive_transition_bounds_fail_before_physics(field, value):
    with pytest.raises(ValueError):
        G1LocomotionContactTeacherConfig(**{field: value})


def test_sampled_parameters_survive_actual_receive_transition_mapping():
    teacher = G1LocomotionContactTeacherConfig(one_touch_finish_enabled=False)
    world = IndependentTeamWorldConfig(
        strict_receive_handoff=True,
        loose_ball_capture_hold=True,
        loose_ball_capture_control=True,
        loose_ball_capture_live_foundation=True,
        loose_ball_capture_follow_navigation=True,
    )
    vectors = sample_teacher_population(
        mean=np.zeros(7), sigma=np.ones(7), incumbent=np.zeros(7), seed=73, count=32
    )
    for vector in vectors:
        _, c = apply_receiving_teacher(vector, world, teacher)
        transitioned = replace(
            c,
            receive_follow_through_speed_mps=c.committed_receive_follow_through_speed_mps,
            receive_ankle_lateral_offset_m=c.committed_receive_ankle_lateral_offset_m,
            velocity_damping_n_per_mps=c.committed_receive_velocity_damping_n_per_mps,
            maximum_task_force_n=c.committed_receive_maximum_task_force_n,
            maximum_joint_residual_nm=c.committed_receive_maximum_joint_residual_nm,
            aim_yaw_bias_rad=c.committed_receive_aim_yaw_bias_rad,
        )
        assert transitioned.maximum_joint_residual_nm <= 20


@pytest.mark.parametrize(
    "change",
    [
        {"count": 3},
        {"count": 33},
        {"count": True},
        {"seed": -1},
        {"seed": True},
        {"sigma": np.zeros(7)},
        {"sigma": np.full(7, np.nan)},
        {"sigma": np.full(7, 1.1)},
        {"mean": np.zeros(8)},
    ],
)
def test_population_rejects_invalid_budget_and_scale(change):
    values = dict(
        mean=np.zeros(7), sigma=np.full(7, 0.6), incumbent=np.zeros(7), seed=2734, count=8
    )
    with pytest.raises(ValueError):
        sample_teacher_population(**{**values, **change})
