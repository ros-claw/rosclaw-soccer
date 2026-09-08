import pytest

from rosclaw_soccer.training.near_ball_curriculum import (
    RoleCourse,
    examination_courses,
    training_batch,
    training_courses,
)
from rosclaw_soccer.training.near_ball_learning_audit import _verify_training_course
from rosclaw_soccer.training.near_ball_residual_ppo import train


def test_balanced_batch_covers_all_training_offsets_without_exam_leakage():
    courses = training_batch(2, rounds=5)
    assert len(courses) == len({c.key for c in courses}) == 40
    for role in {c.role for c in courses}:
        for blue in (False, True):
            assert len([c for c in courses if c.role == role and c.blue == blue]) == 5
    assert not {c.offset for c in courses} & {
        c.offset for c in examination_courses(strict_handoff=True)
    }
    assert training_batch(2) == training_courses(2)


@pytest.mark.parametrize("rounds", [0, 6, True, 1.5])
def test_invalid_batch_is_rejected_before_collection(tmp_path, rounds):
    with pytest.raises(ValueError):
        training_batch(0, rounds=rounds)
    with pytest.raises(ValueError, match="budget"):
        train(
            assets=tmp_path,
            output=tmp_path / "run",
            iterations=1,
            duration=12,
            role_curriculum=True,
            prospective_curriculum=True,
            all_role_clearance=True,
            role_batch_rounds=rounds,
        )
    assert not (tmp_path / "run").exists()


def test_large_batch_requires_role_curriculum(tmp_path):
    with pytest.raises(ValueError, match="budget"):
        train(
            assets=tmp_path, output=tmp_path / "run", iterations=1, duration=12, role_batch_rounds=5
        )


@pytest.mark.parametrize("role", ["goalkeeper", "defender", "playmaker", "finisher"])
@pytest.mark.parametrize("blue", [False, True])
def test_audit_binds_actual_role_ball_position_and_seed(role, blue):
    side = "blue" if blue else "red"
    course = RoleCourse(role, blue, 0.12)
    origin_y = 0.7
    ball_y = (
        (1.22 - course.offset if blue else -1.22 + course.offset)
        if role == "playmaker"
        else origin_y
    )
    report = {
        "basic_ball_play": True,
        "kickoff_role": role,
        "forward_receiver_lane": role == "playmaker",
        "near_ball_residual": {"seed": 22100},
        "scenario": {"scenario_id": f"course-{side}", "ball_initial_position_m": [0, ball_y, 0.11]},
        "players": [{"agent_id": f"{side}.{role}", "origin_m": [0, origin_y, 0.78]}],
    }
    _verify_training_course(report, course, 22100)
    with pytest.raises(ValueError, match="seed"):
        _verify_training_course(report, course, 22101)
    with pytest.raises(ValueError, match="position"):
        _verify_training_course(report, RoleCourse(role, blue, 0.10), 22100)
    with pytest.raises(ValueError, match="side"):
        _verify_training_course(report, RoleCourse(role, not blue, 0.12), 22100)
