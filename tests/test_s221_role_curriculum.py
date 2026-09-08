from dataclasses import replace
from pathlib import Path

import pytest

from rosclaw_soccer.growth.near_ball_residual import NearBallResidualPolicy
from rosclaw_soccer.training import near_ball_curriculum as module


def test_courses_cover_all_roles_and_keep_exam_positions_separate():
    train = [c for i in range(5) for c in module.training_courses(i)]
    exam = module.examination_courses()
    assert len(exam) == len({c.key for c in exam}) == 16
    assert len({(c.role, c.blue) for c in train}) == 8
    assert not {c.offset for c in train} & {c.offset for c in exam}
    with pytest.raises(ValueError):
        module.RoleCourse("coach", False, 0.0)
    with pytest.raises(ValueError):
        module.training_courses(True)


def test_collection_uses_shared_physics_and_private_parent(monkeypatch, tmp_path: Path):
    checkpoint = tmp_path / "parent.npz"
    policy = NearBallResidualPolicy.initialize(
        tuple(f"agent.{i}" for i in range(8)), "sha256:" + "a" * 64
    )
    policy.save(checkpoint)
    calls = []
    monkeypatch.setattr(module, "run_probe", lambda **kwargs: calls.append(kwargs))
    job = module.RoleRolloutJob(
        tmp_path,
        tmp_path / "run",
        checkpoint,
        12,
        module.RoleCourse("goalkeeper", True, 0),
        9,
        True,
    )
    module.collect_role_course(job)
    call = calls[-1]
    assert call["near_ball_policy"].policy_hash == policy.policy_hash
    assert call["near_ball_explore"] and call["near_ball_seed"] == 9
    assert call["basic_ball_play"] and call["all_role_clearance"]
    assert call["kickoff_role"] == "goalkeeper" and not call["forward_receiver_lane"]
    module.collect_role_course(
        replace(job, explore=False, course=module.RoleCourse("playmaker", False, 0.06))
    )
    assert not calls[-1]["near_ball_explore"] and calls[-1]["forward_receiver_lane"]
