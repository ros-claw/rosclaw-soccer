"""Split is fixed before optimization; filenames are not unseen-world proof."""

from pathlib import PurePath

import pytest

from rosclaw_soccer.rsi.mjlab_motion_curriculum import (
    DEVELOPMENT_MOTION_NAMES,
    TRAIN_MOTION_GLOB,
    TRAIN_MOTION_NAMES,
    MotionLearningBudget,
    motion_checkpoint_filename,
    motion_names_for_pattern,
    require_completed_motion_smoke,
    require_completed_stable_motion_course,
    require_completed_stable_motion_smoke,
)


def test_eight_training_sources_and_balanced_two_development_sources_are_disjoint():
    assert len(TRAIN_MOTION_NAMES) == 8
    assert len(DEVELOPMENT_MOTION_NAMES) == 2
    assert not set(TRAIN_MOTION_NAMES) & set(DEVELOPMENT_MOTION_NAMES)
    assert motion_names_for_pattern(TRAIN_MOTION_GLOB) == TRAIN_MOTION_NAMES
    assert all(PurePath(x).match(TRAIN_MOTION_GLOB) for x in TRAIN_MOTION_NAMES)
    assert not any(PurePath(x).match(TRAIN_MOTION_GLOB) for x in DEVELOPMENT_MOTION_NAMES)
    for name in DEVELOPMENT_MOTION_NAMES:
        assert motion_names_for_pattern(name) == (name,)


@pytest.mark.parametrize("pattern", ["*", "soccer-standard-*.npz", "../*", None, True])
def test_no_future_expansion_of_training_split(pattern):
    with pytest.raises(ValueError):
        motion_names_for_pattern(pattern)


def completed_smoke():
    return dict(
        status="MJLAB_MOTION_GPU_PPO_SMOKE_NOT_SKILL_QUALIFIED",
        gpu_step_calls=560,
        gpu_world_substeps=17920,
        policy_control_samples=1536,
        new_cpu_native_steps=0,
        new_optimizer_steps=40,
        optimizer_counts=[dict(attempted=40, completed=40)],
        changed_actor_weights=["mlp.0.weight"],
        changed_critic_weights=["mlp.0.bias"],
        weights_only_checkpoint_reload=True,
        football_skill_proven=False,
        goal_complete=False,
        hardware_authorized=False,
        promotion_authorized=False,
    )


def test_frozen_smoke_and_bounded_course_counts():
    smoke = MotionLearningBudget()
    assert smoke.training_control_steps == 48
    assert smoke.gpu_world_substeps == 17920
    course = MotionLearningBudget("course", 256, 100, 502)
    assert course.training_control_steps == 2400
    assert course.gpu_world_substeps == 6164480
    require_completed_motion_smoke(completed_smoke())


@pytest.mark.parametrize(
    "kwargs",
    [
        dict(mode="other"),
        dict(num_envs=64),
        dict(iterations=3),
        dict(seed=502),
        dict(num_envs=True),
        dict(iterations=2.0),
        dict(seed=-1),
        dict(mode="course", iterations=2),
        dict(mode="course", iterations=201),
        dict(mode="course", iterations=3, num_envs=513),
        dict(mode="course", iterations=3, num_envs=31),
        dict(mode="course", iterations=3, seed=2**31),
    ],
)
def test_no_implicit_smoke_expansion_or_unbounded_course(kwargs):
    with pytest.raises(ValueError):
        MotionLearningBudget(**kwargs)


@pytest.mark.parametrize(
    ("key", "value"),
    [
        ("status", "RUNNING"),
        ("gpu_step_calls", 559),
        ("gpu_world_substeps", 0),
        ("policy_control_samples", 0),
        ("new_cpu_native_steps", False),
        ("new_optimizer_steps", 0),
        ("new_optimizer_steps", True),
        ("optimizer_counts", []),
        ("optimizer_counts", [dict(attempted=40, completed=39)]),
        ("optimizer_counts", [dict(attempted=1, completed=1)]),
        ("optimizer_counts", [None]),
        ("changed_actor_weights", []),
        ("changed_critic_weights", ["running_mean"]),
        ("changed_actor_weights", [None]),
        ("weights_only_checkpoint_reload", False),
        ("hardware_authorized", True),
        ("football_skill_proven", True),
        ("goal_complete", True),
        ("promotion_authorized", True),
    ],
)
def test_course_does_not_unlock_from_incomplete_or_promoted_smoke(key, value):
    report = completed_smoke()
    report[key] = value
    with pytest.raises(ValueError):
        require_completed_motion_smoke(report)


def stable_smoke():
    report = completed_smoke()
    report.update(
        status="MJLAB_STABLE_PRIOR_REHEARSAL_PPO_SMOKE_NOT_SKILL_QUALIFIED",
        stable_prior_rehearsal=True,
        iterations=2,
        num_envs=32,
        seed=501,
        new_optimizer_steps=42,
        optimizer_counts=[dict(attempted=42, completed=42)],
        native_rehearsal=dict(
            bank_hash="sha256:" + "a" * 64,
            attempted_updates=2,
            updates=2,
            batch_size=256,
            rows=512,
            weight=1.0,
        ),
    )
    return report


def test_stable_course_requires_its_own_smoke_not_old_random_policy():
    require_completed_stable_motion_smoke(stable_smoke())
    with pytest.raises(ValueError):
        require_completed_stable_motion_smoke(completed_smoke())
    with pytest.raises(ValueError):
        require_completed_motion_smoke(stable_smoke())


@pytest.mark.parametrize(
    "key,value",
    [
        ("updates", 1),
        ("attempted_updates", 3),
        ("rows", 511),
        ("batch_size", 128),
        ("weight", 0),
        ("bank_hash", "unbound"),
    ],
)
def test_stable_course_rejects_incomplete_rehearsal(key, value):
    report = stable_smoke()
    report["native_rehearsal"][key] = value
    with pytest.raises(ValueError):
        require_completed_stable_motion_smoke(report)


@pytest.mark.parametrize("missing", [None, [], "passed"])
def test_missing_rehearsal_fails_closed(missing):
    report = stable_smoke()
    report["native_rehearsal"] = missing
    with pytest.raises(ValueError):
        require_completed_stable_motion_smoke(report)


def test_explicit_actor_checkpoint_is_not_misnamed_as_ppo_iteration():
    assert motion_checkpoint_filename(dict(iterations=2)) == "learner-update2.pt"
    assert (
        motion_checkpoint_filename(dict(checkpoint_file="actor-imitation50.pt"))
        == "actor-imitation50.pt"
    )


@pytest.mark.parametrize("name", ["../x.pt", "/x.pt", "dir/x.pt", ".hidden.pt", "x.pkl", None])
def test_checkpoint_artifact_cannot_escape_its_evidence_root(name):
    with pytest.raises(ValueError):
        motion_checkpoint_filename(dict(checkpoint_file=name))


def stable_course():
    report = stable_smoke()
    report.update(
        status="MJLAB_STABLE_PRIOR_REHEARSAL_PPO_COURSE_NOT_SKILL_QUALIFIED",
        mode="course",
        num_envs=128,
        iterations=200,
        seed=2501,
        gpu_step_calls=48080,
        gpu_world_substeps=6154240,
        policy_control_samples=614400,
        new_optimizer_steps=4200,
        optimizer_counts=[dict(attempted=4200, completed=4200)],
    )
    report["native_rehearsal"].update(updates=202, attempted_updates=202, rows=51712)
    return report


def test_actual_online_course_counts_do_not_inherit_smoke_or_promote():
    report = stable_course()
    require_completed_stable_motion_course(report)
    with pytest.raises(ValueError):
        require_completed_stable_motion_course(stable_smoke())
    with pytest.raises(ValueError):
        require_completed_stable_motion_smoke(report)


@pytest.mark.parametrize(
    "key,value",
    [
        ("gpu_step_calls", 48079),
        ("gpu_world_substeps", 6154241),
        ("policy_control_samples", 614399),
        ("new_optimizer_steps", 4199),
        ("optimizer_counts", []),
        ("optimizer_counts", [dict(attempted=4201, completed=4200)]),
        ("changed_actor_weights", []),
        ("changed_critic_weights", ["distribution.std"]),
        ("hardware_authorized", True),
        ("promotion_authorized", True),
        ("native_rehearsal", None),
        ("weights_only_checkpoint_reload", False),
    ],
)
def test_incomplete_course_cannot_reach_native_contact_research(key, value):
    report = stable_course()
    report[key] = value
    with pytest.raises(ValueError):
        require_completed_stable_motion_course(report)


def test_course_must_include_all_replay_updates_from_two_iteration_parent():
    report = stable_course()
    report["native_rehearsal"]["rows"] = 51200
    with pytest.raises(ValueError, match="replay"):
        require_completed_stable_motion_course(report)
