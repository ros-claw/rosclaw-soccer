"""Curriculum source balance, lifecycle guard and declared reward bounds."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_contact_course import NativeContactCourse, balanced_course_indices


def bank():
    return dict(
        source_id=np.repeat(np.arange(64) % 8, 4), lead_controls=np.tile([10, 20, 40, 80], 64)
    )


@pytest.mark.parametrize("worlds", [8, 16, 32, 64, 256])
@pytest.mark.parametrize("lead", [None, 10, 20, 40, 80])
def test_source_balance_and_pool_coverage(worlds, lead):
    b = bank()
    indices = []
    for episode in range(32):
        ids = balanced_course_indices(b, worlds=worlds, episode=episode, lead=lead)
        assert np.array_equal(np.bincount(b["source_id"][ids]), np.full(8, worlds // 8))
        if lead is not None:
            assert np.all(b["lead_controls"][ids] == lead)
        indices.extend(ids.tolist())
    assert len(set(indices)) == (256 if lead is None else 64)


def test_active_and_faulted_course_cannot_reset_or_mutate():
    c = NativeContactCourse.__new__(NativeContactCourse)
    c._closed, c.faulted, c.batch = False, False, object()
    c.ended = np.array([True, False])
    with pytest.raises(ValueError, match="cannot reset"):
        c.reset()
    c.faulted = True
    c.ended[:] = True
    with pytest.raises(ValueError, match="cannot reset"):
        c.reset()
    with pytest.raises(ValueError, match="active native course"):
        c.step(np.zeros((2, 29), dtype=np.float32))


def test_course_window_reward_dominance_and_observation_abi():
    # Five tracking terms + proximity <=6 per control; maximum once-per-episode
    # contact50 + five*(alignment30+speed10)=250, all other terms nonpositive.
    assert NativeContactCourse.window * 6 + 250 < 400
    # Experimental receiving has at most 50 foot + 200 capture once per
    # episode; extending it to 40 controls requires a larger failure penalty.
    assert 40 * 6 + 250 < 600
    assert 60 * 6 + 250 < 800
    assert (
        NativeContactCourse.actor_observation_dim
        == NativeContactCourse.critic_observation_dim
        == 177
    )


def test_pre_step_rejections_freeze_only_declared_live_worlds():
    course = NativeContactCourse.__new__(NativeContactCourse)
    course._closed, course.faulted, course.batch = False, False, object()
    course.worlds = 3
    course.ended = np.array([False, False, True])
    course.reject_worlds(np.array([True, False, False]))
    np.testing.assert_array_equal(course.ended, [True, False, True])
    with pytest.raises(ValueError, match="live-world"):
        course.reject_worlds(np.array([True, False, False]))
    with pytest.raises(ValueError, match="live-world"):
        course.reject_worlds(np.array([False, False, False]))
