import numpy as np
import pytest

from rosclaw_soccer.rsi.motion_learning_windows import correction_window, success_memory_window


def test_correction_does_not_learn_post_contact_or_timeout_standing():
    a = correction_window(
        first_foot_frame=64,
        first_nonfoot_frame=None,
        teacher_handoff_frame=None,
        parent_qualified=False,
    )
    np.testing.assert_array_equal(np.flatnonzero(a), np.arange(20, 64))
    b = correction_window(
        first_foot_frame=None,
        first_nonfoot_frame=None,
        teacher_handoff_frame=None,
        parent_qualified=False,
    )
    np.testing.assert_array_equal(np.flatnonzero(b), np.arange(20, 91))
    c = correction_window(
        first_foot_frame=64,
        first_nonfoot_frame=30,
        teacher_handoff_frame=40,
        parent_qualified=False,
    )
    np.testing.assert_array_equal(np.flatnonzero(c), np.arange(20, 30))


def test_parent_success_is_not_overwritten_by_conflicting_teacher():
    assert not correction_window(
        first_foot_frame=35,
        first_nonfoot_frame=None,
        teacher_handoff_frame=None,
        parent_qualified=True,
    ).any()
    retention = success_memory_window(first_foot_frame=35, qualified=True, full_retention=True)
    assert retention.sum() == 280 and not retention[:20].any()


def test_empirical_success_contact_window_uses_only_verified_qualified_episodes():
    window = success_memory_window(first_foot_frame=42, qualified=True, full_retention=False)
    np.testing.assert_array_equal(np.flatnonzero(window), np.arange(22, 47))
    with pytest.raises(ValueError):
        success_memory_window(first_foot_frame=240, qualified=True, full_retention=False)
    with pytest.raises(ValueError):
        success_memory_window(first_foot_frame=42, qualified=False, full_retention=False)


def test_invalid_frame_rejected_not_clipped():
    with pytest.raises(ValueError):
        correction_window(
            first_foot_frame=-1,
            first_nonfoot_frame=None,
            teacher_handoff_frame=None,
            parent_qualified=False,
        )
