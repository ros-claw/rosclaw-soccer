import numpy as np
import pytest

from rosclaw_soccer.rsi.receiving_contact_quality_reward import contact_quality_rewards


def history(n=20):
    return (
        np.zeros((n, 2)),
        np.zeros((n, 2)),
        np.zeros((n, 10, 3)),
        np.zeros(n, dtype=np.bool_),
    )


def test_foot_first_and_retention_are_separate_observed_events():
    r, b, f, fail = history()
    f[2, 3, 0] = 11
    out = contact_quality_rewards(r, b, f, fail)
    assert out[2] == 1 and out[12] == 1 and out.sum() == 2
    b[12, 0] = 0.1  # 5 m/s relative speed; proximity alone is not retention.
    assert contact_quality_rewards(r, b, f, fail).sum() == 1


def test_simultaneous_nonfoot_and_preceding_nonfoot_get_no_positive_credit():
    r, b, f, fail = history()
    f[2, 3, 0] = 11
    f[2, 3, 2] = 2
    assert contact_quality_rewards(r, b, f, fail).sum() == -1
    f[2, 3, 2] = 0
    f[2, 2, 2] = 2
    assert contact_quality_rewards(r, b, f, fail).sum() == -1


def test_later_nonfoot_fall_and_censor_do_not_create_retention():
    r, b, f, fail = history()
    f[2, 3, 0] = 11
    f[12, 9, 2] = 2
    assert contact_quality_rewards(r, b, f, fail).sum() == 0
    f[12, 9, 2] = 0
    fail[8:] = True
    assert contact_quality_rewards(r, b, f, fail).sum() == -1
    r, b, f, fail = history(8)
    f[2, 3, 0] = 11
    assert contact_quality_rewards(r, b, f, fail).sum() == 1


def test_reject_misaligned_nonfinite_or_negative_forces():
    r, b, f, fail = history()
    f[0, 0, 0] = -1
    with pytest.raises(ValueError):
        contact_quality_rewards(r, b, f, fail)
    f[0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        contact_quality_rewards(r, b, f, fail)
