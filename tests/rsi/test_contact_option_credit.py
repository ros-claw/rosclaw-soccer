import numpy as np
import pytest

from rosclaw_soccer.rsi.contact_option_credit import (
    contact_option_gae,
    measured_tail_return,
    recovery_terminal_reward,
)


def test_terminal_reward_requires_real_correct_foot_outcome_and_whole_option_safety():
    row = dict(
        stable=True,
        forbidden=False,
        postcontact=dict(
            post_contact_quality_observed=True,
            correct_foot_first=True,
            goal_progress_1p2s_m=1.0,
            goal_lateral_1p2s_m=0.0,
        ),
    )
    assert recovery_terminal_reward(row) == 12.0
    row["postcontact"]["goal_progress_1p2s_m"] = 0.5
    assert recovery_terminal_reward(row) == 1.0
    row["postcontact"]["correct_foot_first"] = False
    assert recovery_terminal_reward(row) == 0.0
    row["postcontact"]["post_contact_quality_observed"] = False
    assert recovery_terminal_reward(row) == 0.0
    row["forbidden"] = True
    assert recovery_terminal_reward(row) == -20.0
    row["forbidden"], row["stable"] = False, False
    assert recovery_terminal_reward(row) == -20.0


def test_whole_option_safety_penalty_dominates_all_head_positive_shaping():
    head_maximum = 0.02 * (24 * 6 + 250)
    assert head_maximum + 0.99**60 * (-20) < 0


def test_real_tail_outcome_reaches_contact_credit_with_exact_discount():
    zero = np.zeros((24, 1), dtype=np.float32)
    valid = np.ones((24, 1), dtype=bool)
    term = np.zeros((24, 1), dtype=bool)
    handoff = term.copy()
    handoff[-1] = True
    tail = np.zeros((60, 1), dtype=np.float32)
    tail[-1] = 1
    tv = np.ones((60, 1), dtype=bool)
    tt = np.zeros_like(tv)
    tt[-1] = True
    advantage, returns = contact_option_gae(
        zero, zero, zero, term, handoff, valid, tail_reward=tail, tail_valid=tv, tail_terminal=tt
    )
    np.testing.assert_allclose(advantage[-1, 0], 0.99**60, rtol=2e-6)
    np.testing.assert_allclose(advantage[0, 0], (0.99 * 0.95) ** 23 * 0.99**60, rtol=2e-6)
    assert np.array_equal(advantage, returns)
    # No fictitious critic value after finite task termination.
    huge = zero + 999
    future = measured_tail_return(tail, tv, tt)
    a, _ = contact_option_gae(
        zero, zero, huge, term, handoff, valid, tail_reward=tail, tail_valid=tv, tail_terminal=tt
    )
    np.testing.assert_allclose(a[-1], 0.99 * future)


def test_failed_contact_cannot_claim_future_reward():
    r = np.zeros((24, 1), dtype=np.float32)
    v = np.zeros_like(r, dtype=bool)
    v[:3] = True
    term = np.zeros_like(v)
    term[2] = True
    handoff = np.zeros_like(v)
    tr = np.zeros((60, 1), dtype=np.float32)
    tv = np.zeros_like(tr, dtype=bool)
    a, _ = contact_option_gae(
        r, r, r, term, handoff, v, tail_reward=tr, tail_valid=tv, tail_terminal=tv
    )
    assert not a.any()
    tv[0] = True
    tr[0] = 1
    with pytest.raises(ValueError):
        contact_option_gae(
            r, r, r, term, handoff, v, tail_reward=tr, tail_valid=tv, tail_terminal=tv
        )


def test_early_handoff_cannot_shorten_declared_contact_option():
    r = np.zeros((24, 1), dtype=np.float32)
    valid = np.zeros_like(r, dtype=bool)
    valid[:3] = True
    event = np.zeros_like(valid)
    event[2] = True
    tail = np.ones((1, 1), dtype=np.float32)
    mask = np.ones_like(tail, dtype=bool)
    with pytest.raises(ValueError, match="all24"):
        contact_option_gae(
            r,
            r,
            r,
            np.zeros_like(event),
            event,
            valid,
            tail_reward=tail,
            tail_valid=mask,
            tail_terminal=mask,
        )


@pytest.mark.parametrize("defect", ["no_terminal", "hole", "post_reward", "two_terminals"])
def test_incomplete_or_restarted_tail_is_rejected(defect):
    r = np.zeros((60, 1), dtype=np.float32)
    v = np.zeros_like(r, dtype=bool)
    v[:10] = True
    t = np.zeros_like(v)
    t[9] = True
    if defect == "no_terminal":
        t[:] = False
    if defect == "hole":
        v[3] = False
    if defect == "post_reward":
        r[10] = 1
    if defect == "two_terminals":
        t[3] = True
    with pytest.raises(ValueError):
        measured_tail_return(r, v, t)
