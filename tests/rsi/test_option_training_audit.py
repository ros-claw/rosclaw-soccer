"""Synthetic read-only raw-data audits; no simulator or neural execution."""

import numpy as np
import pytest

from scripts.rsi_audit_option_training import reconstruct


def fixture():
    q = np.zeros((84, 1, 10, 43))
    q[..., 2], q[..., 3] = 0.8, 1.0
    q[..., 36] = np.arange(840).reshape(84, 1, 10) * 0.002
    f = np.zeros((84, 1, 10, 3))
    f[10, 0, 0, 0] = 11.0
    parts = []
    for sl in (slice(0, 24), slice(24, 84)):
        n = len(q[sl])
        parts.append(
            dict(
                qpos=q[sl].copy(),
                qvel=np.zeros((n, 1, 10, 41)),
                forces=f[sl].copy(),
                valid=np.ones((n, 1), dtype=bool),
                terminated=np.zeros((n, 1), dtype=bool),
            )
        )
    tail = parts[1]
    tail["completed"] = np.zeros((60, 1), dtype=bool)
    tail["completed"][-1] = True
    tail["terminal"] = tail["completed"].copy()
    tail["reward"] = np.zeros((60, 1), dtype=np.float32)
    tail["reward"][-1] = 12.0
    kwargs = dict(
        qi=np.arange(43),
        vi=np.arange(41),
        goals=np.array([[10.0, 0, 0]]),
        feet=np.array([0]),
        indices=np.array([0]),
    )
    return parts, kwargs


def test_reconstructs_actual_success_at_last_tail_sample():
    parts, kwargs = fixture()
    result = reconstruct(*parts, **kwargs)[0]
    assert result["recovery_reward"] == 12.0
    assert result["recovery_controls"] == 60
    assert result["postcontact"]["correct_foot_first"]
    assert result["postcontact"]["goal_progress_1p2s_m"] == pytest.approx(1.2)


@pytest.mark.parametrize("mutation", ["reward", "boundary", "hole", "nonfinite"])
def test_rejects_corrupted_recorded_data(mutation):
    parts, kwargs = fixture()
    tail = parts[1]
    if mutation == "reward":
        tail["reward"][0] = 1.0
    elif mutation == "boundary":
        tail["terminal"][0] = True
    elif mutation == "hole":
        tail["valid"][5] = False
    else:
        tail["qvel"][0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError):
        reconstruct(*parts, **kwargs)
