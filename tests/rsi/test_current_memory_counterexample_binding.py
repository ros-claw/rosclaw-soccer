import numpy as np
import pytest

from scripts.rsi_retest_current_memory_counterexample import (
    check_execution_parent,
    check_trace_arrays,
)


def test_same_runner_parent_is_required_before_an_actor_runs():
    check_execution_parent(
        dict(source_hash="new", asset_hash="g1"), runner_hash="new", asset_hash="g1"
    )
    for report in (
        dict(source_hash="old", asset_hash="g1"),
        dict(source_hash="new", asset_hash="other"),
        {},
    ):
        with pytest.raises(ValueError, match="same-source"):
            check_execution_parent(report, runner_hash="new", asset_hash="g1")


def test_parent_and_actor_controls_use_all_physical_arrays_not_just_outcomes():
    old = dict(pose=np.arange(12, dtype=np.float32).reshape(3, 4), ball=np.zeros((3, 3)))
    check_trace_arrays(old, {k: v.copy() for k, v in old.items()})
    new = {k: v.copy() for k, v in old.items()}
    new["pose"][1, 2] += 1e-5
    with pytest.raises(ValueError, match="every unchanged"):
        check_trace_arrays(old, new)
