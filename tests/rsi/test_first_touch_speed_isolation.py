"""Guard the predeclared ablation and exact trajectory comparisons."""

import numpy as np

from scripts.rsi_isolate_first_touch_speed import arms, exact_arrays


def test_one_factor_arms_are_frozen():
    assert arms() == (
        ("strict-1.0", False, 1.0),
        ("experimental-1.0", True, 1.0),
        ("experimental-1.5", True, 1.5),
    )
    assert arms(True) == (("experimental-1.5", True, 1.5), ("feedback-1.5", True, 1.5))


def test_replay_rejects_missing_or_changed_fields():
    baseline = {"qpos": np.array([1.0, 2.0]), "torque": np.array([0.0])}
    assert exact_arrays(baseline, {k: v.copy() for k, v in baseline.items()})
    assert not exact_arrays(baseline, {"qpos": baseline["qpos"]})
    assert not exact_arrays(baseline, {"qpos": np.array([1.0, 2.01]), "torque": baseline["torque"]})
