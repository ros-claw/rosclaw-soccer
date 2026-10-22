import copy

import numpy as np
import pytest

from scripts.rsi_stroke_world_audit import verify_matched_prefix


def raw():
    return dict(
        kick={
            k: np.zeros((100, 2), np.float32)
            for k in (
                "NN_obs",
                "NN_h_before",
                "NN_c_before",
                "NN_action",
                "qpos",
                "qvel",
                "forces",
                "torque",
            )
        }
    )


def test_only_the_pre_intervention_prefix_must_match():
    a, b = raw(), raw()
    before = copy.deepcopy(a)
    b["kick"]["NN_action"][80] = 1
    verify_matched_prefix(a, b)
    assert all(np.array_equal(a["kick"][k], before["kick"][k]) for k in a["kick"])
    b["kick"]["NN_obs"][79] = 1
    with pytest.raises(ValueError):
        verify_matched_prefix(a, b)


def test_early_termination_cannot_hide_a_different_prefix():
    a, b = raw(), raw()
    a["kick"] = {k: v[:20] for k, v in a["kick"].items()}
    with pytest.raises(ValueError):
        verify_matched_prefix(a, b)
    b["kick"] = {k: v[:20] for k, v in b["kick"].items()}
    verify_matched_prefix(a, b)
