import numpy as np
import pytest

from scripts.rsi_focal_authority_ppo_pilot import learning_mask, same_arrays


def test_only_complete_active_explored_full_authority_focal_frames_admitted():
    ids = ("blue.playmaker", "red.finisher")
    shape = (150, 2)
    trace = {
        "residual_active": np.ones(shape, dtype=bool),
        "residual_exploration_mask": np.ones(shape, dtype=bool),
        "residual_pd_target_injected_substeps": np.full(shape, 10, dtype=np.int64),
        "residual_pd_target_override_substeps": np.zeros(shape, dtype=np.int64),
    }
    trace["residual_active"][5, 1] = False
    trace["residual_exploration_mask"][6, 1] = False
    trace["residual_pd_target_injected_substeps"][7, 1] = 9
    trace["residual_pd_target_injected_substeps"][8, 1] = 9
    trace["residual_pd_target_override_substeps"][8, 1] = 1
    trials = [
        {"agent_id": "red.finisher", "censored": False, "start_frame": 1, "frames": 100},
        {"agent_id": "red.finisher", "censored": True, "start_frame": 101, "frames": 49},
    ]
    mask = learning_mask(trace, agent_ids=ids, agent_id="red.finisher", trials=trials)
    assert not mask[:, 0].any()
    assert mask[:, 1].sum() == 96
    assert not mask[101:, 1].any()
    trace["residual_pd_target_injected_substeps"][9, 1] = 11
    with pytest.raises(ValueError):
        learning_mask(trace, agent_ids=ids, agent_id="red.finisher", trials=trials)


def test_exact_array_comparison_rejects_changed_and_missing_data():
    a = {"x": np.array([1.0, np.nan]), "name": np.array(["g1"])}
    assert same_arrays(a, {"x": np.array([1.0, np.nan]), "name": np.array(["g1"])})
    assert not same_arrays(a, {"x": np.array([2.0, np.nan]), "name": np.array(["g1"])})
    assert not same_arrays(a, {"x": np.array([1.0, np.nan])})
