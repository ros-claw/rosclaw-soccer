import numpy as np
import pytest

from rosclaw_soccer.rsi.receiver_exploration_stream import ReceiverExplorationStream
from rosclaw_soccer.rsi.return_player_sampling_audit import audit_return_player_sampling


def test_full_history_density_and_faults():
    scale = np.ones(29, np.float32)
    sampler = ReceiverExplorationStream(scale, seed=41)
    mean, reference = np.zeros(29, np.float32), np.zeros(29, np.float32)
    rows = []
    for control in range(3):
        action, witness = sampler.propose(mean, reference, control=control)
        sampler.commit(action)
        rows.append((action, witness["logp"]))
    trace = {
        "sampling_active": np.array([False, True, True, True, False]),
        "sampling_mean": np.zeros((5, 29), np.float32),
        "sampling_reference": np.zeros((5, 29), np.float32),
        "sampling_action": np.stack([mean, *(r[0] for r in rows), mean]),
        "sampling_logp": np.array([0, *(r[1] for r in rows), 0], np.float32),
    }
    assert audit_return_player_sampling(trace, scale)["samples"] == 3
    trace["sampling_logp"][2] += 1
    with pytest.raises(ValueError, match="density mismatch"):
        audit_return_player_sampling(trace, scale)
    trace["sampling_active"][2] = False
    with pytest.raises(ValueError, match="consecutive"):
        audit_return_player_sampling(trace, scale)


def test_empty_window_is_not_training_data():
    assert audit_return_player_sampling(
        {"sampling_active": np.zeros(5, bool)}, np.ones(29, np.float32)
    ) == {
        "samples": 0,
        "max_logp_error": 0.0,
    }
