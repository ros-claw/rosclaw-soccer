import math

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.motor_sampling_law import history_motor_logp
from rosclaw_soccer.rsi.receiver_exploration_stream import ReceiverExplorationStream


def test_seeded_samples_match_and_feedback_is_mandatory():
    scale = np.full(29, 0.25, np.float32)
    mean = np.zeros(29, np.float32)
    a, b = ReceiverExplorationStream(scale, seed=12), ReceiverExplorationStream(scale, seed=12)
    for t in range(3):
        x, trace_x = a.propose(mean, mean, control=t)
        y, trace_y = b.propose(mean, mean, control=t)
        assert np.array_equal(x, y)
        assert np.array_equal(trace_x["rng_after"], trace_y["rng_after"])
        assert np.max(np.abs(x * scale)) < 0.2
        a.commit(x)
        b.commit(y)
    x, _ = a.propose(mean, mean, control=3)
    with pytest.raises(ValueError):
        a.commit(x + np.float32(0.01))
    with pytest.raises(ValueError, match="latched"):
        a.propose(mean, mean, control=4)


def test_double_proposal_is_terminal():
    scale = np.full(29, 0.25, np.float32)
    mean = np.zeros(29, np.float32)
    stream = ReceiverExplorationStream(scale, seed=2)
    action, _ = stream.propose(mean, mean, control=0)
    with pytest.raises(ValueError):
        stream.propose(mean, mean, control=0)
    with pytest.raises(ValueError, match="latched"):
        stream.commit(action)


def test_sampling_log_density_matches_complete_executed_history():
    scale = np.full(29, 0.25, np.float32)
    means = np.zeros((6, 29), np.float32)
    stream = ReceiverExplorationStream(scale, seed=18)
    actions, logps = [], []
    for t in range(len(means)):
        action, trace = stream.propose(means[t], means[t], control=t)
        stream.commit(action)
        actions.append(action)
        logps.append(trace["logp"])
    density = history_motor_logp(
        "box-truncated-ar1",
        torch.from_numpy(means),
        torch.from_numpy(np.stack(actions)),
        torch.from_numpy(means),
        torch.from_numpy(scale),
        torch.full((29,), math.log(0.02), dtype=torch.float32),
        rho=0.9,
    )
    assert np.allclose(density.numpy(), np.stack(logps), atol=1e-4, rtol=0)


def test_sampling_density_tracks_nonzero_learned_actor_mean():
    scale = np.full(29, 0.25, np.float32)
    reference = np.zeros((6, 29), np.float32)
    means = np.stack([np.full(29, 0.08 + 0.01 * t, np.float32) for t in range(6)])
    stream = ReceiverExplorationStream(scale, seed=92117)
    actions, logps = [], []
    for t, mean in enumerate(means):
        action, trace = stream.propose(mean, reference[t], control=t)
        stream.commit(action)
        actions.append(action)
        logps.append(trace["logp"])
        assert np.max(np.abs((action - reference[t]) * scale)) < 0.2
    density = history_motor_logp(
        "box-truncated-ar1",
        torch.from_numpy(means),
        torch.from_numpy(np.stack(actions)),
        torch.from_numpy(reference),
        torch.from_numpy(scale),
        torch.full((29,), math.log(0.02), dtype=torch.float32),
        rho=0.9,
    )
    assert np.allclose(density.numpy(), np.stack(logps), atol=1e-4, rtol=0)
