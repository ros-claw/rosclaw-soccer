"""Synthetic CPU session observation/call accounting; no G1 physical result."""

import numpy as np
import pytest
from test_shared_nn_forecast import setup

from rosclaw_soccer.rsi.shared_walk_kick_forecast import _KickCalls


def test_inference_observer_never_changes_original_output_and_caps_calls(tmp_path):
    player, _ = setup(tmp_path)
    session = player.policy.ort_session
    observer = _KickCalls(session, 2)
    obs = np.linspace(-0.1, 0.1, 547, dtype=np.float32)[None]
    expected = session.run(["actions"], {"obs": obs})[0]
    for _ in range(2):
        result = observer.run(["actions"], {"obs": obs})[0]
        np.testing.assert_array_equal(expected, result)
    np.testing.assert_array_equal(observer.observations[0], obs[0])
    assert observer.attempted == len(observer.actions) == 2
    with pytest.raises(RuntimeError, match="budget"):
        observer.run(["actions"], {"obs": obs})
    assert observer.attempted == len(observer.observations) == 2


@pytest.mark.parametrize("bad", ["shape", "dtype", "nan", "name"])
def test_malformed_input_rejected_before_extra_nn_or_physics(tmp_path, bad):
    player, _ = setup(tmp_path)
    observer = _KickCalls(player.policy.ort_session, 1)
    obs = np.zeros((1, 547), np.float32)
    if bad == "shape":
        obs = obs[:, :10]
    elif bad == "dtype":
        obs = obs.astype(np.float64)
    elif bad == "nan":
        obs[0, 0] = np.nan
    with pytest.raises(ValueError):
        observer.run(["wrong"] if bad == "name" else ["actions"], {"obs": obs})
    assert observer.attempted == len(observer.actions) == 0
