from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.rsi.paid_causal_context import CausalMotorContext


def context(ep, **updates):
    values = dict(
        episode_key=id(ep),
        frame=ep.frame,
        observation=np.zeros(160, np.float32),
        **{
            k: np.zeros((2, 1, 128), np.float32)
            for k in ("h_before", "c_before", "h_after", "c_after")
        },
    )
    values.update(updates)
    return CausalMotorContext(**values)


def test_context_is_not_an_alias_to_actor_memory_and_cannot_enable_numpy_writes():
    ep = SimpleNamespace(frame=0)
    before = np.ones((2, 1, 128), np.float32)
    c = context(ep, h_before=before)
    before.fill(42)
    assert np.all(c.h_before == 1)
    with pytest.raises(ValueError):
        c.h_before.setflags(write=True)
    with pytest.raises(ValueError):
        c.observation[0] = 1
    c.assert_current(ep)


def test_foreign_player_and_stale_control_cannot_reuse_context():
    ep = SimpleNamespace(frame=0)
    c = context(ep)
    with pytest.raises(ValueError):
        c.assert_current(SimpleNamespace(frame=0))
    ep.frame = 1
    with pytest.raises(ValueError):
        c.assert_current(ep)


@pytest.mark.parametrize(
    "updates",
    [
        dict(frame=True),
        dict(frame=300),
        dict(episode_key=0),
        dict(observation=np.zeros(159, np.float32)),
        dict(h_after=np.zeros((2, 1, 128), np.float64)),
        dict(c_before=np.full((2, 1, 128), np.nan, np.float32)),
    ],
)
def test_invalid_context_rejected(updates):
    with pytest.raises(ValueError):
        context(SimpleNamespace(frame=0), **updates)
