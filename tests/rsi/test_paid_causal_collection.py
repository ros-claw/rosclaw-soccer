from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_research_bench import PaidResearchBench


class Actor:
    def __call__(self, obs, state):
        return torch.zeros(1, 1, 29), (state[0] + 0.01, state[1] + 0.02)


class Episode:
    def __init__(self):
        self.frame = 0
        self.done = False
        self.last_action = np.zeros(29, np.float32)
        self.clip = SimpleNamespace(length=3, name="mock_left", path=Path("mock_left.npz"))
        self.goal = np.array([6.0, 0, 0.11])
        self.initial = {"qpos": np.zeros(43)}
        self.actions = []
        data = SimpleNamespace(
            time=0.0,
            qpos=np.zeros(43),
            qvel=np.zeros(41),
            act=np.zeros(0),
            ctrl=np.zeros(29),
            qacc_warmstart=np.zeros(41),
            mocap_pos=np.zeros((1, 3)),
            mocap_quat=np.array([[1.0, 0, 0, 0]]),
        )
        self.env = SimpleNamespace(data=data)

    def observation(self):
        x = np.zeros(160, np.float32)
        x[0] = self.frame
        return x

    def step(self, action):
        self.actions.append(action.copy())
        self.frame += 1
        self.done = self.frame == 3
        self.env.data.time += 0.02
        q = np.zeros((10, 43))
        q[:, 2] = 0.8
        q[:, 3] = 1
        q[:, 38] = 0.11
        return dict(qpos=q, qvel=np.zeros((10, 41)), forces=np.zeros((10, 3)), body_failed=False)


def bench():
    b = PaidResearchBench.__new__(PaidResearchBench)
    b.lock = Lock()
    b.neural_rows = 0
    return b


def test_passthrough_context_keeps_all_legacy_neural_and_native_arrays_bit_exact():
    b = bench()
    old, summary = b.collect_current(Episode(), Actor())
    seen = []

    def teacher(ep, nominal, c):
        c.assert_current(ep)
        assert c.observation[0] == ep.frame
        assert np.allclose(c.h_after, c.h_before + 0.01)
        seen.append(c)
        return nominal, dict(context_frame=c.frame)

    new, s = b.collect_current(Episode(), Actor(), causal_transform=teacher)
    for k, v in old.items():
        np.testing.assert_array_equal(v, new[k])
    assert b.neural_rows == 6 and len(seen) == 3
    assert s["explicit_own_causal_teacher_context"] is True
    assert s["composite_teacher"] and not s["policy_likelihood_provided"]
    assert not summary["composite_teacher"]


def test_causal_callback_cannot_rewrite_logged_actor_proposal():
    def teacher(ep, nominal, c):
        nominal[0] = 0.1
        return nominal, dict(context_frame=c.frame)

    z, _ = bench().collect_current(Episode(), Actor(), causal_transform=teacher)
    assert not z["NN_nominal_action"].any()
    assert np.all(z["NN_action"][:, 0] == np.float32(0.1))


@pytest.mark.parametrize("fault", ["physical", "dtype", "nan", "two_transforms", "transition"])
def test_bad_causal_transform_fails_before_any_native_step(fault):
    ep = Episode()

    def teacher(ep, nominal, c):
        if fault == "physical":
            ep.env.data.qpos[0] = 1
        elif fault == "dtype":
            nominal = nominal.astype(np.float64)
        elif fault == "nan":
            nominal[0] = np.nan
        return nominal, {}

    kwargs = dict(causal_transform=teacher)
    if fault == "two_transforms":
        kwargs["transform"] = lambda ep, action: (action, {})
    elif fault == "transition":
        kwargs["transition_controls"] = 12
    with pytest.raises(ValueError):
        bench().collect_current(ep, Actor(), **kwargs)
    assert not ep.actions
