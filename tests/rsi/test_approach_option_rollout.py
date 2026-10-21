"""Synthetic collector test: all-failed heads must not invent recovery data."""

from types import SimpleNamespace

import numpy as np
import torch

from rosclaw_soccer.rsi import approach_option_rollout as module


def test_failed_head_has_no_recovery_inference_physics_or_fake_samples(monkeypatch):
    class Tail:
        def __init__(self, env):
            self.active_ids = np.array([], dtype=np.int64)
            self.failed = np.array([True])
            self.elapsed = np.array([0])

    monkeypatch.setattr(module, "NativeApproachRecoveryTail", Tail)
    env = SimpleNamespace(
        worlds=1,
        index=np.array([0]),
        head_failed=np.array([True]),
        elapsed=np.array([1]),
        layout=SimpleNamespace(
            rq=0, rv=0, bq=36, bv=35, qi=list(range(7, 36)), vi=list(range(6, 35))
        ),
        bank=dict(goal_world=np.array([[10.0, 0, 0]]), expected_foot=np.array([0])),
    )
    q = np.zeros((1, 1, 10, 43))
    q[..., 2], q[..., 3] = 0.8, 1.0
    head = dict(
        valid=np.ones((1, 1), dtype=bool),
        qpos=q,
        qvel=np.zeros((1, 1, 10, 41)),
        forces=np.zeros((1, 1, 10, 3)),
    )
    counters, trace = {}, []
    raw, outcomes = module.finish_approach_option(
        env, head, torch.nn.Linear(177, 29), counters=counters, trace=trace
    )
    assert not trace and not counters
    assert not raw["valid"].any() and not raw["reward"].any() and not raw["terminal"].any()
    assert "qpos" not in raw and "observation" not in raw
    assert outcomes[0]["contact_failed"] and outcomes[0]["recovery_controls"] == 0
    assert not outcomes[0]["recovery_failed"] and outcomes[0]["recovery_reward"] == 0
