"""Synthetic collector edge cases, not physical or learning evidence."""

from types import SimpleNamespace

import numpy as np
import torch
from torch import nn

from rosclaw_soccer.rsi import contact_option_rollout as module


class Foundation(nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = nn.Parameter(torch.zeros(1), requires_grad=False)
        self.rows = 0

    def forward(self, obs):
        assert not obs[:, 98:].any()
        self.rows += len(obs)
        return torch.zeros((len(obs), 29))


def fixture():
    q = np.zeros((24, 2, 10, 43))
    q[:, :, :, 9] = 0.8
    q[:, :, :, 10] = 1
    q[:, :, :, 3] = 1
    q[:, 0, :, 0] = np.arange(240).reshape(24, 10) * 0.002
    valid = np.ones((24, 2), dtype=bool)
    valid[1:, 1] = False
    terminated = np.zeros_like(valid)
    terminated[0, 1] = True
    force = np.zeros((24, 2, 10, 3))
    force[0, 0, 0, 0] = 20
    head = dict(
        valid=valid, terminated=terminated, qpos=q, qvel=np.zeros((24, 2, 10, 41)), forces=force
    )
    env = SimpleNamespace(
        worlds=2,
        index=np.arange(2),
        layout=SimpleNamespace(rq=7, qi=np.arange(14, 43), bq=0, rv=6, vi=np.arange(12, 41), bv=0),
        bank=dict(
            goal_world=np.array([[5.0, 0.0, 0.0], [5.0, 0.0, 0.0]]),
            expected_foot=np.zeros(2, dtype=int),
        ),
    )
    return env, head


class Tail:
    def __init__(self, env, *, contact_failed):
        self.ended = contact_failed.copy()
        self.failed = contact_failed.copy()
        self.elapsed = np.zeros(2, dtype=int)
        self.observation = np.zeros((2, 177), dtype=np.float32)
        self.count = 0

    @property
    def active_ids(self):
        return np.flatnonzero(~self.ended)

    def step(self, action):
        assert not action[1].any()
        q = np.zeros((2, 10, 43))
        q[:, :, 9] = 0.8
        q[:, :, 10] = 1
        q[:, :, 3] = 1
        q[0, :, 0] = (240 + self.count * 10 + np.arange(10)) * 0.002
        self.count += 1
        self.elapsed[0] += 1
        completed = np.array([self.count == 60, False])
        self.ended |= completed
        return dict(
            valid=np.array([True, False]),
            completed=completed,
            terminated=np.zeros(2, dtype=bool),
            qpos=q,
            qvel=np.zeros((2, 10, 41)),
            forces=np.zeros((2, 10, 3)),
        )


def test_measured_tail_reward_once_and_failed_world_never_inferred(monkeypatch):
    monkeypatch.setattr(module, "NativeRecoveryTail", Tail)
    env, head = fixture()
    model = Foundation()
    raw, outcomes, rows = module.finish_contact_option(env, head, model)
    assert rows == model.rows == 60
    assert not raw["reward"][:-1].any()
    assert raw["reward"][-1, 0] == 12
    assert not raw["reward"][:, 1].any()
    assert outcomes[0]["recovery_controls"] == 60
    assert outcomes[1]["contact_failed"] and outcomes[1]["recovery_controls"] == 0


def test_all_failed_head_produces_only_invalid_padding_not_fake_states(monkeypatch):
    monkeypatch.setattr(module, "NativeRecoveryTail", Tail)
    env, head = fixture()
    head["valid"][1:] = False
    head["terminated"][0] = True
    model = Foundation()
    raw, outcomes, rows = module.finish_contact_option(env, head, model)
    assert rows == model.rows == 0
    assert not raw["valid"].any() and not raw["reward"].any() and not raw["terminal"].any()
    assert "qpos" not in raw
    assert all(r["contact_failed"] for r in outcomes)
