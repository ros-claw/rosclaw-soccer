from types import SimpleNamespace

import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_skill_handoff import snapshot
from rosclaw_soccer.rsi.paid_velocity_shadow import (
    AppliedVelocityControl,
    OwnVelocityShadow,
    policy_action_from_hg_target,
)


class ToyActor(torch.nn.Module):
    def forward(self, obs, state):
        return torch.full((1, 1, 29), 0.9), (state[0] + 1, state[1] + 1)


class ToyEpisode:
    def __init__(self):
        d = SimpleNamespace(
            qpos=np.zeros(43),
            qvel=np.zeros(41),
            ctrl=np.zeros(29),
            qacc_warmstart=np.zeros(41),
            mocap_pos=np.zeros((1, 3)),
            mocap_quat=np.ones((1, 4)),
            qfrc_applied=np.zeros(41),
            xfrc_applied=np.zeros((1, 6)),
            act=np.zeros(0),
            time=0.0,
        )
        d.qpos[2:4] = [0.8, 1]
        self.env = SimpleNamespace(data=d)
        self.work = SimpleNamespace(native_steps=0)
        self.offset = np.zeros(29)
        self.scale = np.full(29, 0.5)
        self.order = np.arange(29)[::-1].copy()
        self.done = False
        self.frame = 0
        self.ready = False
        self.last_action = np.zeros(29, np.float32)
        self.clip = SimpleNamespace(
            joint_pos=np.full((289, 29), 0.1, np.float32),
            joint_vel=np.zeros((289, 29), np.float32),
            body_ang_vel_w=np.zeros((289, 1, 3), np.float32),
        )
        self.torso_index = 0

    def observation(self):
        self.ready = True
        obs = np.zeros(160, np.float32)
        obs[125:154] = self.last_action
        return obs


def synthetic_control(ep, target):
    # Unit-only boundary evidence; no claim that this is native simulation.
    before, clock = ep.work.native_steps, ep.env.data.time
    ep.work.native_steps += 10
    ep.env.data.time += 0.02
    ep.env.data.qpos[0] += 0.001
    return AppliedVelocityControl(id(ep), before, clock, target, snapshot(ep.env.data), {})


def test_target_conversion_is_policy_ordered_actual_command_not_shadow_proposal():
    ep = ToyEpisode()
    target = np.linspace(0.01, 0.29, 29)
    action = policy_action_from_hg_target(target, ep.offset, ep.scale, ep.order)
    assert np.allclose(ep.offset + action[ep.order] * ep.scale, target, atol=3e-7, rtol=0)
    assert not np.array_equal(action, np.full(29, 0.9, np.float32))


def test_full_shadow_has_real_control_boundaries_and_own_history_no_reset():
    ep = ToyEpisode()
    shadow = OwnVelocityShadow(ep, ToyActor().eval())
    target = np.linspace(0.01, 0.29, 29)
    for i in range(80):
        context, proposal = shadow.observe()
        if i:
            assert np.array_equal(context.observation[125:154], ep.last_action)
        control = synthetic_control(ep, target)
        before = snapshot(ep.env.data)
        shadow.commit_velocity(control)
        assert all(np.array_equal(v, snapshot(ep.env.data)[k]) for k, v in before.items())
        assert not shadow.records[-1]["proposal_executed"]
        assert np.array_equal(shadow.records[-1]["R2_shadow_proposal"], proposal)
    h, c = shadow.suffix_state()
    assert torch.equal(h, torch.full((2, 1, 128), 80.0)) and torch.equal(c, h)
    assert ep.frame == 80 and ep.work.native_steps == 800
    assert abs(ep.env.data.time - 1.6) < 1e-12


def test_missing_native_control_and_foreign_or_stale_evidence_rejected():
    ep = ToyEpisode()
    shadow = OwnVelocityShadow(ep, ToyActor().eval())
    shadow.observe()
    with pytest.raises(ValueError):
        shadow.suffix_state()
    with pytest.raises(ValueError):
        shadow.observe()
    foreign = synthetic_control(ToyEpisode(), np.zeros(29))
    with pytest.raises(ValueError):
        shadow.commit_velocity(foreign)
    own = synthetic_control(ep, np.zeros(29))
    ep.env.data.qacc_warmstart[0] = 1
    with pytest.raises(ValueError):
        shadow.commit_velocity(own)
    assert ep.frame == 0
