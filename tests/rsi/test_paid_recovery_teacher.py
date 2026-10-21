from copy import deepcopy

import numpy as np
import pytest
import torch
from torch import nn

from rosclaw_soccer.rsi.paid_practice_chain import run_practice_chain
from rosclaw_soccer.rsi.paid_recovery_teacher import RecoveryMixtureTeacher, contact_weight


class TinyActor(nn.Module):
    def __init__(self, base, proposal, *, mismatch=False):
        super().__init__()
        self.base = base
        self.register_buffer("action_scale", torch.ones(29))
        self.proposal, self.mismatch, self.calls = proposal, mismatch, 0

    def forward(self, obs, state):
        self.calls += 1
        return torch.full((1, 1, 29), self.proposal), tuple(
            t + (2 if self.mismatch else 1) for t in state
        )


def actors():
    a = nn.Linear(1, 1)
    return TinyActor(a, 0.1), TinyActor(deepcopy(a), -0.1)


@pytest.mark.parametrize("cutover", [140, 160, 190])
def test_smooth_weight_endpoints_monotone_and_bounded(cutover):
    weights = [contact_weight(i, cutover) for i in range(300)]
    assert weights[cutover - 12] == 1
    assert weights[cutover + 12] == 0
    assert weights[cutover] == 0.5
    assert all(0 <= x <= 1 for x in weights)
    assert all(x >= y for x, y in zip(weights, weights[1:], strict=False))


@pytest.mark.parametrize("step,cutover", [(-1, 140), (True, 140), (0, 150), (0, True)])
def test_undeclared_schedule_rejected(step, cutover):
    with pytest.raises(ValueError):
        contact_weight(step, cutover)


def test_both_own_actors_execute_on_actual_same_input_and_memory():
    contact, recovery = actors()
    teacher = RecoveryMixtureTeacher(contact, recovery, cutover=140)
    obs = torch.zeros(1, 1, 160)
    state = (torch.zeros(2, 1, 128), torch.zeros(2, 1, 128))
    for i in range(155):
        before = tuple(t.clone() for t in state)
        action, state = teacher(obs, state)
        assert all(torch.equal(t, b + 1) for t, b in zip(state, before, strict=True))
        if i == 0:
            assert torch.equal(action, torch.full_like(action, 0.1))
        if i == 154:
            assert torch.equal(action, torch.full_like(action, -0.1))
    assert contact.calls == recovery.calls == 155
    assert all(not p.requires_grad for p in teacher.parameters())
    trace = teacher.trace()
    assert np.array_equal(trace["TEACHER_step"], np.arange(155))
    assert trace["TEACHER_contact_proposal"].shape == (155, 29)
    assert trace["TEACHER_recovery_proposal"].shape == (155, 29)
    teacher.begin_skill()
    assert teacher.rows == []
    with pytest.raises(ValueError, match="execute"):
        teacher.trace()


def test_hidden_memory_mismatch_not_silently_blended():
    contact, recovery = actors()
    recovery.mismatch = True
    teacher = RecoveryMixtureTeacher(contact, recovery, cutover=160)
    with pytest.raises(ValueError, match="shared-memory"):
        teacher(torch.zeros(1, 1, 160), (torch.zeros(2, 1, 128),) * 2)


def test_different_base_or_scale_rejected_before_physics():
    contact, recovery = actors()
    recovery.action_scale[0] += 0.1
    with pytest.raises(ValueError, match="shared frozen base"):
        RecoveryMixtureTeacher(contact, recovery, cutover=160)


def test_teacher_cannot_enter_pure_neural_chain_without_explicit_authority():
    contact, recovery = actors()
    teacher = RecoveryMixtureTeacher(contact, recovery, cutover=160)
    with pytest.raises(ValueError, match="separate explicit diagnostic"):
        run_practice_chain(None, {}, [contact.base, teacher], store=lambda *_: None)
