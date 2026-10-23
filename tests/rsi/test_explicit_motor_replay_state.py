import pytest
import torch
from test_dual_recurrent_motor import model

from rosclaw_soccer.rsi.consolidated_motor_adapter import ConsolidatedMotorAdapter
from rosclaw_soccer.rsi.paid_run_teacher_audit import private_replay_state


@pytest.mark.parametrize("adapter", [False, True])
def test_replay_allocates_own_zero_context_for_each_explicit_frozen_cpu_motor(adapter):
    _, actor = model()
    if adapter:
        actor = ConsolidatedMotorAdapter(actor)
    actor.eval().requires_grad_(False)
    h, c = private_replay_state(actor, 256)
    other = private_replay_state(actor, 256)
    assert h.shape == c.shape == (2, 1, 256)
    assert not h.any() and not c.any()
    assert h.data_ptr() != c.data_ptr() != other[1].data_ptr()
    assert h.data_ptr() != other[0].data_ptr()


def test_replay_does_not_silently_admit_trainable_or_duck_typed_motor():
    _, motor = model()
    with pytest.raises(ValueError):
        private_replay_state(motor, 256)

    class Foreign:
        def initial_state(self):
            raise AssertionError("must reject before asking foreign actor")

    with pytest.raises(ValueError):
        private_replay_state(Foreign(), 256)

    class ForeignSubclass(ConsolidatedMotorAdapter):
        pass

    fake = ForeignSubclass(motor).eval().requires_grad_(False)
    with pytest.raises(ValueError):
        private_replay_state(fake, 256)


@pytest.mark.parametrize("width", [True, 256.0, 0, 129, 512])
def test_unknown_memory_contract_rejected(width):
    with pytest.raises(ValueError):
        private_replay_state(None, width)


def test_legacy128_context_is_still_private_and_unpadded():
    h, c = private_replay_state(None, 128)
    assert h.shape == c.shape == (2, 1, 128)
    assert torch.equal(h, torch.zeros_like(h))
