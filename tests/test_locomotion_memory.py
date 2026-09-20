from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.locomotion_memory import (
    LocomotionMemory,
    capture_locomotion_memory,
    restore_private_locomotion_memory,
)

torch = pytest.importorskip("torch")
POLICY = "a" * 64


def model():
    return SimpleNamespace(
        hidden_state=torch.arange(256, dtype=torch.float32).reshape(1, 1, 256),
        cell_state=torch.ones(1, 1, 256),
    )


def test_snapshot_has_no_alias_and_restore_is_exact():
    live = model()
    snapshot = capture_locomotion_memory(live, policy_hash=POLICY)
    before = snapshot.state_hash
    live.hidden_state.zero_()
    private = model()
    private.cell_state.zero_()
    restore_private_locomotion_memory(private, snapshot, policy_hash=POLICY, private_replay=True)
    assert torch.equal(private.hidden_state, torch.arange(256).reshape(1, 1, 256))
    assert torch.equal(private.cell_state, torch.ones(1, 1, 256))
    assert live.hidden_state.count_nonzero() == 0
    assert capture_locomotion_memory(private, policy_hash=POLICY) == snapshot
    assert snapshot.state_hash == before


@pytest.mark.parametrize("owner,policy", [(False, POLICY), (1, POLICY), (True, "b" * 64)])
def test_rejected_restore_does_not_mutate(owner, policy):
    private = model()
    snapshot = capture_locomotion_memory(private, policy_hash=POLICY)
    with pytest.raises(ValueError):
        restore_private_locomotion_memory(
            private, snapshot, policy_hash=policy, private_replay=owner
        )
    assert capture_locomotion_memory(private, policy_hash=POLICY) == snapshot


@pytest.mark.parametrize("bad", [b"", bytearray(1024), np.full(256, np.nan, dtype="<f4").tobytes()])
def test_invalid_payload(bad):
    with pytest.raises(ValueError):
        LocomotionMemory(POLICY, bytes(1024), bad)


def test_invalid_second_state_does_not_mutate_first():
    private = model()
    snapshot = capture_locomotion_memory(private, policy_hash=POLICY)
    private.hidden_state.zero_()
    private.cell_state = torch.zeros(1, 1, 255)
    with pytest.raises(ValueError):
        restore_private_locomotion_memory(
            private, snapshot, policy_hash=POLICY, private_replay=True
        )
    assert private.hidden_state.count_nonzero() == 0


def test_alias_is_rejected():
    private = model()
    private.cell_state = private.hidden_state
    with pytest.raises(ValueError):
        capture_locomotion_memory(private, policy_hash=POLICY)
