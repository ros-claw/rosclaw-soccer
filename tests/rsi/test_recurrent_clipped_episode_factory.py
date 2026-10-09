"""Factory ownership and original numerical parity, NOT physical qualification."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.recurrent_clipped_episode_factory import (
    RecurrentClippedEpisodeFactory,
    validate_compilation_contract,
)
from rosclaw_soccer.rsi.recurrent_clipped_motor import CompiledRecurrentClippedMotor
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_clipped_motor import learned  # noqa: F401
from tests.rsi.test_recurrent_sampling_motor import long_body
from tests.rsi.test_recurrent_success_motor import boundary
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


def test_full300_original_parity_and_independent_contact_state(learned):  # noqa: F811
    artifact = learned[-1]
    snapshot = copy.deepcopy(artifact)
    factory = RecurrentClippedEpisodeFactory(artifact)
    policy = factory.preview(artifact)
    a, b = factory.new_episode(), factory.new_episode()
    original = CompiledRecurrentClippedMotor(policy)
    trace, previous = long_body(), np.zeros(12)
    for frame in range(300):
        step = boundary(frame, previous)
        delta = a.delta_at_frame(policy, trace, **step)
        np.testing.assert_array_equal(delta, original.delta_at_frame(policy, trace, **step))
        np.testing.assert_array_equal(a.hidden_state, original.hidden_state)
        previous = delta
    assert type(a) is CompiledRecurrentClippedMotor
    assert b._active_frame is None
    assert factory._prototype._active_frame is None
    np.testing.assert_array_equal(b.hidden_state, np.zeros(64))
    np.testing.assert_array_equal(factory.new_episode().hidden_state, np.zeros(64))
    for obj in (a._memory, a._behavior._memory, a._parent._memory):
        assert (
            obj is not b._memory and obj is not b._behavior._memory and obj is not b._parent._memory
        )
    assert a._parent._warm is not b._parent._warm
    assert a._recurrent is not b._recurrent
    assert a._parent is a._behavior._parent
    assert artifact == snapshot
    validate_compilation_contract(factory.contract(), policy)
    for flag in ("runtime_execution_authorized", "promotion_authorized", "hardware_authorized"):
        assert factory.contract()[flag] is False


def test_caller_model_contract_mutation_and_changed_critic_refused(learned):  # noqa: F811
    artifact = copy.deepcopy(learned[-1])
    factory = RecurrentClippedEpisodeFactory(artifact)
    contract = factory.contract()
    contract["source_pins"].clear()
    assert factory.contract()["source_pins"]
    policy = factory.preview(artifact)
    weights = artifact["critic_parameters"]
    weights[next(iter(weights))][0][0] += 0.001
    with pytest.raises(ValueError, match="canonical"):
        factory.preview(artifact)
    assert policy["step_motor_proof"]["model"] != artifact
    contract = factory.contract()
    contract["hardware_authorized"] = 0
    with pytest.raises(ValueError, match="contract"):
        validate_compilation_contract(contract, policy)


@pytest.mark.parametrize("mutation", ["state", "policy", "source", "snapshot"])
def test_cached_prototype_source_or_policy_drift_refused(learned, mutation):  # noqa: F811
    factory = RecurrentClippedEpisodeFactory(learned[-1])
    if mutation == "state":
        factory._prototype._recurrent._state[0] = 0.1
    elif mutation == "policy":
        factory._prototype._policy_hash = "sha256:" + "0" * 64
    elif mutation == "source":
        factory._pins[next(iter(factory._pins))] = "sha256:" + "0" * 64
    else:
        object.__setattr__(factory._policy_snapshot, "_data", b"{}")
    for action in (factory.new_episode, factory.contract, lambda: factory.policy_hash):
        with pytest.raises(ValueError):
            action()
