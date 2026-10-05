"""Owned sequence factory contracts, not actual native physics evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.recurrent_success_episode_factory import (
    RecurrentSuccessEpisodeFactory,
    validate_compilation_contract,
)
from rosclaw_soccer.rsi.recurrent_success_motor import CompiledRecurrentSuccessMotor, make_model
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_imitation_proposal_motor import imitation_parent  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_recurrent_success_motor import boundary, fitted_synthetic_model
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_execution import body
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.mark.parametrize("trained", [False, True])
def test_complete_original_parity_and_independent_episode_state(imitation_parent, trained):  # noqa: F811
    artifact = fitted_synthetic_model(imitation_parent) if trained else make_model(imitation_parent)
    snapshot = copy.deepcopy(artifact)
    factory = RecurrentSuccessEpisodeFactory(artifact)
    policy = factory.preview(artifact)
    a, b = factory.new_episode(), factory.new_episode()
    original = CompiledRecurrentSuccessMotor(policy)
    previous = np.zeros(12)
    observation = body()
    for frame in range(36):
        step = boundary(frame, previous)
        actual = a.delta_at_frame(policy, observation, **step)
        expected = original.delta_at_frame(policy, observation, **step)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(a.hidden_state, original.hidden_state)
        previous = actual
    np.testing.assert_array_equal(b.hidden_state, np.zeros(64))
    np.testing.assert_array_equal(factory._prototype.hidden_state, np.zeros(64))
    assert a._memory is not b._memory
    assert a._parent._memory is not b._parent._memory
    assert a._parent._warm is not b._parent._warm
    assert a._behavior._layers is not b._behavior._layers
    assert a._behavior._layers[0][0] is b._behavior._layers[0][0]
    assert a._recurrent is not b._recurrent
    with pytest.raises(ValueError):
        a._behavior._layers[0][0][0, 0] = 1
    validate_compilation_contract(factory.contract(), policy)
    contract = factory.contract()
    contract["hardware_authorized"] = 0
    with pytest.raises(ValueError):
        validate_compilation_contract(contract, policy)
    artifact["parameters"]["head_bias"][0] += 1
    assert policy["step_motor_proof"]["model"] == snapshot
    with pytest.raises(ValueError, match="canonical"):
        factory.preview(artifact)


def test_changed_factory_source_rejected(imitation_parent, monkeypatch):  # noqa: F811
    from rosclaw_soccer.rsi import recurrent_success_episode_factory as module

    factory = RecurrentSuccessEpisodeFactory(make_model(imitation_parent))
    monkeypatch.setattr(module, "hash_bytes", lambda _: "sha256:" + "f" * 64)
    for action in (factory.new_episode, factory.contract, lambda: factory.policy_hash):
        with pytest.raises(ValueError, match="sources changed"):
            action()


@pytest.mark.parametrize(
    "options,foreign,other",
    [
        (["--step-model", "missing.json"], True, None),
        ([], False, None),
        (["--step-model", "missing.json", "--proposal-decoder", "owned_snapshot"], False, None),
        (["--step-model", "missing.json", "--cached-proposal-envelope"], False, None),
        (["--step-model", "missing.json", "--body-response-bundle", "missing.json"], False, None),
        (["--step-model", "missing.json", "--foundation-only"], False, None),
        (["--step-model", "missing.json"], False, "sampling_factory"),
        (["--step-model", "missing.json"], False, "proposal_factory"),
        (["--step-model", "missing.json"], False, "extended_factory"),
        (["--step-model", "missing.json"], False, "imitation_factory"),
    ],
)
def test_foreign_or_mixed_native_factories_rejected(tmp_path, options, foreign, other):
    from scripts.rsi_mujoco_motor_transfer import main

    factory = object() if foreign else object.__new__(RecurrentSuccessEpisodeFactory)
    output = tmp_path / "must-not-exist"
    args = [
        "--scene",
        str(tmp_path / "absent.xml"),
        "--model-root",
        str(tmp_path / "absent"),
        "--late-swing-policy",
        str(tmp_path / "absent.json"),
        "--seed",
        "0",
        "--lane",
        "0",
        "--output-root",
        str(output),
        *options,
    ]
    with pytest.raises(SystemExit) as error:
        main(args, recurrent_factory=factory, **({other: object()} if other else {}))
    assert error.value.code == 2
    assert not output.exists()
