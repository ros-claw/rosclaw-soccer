"""Synthetic learning/compilation contracts, not successful football evidence."""

import copy

import numpy as np
import pytest
from rosclaw.growth.bounded_residual_imitation import (
    BoundedResidualImitationConfig,
    fit_bounded_residual_imitation,
)

from rosclaw_soccer.rsi.imitation_proposal_motor import (
    CompiledImitationProposalMotor,
    make_model,
    make_preview,
    validate_model,
)
from rosclaw_soccer.rsi.imitation_proposal_snapshot_compilation import compile_imitation_snapshot
from rosclaw_soccer.rsi.proposal_decoder_selection import select_proposal_decoder
from rosclaw_soccer.rsi.proposal_memory_motor import initial_model
from rosclaw_soccer.rsi.proposal_memory_motor import make_preview as parent_preview
from rosclaw_soccer.sim.contracts import hash_json
from tests.rsi.test_current_memory_motor import current  # noqa: F401
from tests.rsi.test_kernel_guarded_step_motor import candidate  # noqa: F401
from tests.rsi.test_smooth_memory_motor import smooth_parent  # noqa: F401
from tests.rsi.test_step_motor_network import model  # noqa: F401


@pytest.fixture
def imitation_parent(request):
    return initial_model(request.getfixturevalue("current")[0], maximum_mean_kl=0.05)


def reseal(value):
    value.pop("model_hash")
    value["model_hash"] = hash_json(value)
    return value


def test_initial_copy_exact_and_histories_independent(imitation_parent):
    value = make_model(imitation_parent)
    before = copy.deepcopy(value)
    original = select_proposal_decoder(
        parent_preview(imitation_parent), implementation="bounded_snapshot"
    )
    a, b = [CompiledImitationProposalMotor(make_preview(value)) for _ in range(2)]
    snapshot_policy, snapshot = compile_imitation_snapshot(value)
    assert snapshot_policy == make_preview(value)
    for query in np.random.default_rng(674).normal(size=(5, 134)):
        for phase in range(3):
            np.testing.assert_array_equal(a.raw_mean(query, phase), original.raw_mean(query, phase))
            np.testing.assert_array_equal(snapshot.raw_mean(query, phase), a.raw_mean(query, phase))
        np.testing.assert_array_equal(snapshot.features(query), a.features(query))
    assert a._memory is not b._memory
    assert a._parent._memory is not b._parent._memory
    assert a._parent._output_memory.to_dict() == original._parent._output_memory.to_dict()
    assert value == before
    assert snapshot._parent._output_memory.to_dict() == a._parent._output_memory.to_dict()
    value["residual_layers"][-1]["bias"][0] += 1
    assert snapshot_policy["step_motor_proof"]["model"] == before
    assert snapshot._layers[-1][1][0] == before["residual_layers"][-1]["bias"][0]
    assert (
        snapshot_policy["step_motor_proof"]["qualification"]
        == "UNQUALIFIED_SIM_SUPERVISED_IMITATION_NOT_PPO"
    )


@pytest.mark.parametrize(
    "field",
    [
        "promotion_authorized",
        "hardware_authorized",
        "runtime_execution_authorized",
        "fresh_holdout_open_authorized",
        "protected_memory_changed",
        "physical_action_bounds_changed",
    ],
)
def test_resealed_authority_or_bounds_change_rejected(imitation_parent, field):
    value = make_model(imitation_parent)
    value[field] = True
    with pytest.raises(ValueError):
        validate_model(reseal(value))


def test_unreceipted_change_rejected(imitation_parent):
    layers = copy.deepcopy(imitation_parent["residual_layers"])
    layers[-1]["bias"][0] += 0.01
    with pytest.raises(ValueError, match="unreceipted"):
        make_model(imitation_parent, residual_layers=layers)


def test_snapshot_rejects_resealed_parent_before_numeric_allocation(imitation_parent, monkeypatch):
    from rosclaw_soccer.rsi import imitation_proposal_snapshot_compilation as compiler

    value = make_model(imitation_parent)
    value["frozen_parent"]["runtime_execution_authorized"] = True
    value["parent_model_hash"] = reseal(value["frozen_parent"])["model_hash"]
    reseal(value)
    allocations = []

    def forbidden(values):
        allocations.append(values)
        raise AssertionError("invalid ancestor reached numeric allocation")

    monkeypatch.setattr(compiler, "_layers", forbidden)
    with pytest.raises(ValueError):
        compiler.compile_imitation_snapshot(value)
    assert allocations == []


def test_private_factory_preserves_law_and_independent_contact_state(imitation_parent):
    from rosclaw_soccer.rsi.imitation_proposal_episode_factory import (
        ImitationProposalEpisodeFactory,
        validate_compilation_contract,
    )

    artifact = make_model(imitation_parent)
    factory = ImitationProposalEpisodeFactory(artifact)
    first, second = factory.new_episode(), factory.new_episode()
    assert first._memory is not second._memory
    assert first._parent._memory is not second._parent._memory
    assert first._parent._warm is not second._parent._warm
    assert first._layers is not second._layers
    assert first._layers[0][0] is second._layers[0][0]
    with pytest.raises(ValueError):
        first._layers[0][0][0, 0] = 1
    query = np.ones(134) * 0.23
    for phase in range(3):
        np.testing.assert_array_equal(
            first.raw_mean(query, phase), factory._prototype.raw_mean(query, phase)
        )
    first._memory.advance(0, np.zeros(6))
    first._memory.advance(1, np.ones(6) * 2)
    assert second._memory.last_frame == factory._prototype._memory.last_frame == -1
    policy = factory.preview(artifact)
    validate_compilation_contract(factory.contract(), policy)
    changed = copy.deepcopy(factory.contract())
    changed["hardware_authorized"] = 0
    with pytest.raises(ValueError):
        validate_compilation_contract(changed, policy)
    artifact["residual_layers"][-1]["bias"][0] += 0.01
    with pytest.raises(ValueError, match="canonical"):
        factory.preview(artifact)


def test_private_factory_rejects_changed_source(imitation_parent, monkeypatch):
    from rosclaw_soccer.rsi.imitation_proposal_episode_factory import (
        ImitationProposalEpisodeFactory,
    )

    factory = ImitationProposalEpisodeFactory(make_model(imitation_parent))
    pins = dict(factory._pins)
    pins[next(iter(pins))] = "sha256:" + "a" * 64
    monkeypatch.setattr(factory, "_pins", pins)
    with pytest.raises(ValueError, match="sources"):
        factory.new_episode()


def test_actual_supervised_parameters_bounded_and_receipt_bound(imitation_parent):
    pytest.importorskip("torch")
    parent = select_proposal_decoder(
        parent_preview(imitation_parent), implementation="bounded_snapshot"
    )
    x = np.random.default_rng(675).normal(size=(1080, 134))
    phase = np.tile(np.arange(3), 360)
    context = np.column_stack((np.stack([parent.features(v) for v in x])[:, :134], phase))
    base = np.stack([parent._parent.raw_mean(v, int(p)) for v, p in zip(x, phase, strict=True)])
    result = fit_bounded_residual_imitation(
        layers=[
            (np.asarray(v["weight"]), np.asarray(v["bias"]))
            for v in imitation_parent["residual_layers"]
        ],
        context=context,
        baseline=base,
        gates=parent._guard.gates(context),
        targets=base + 0.015,
        training_weights=np.ones(1080),
        config=BoundedResidualImitationConfig(steps=16),
    )
    fitted = result.pop("layers")
    result.update(
        behavior_model_hash=imitation_parent["model_hash"],
        physical_batch_hash="sha256:" + "a" * 64,
        teacher_selection_hash="sha256:" + "b" * 64,
        teacher_selection_offline=True,
        all_success_and_failure_episodes_retained=True,
        teacher_labels_are_actor_inputs=False,
        private_fresh_accessed=False,
    )
    trained_model = make_model(imitation_parent, residual_layers=fitted, learning_receipt=result)
    actor = CompiledImitationProposalMotor(make_preview(trained_model))
    _, snapshot = compile_imitation_snapshot(trained_model)
    for query in x[:8]:
        for p in range(3):
            np.testing.assert_array_equal(snapshot.raw_mean(query, p), actor.raw_mean(query, p))
            assert (
                np.max(np.abs(actor.raw_mean(query, p) - actor._parent.raw_mean(query, p))) <= 0.2
            )
    assert trained_model["frozen_parent"] == imitation_parent
    changed = copy.deepcopy(trained_model)
    changed["residual_layers"][-1]["bias"][0] += 0.02
    with pytest.raises(ValueError, match="receipt"):
        validate_model(reseal(changed))
    for key, replacement in (
        ("online_rl_claimed", True),
        ("physical_batch_verified", True),
        ("teacher_labels_are_actor_inputs", True),
        ("zero_weight_rows_are_not_negative_gradient_examples", False),
        ("completed_optimizer_steps", True),
        ("positive_weight_rows", 1079),
    ):
        changed = copy.deepcopy(trained_model)
        changed["learning_receipt"][key] = replacement
        with pytest.raises(ValueError):
            validate_model(reseal(changed))
