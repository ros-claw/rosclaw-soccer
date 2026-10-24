import hashlib
import json
from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_role_command import ROLE_COMMAND_SCHEMA, NativeRoleCommandPolicy
from rosclaw_soccer.rsi.native_role_residual import (
    ROLE_COMMAND_LIMITS,
    NativeRoleResidualPolicy,
    guard_role_residual,
    qualified_role_parent_hash,
    serialize_role_residual,
)


def h(i):
    return "sha256:" + f"{i:064x}"


def fixture_encoder(tmp_path):
    tensors = dict(mean=np.zeros(118).tolist(), std=np.ones(118).tolist())
    for i, shape in enumerate(((128, 118), (128, 128), (3, 128))):
        tensors[f"w{i}"] = np.full(shape, 0.01).tolist()
        tensors[f"b{i}"] = np.full(shape[0], 0.1).tolist()
    data = dict(
        schema=ROLE_COMMAND_SCHEMA,
        binding_hash=h(1),
        observation_dim=118,
        action_semantics="ABSOLUTE_BODY_VELOCITY_COMMAND_BEFORE_CLEARANCE",
        hardware_authorized=False,
        optimizer_updates=1024,
        training_kind="PUBLIC_COURSE_DAGGER_CONTINUATION",
        tensors=tensors,
    )
    payload = json.dumps(data).encode()
    path = tmp_path / "encoder.json"
    path.write_bytes(payload)
    digest = "sha256:" + hashlib.sha256(payload).hexdigest()
    return NativeRoleCommandPolicy(path, expected_hash=digest, expected_binding=h(1)), digest


def residual(tmp_path, **changes):
    encoder, digest = fixture_encoder(tmp_path)
    value = serialize_role_residual(
        encoder_hash=digest, binding_hash=h(1), parent_hash=h(2), dataset_hash=h(3), **changes
    )
    payload = json.dumps(value).encode()
    path = tmp_path / "head.json"
    path.write_bytes(payload)
    policy = NativeRoleResidualPolicy(
        path,
        expected_hash="sha256:" + hashlib.sha256(payload).hexdigest(),
        encoder=encoder,
        expected_encoder_hash=digest,
        expected_binding=h(1),
        expected_parent_hash=h(2),
    )
    return policy


def test_zero_head_preserves_parent_signed_bits_and_never_uses_absolute_head(tmp_path):
    policy = residual(tmp_path)

    def forbidden(_):
        raise AssertionError("old absolute output head was evaluated")

    policy.encoder.propose = forbidden
    parent = np.array([-0.0, 0.0, -1], np.float32)
    proposal = policy.propose(np.ones(118, np.float32), parent)
    final, feasible, changed = guard_role_residual(
        proposal, 0.7, np.zeros((0, 2)), maximum_speed_mps=1.0
    )
    np.testing.assert_array_equal(final.view(np.uint32), parent.view(np.uint32))
    assert feasible and not changed and proposal.exact_parent
    assert proposal.context121.shape == (121,) and proposal.features131.shape == (131,)
    np.testing.assert_array_equal(proposal.features131[-3:], parent / ROLE_COMMAND_LIMITS)
    assert not policy.tensors["head_weight"].flags.writeable


def test_nonzero_proposal_stays_bounded_but_needs_final_asymmetric_body_guard(tmp_path):
    policy = residual(tmp_path, update_index=1, maximum_ratio=0.1, head_bias=np.array([8, -8, 8]))
    parent = np.array([1, 0, 1], np.float32)
    proposal = policy.propose(np.zeros(118), parent)
    assert not proposal.exact_parent
    assert np.all(abs(proposal.delta) <= ROLE_COMMAND_LIMITS * 0.1)
    assert proposal.raw_command[0] > 1 and proposal.raw_command[2] > 1
    final, feasible, changed = guard_role_residual(
        proposal, 0.0, np.zeros((0, 2)), maximum_speed_mps=1.0
    )
    assert feasible and changed
    assert -0.5 <= final[0] <= 1 and abs(final[1]) <= 0.5 and abs(final[2]) <= 1
    assert np.linalg.norm(final[:2]) <= 1 + 1e-6
    assert np.all(abs(final - parent) <= ROLE_COMMAND_LIMITS * 0.1 + 1e-6)


@pytest.mark.parametrize("yaw", [0.0, 0.7, 1.4, 2.7])
def test_final_projection_cannot_expand_parent_distance_budget(tmp_path, yaw):
    policy = residual(tmp_path, update_index=1, maximum_ratio=0.05, head_bias=np.ones(3) * 8)
    parent = np.array([0.99, 0.1, 0.99], np.float32)
    proposal = policy.propose(np.ones(118), parent)
    final, feasible, _ = guard_role_residual(proposal, yaw, np.zeros((0, 2)), maximum_speed_mps=1.0)
    assert feasible
    assert np.all(abs(final - parent) <= ROLE_COMMAND_LIMITS * 0.05 + 1e-6)


def test_explicit_exploration_is_not_mistaken_for_the_mean_or_applied_action(tmp_path):
    policy = residual(tmp_path)
    parent = np.array([0.5, 0, 0], np.float32)
    proposal = policy.propose(np.zeros(118), parent, latent_noise=np.array([0.2, -0.1, 0.3]))
    np.testing.assert_allclose(proposal.normalized_action, np.tanh([0.2, -0.1, 0.3]), atol=1e-14)
    np.testing.assert_allclose(
        proposal.delta, proposal.normalized_action * ROLE_COMMAND_LIMITS * 0.05
    )
    assert not proposal.exact_parent


@pytest.mark.parametrize(
    "changes",
    [
        dict(maximum_ratio=float("nan")),
        dict(maximum_ratio=0.2),
        dict(update_index=True),
        dict(head_weight=np.ones((3, 130))),
        dict(head_bias=np.ones(3)),
    ],
)
def test_invalid_or_nonzero_untrained_head_is_rejected(tmp_path, changes):
    with pytest.raises(ValueError):
        residual(tmp_path, **changes)


def test_forged_parent_link_and_nonfinite_guard_context_are_rejected(tmp_path):
    policy = residual(tmp_path)
    proposal = policy.propose(np.zeros(118), np.zeros(3, np.float32))
    with pytest.raises(ValueError):
        guard_role_residual(
            replace(proposal, raw_command=np.ones(3)), 0.0, np.zeros((0, 2)), maximum_speed_mps=1.0
        )
    with pytest.raises(ValueError):
        guard_role_residual(proposal, float("nan"), np.zeros((0, 2)), maximum_speed_mps=1.0)
    with pytest.raises(ValueError):
        policy.propose(np.zeros(118), np.array([-0.6, 0, 0], np.float32))


def test_qualified_parent_content_hash_is_stable_not_a_checkout_path_hash():
    assert qualified_role_parent_hash() == qualified_role_parent_hash()
    assert qualified_role_parent_hash().startswith("sha256:")
