import numpy as np
import pytest
import torch
from test_velocity_prior_actor import tensors

from rosclaw_soccer.rsi.anchored_receiving_motor import AnchoredReceivingMotor
from rosclaw_soccer.rsi.receiver_body_context import receiver_context14
from rosclaw_soccer.rsi.velocity_prior_actor import VelocityPriorActor


def motor():
    original = VelocityPriorActor(tensors(), reference_features=14)
    actor = AnchoredReceivingMotor(original, torch.full((29,), 0.25, dtype=torch.float32))
    return original, actor


def test_zero_identity_is_exact_and_all_whole_MLP_layers_receive_gradient():
    original, actor = motor()
    obs = torch.randn(9, 112)
    reference = torch.randn(9, 29)
    result, evidence = actor.forward_with_evidence(obs, reference)
    assert torch.equal(result, reference)
    assert evidence["realized_delta_radians"].eq(0).all()
    actor.train()
    result.square().mean().backward()
    assert len(actor.plastic_parameters()) == 8
    assert all(p.grad is not None and p.grad.abs().sum() > 0 for p in actor.plastic_parameters())
    assert all(p.grad is None and not p.requires_grad for p in actor.baseline.parameters())
    assert not actor.baseline.training
    assert all(p.grad is None for p in original.parameters())
    assert all(
        p.untyped_storage().data_ptr() != q.untyped_storage().data_ptr()
        for p, q in zip(original.parameters(), actor.plastic.parameters(), strict=True)
    )
    assert not evidence["neural_learning_proven"] and not evidence["physical_retention_proven"]


def test_extreme_plastic_request_remains_inside_original_radian_box():
    _, actor = motor()
    with torch.no_grad():
        actor.plastic.mlp[6].bias.add_(100)
    reference = torch.linspace(-2.0, 2.0, 29)[None].repeat(7, 1)
    out, ev = actor.forward_with_evidence(torch.randn(7, 112), reference)
    assert (ev["realized_delta_radians"].abs() <= 0.2).all()
    assert not torch.equal(out, reference)


def test_nonfinite_frozen_clone_is_rejected_not_saturated_to_finite_action():
    _, actor = motor()
    with torch.no_grad():
        actor.baseline.mlp[6].bias.fill_(float("inf"))
    with pytest.raises(ValueError, match="nonfinite"):
        actor(torch.randn(2, 112), torch.zeros(2, 29))


@pytest.mark.parametrize(
    "fault", ["nan", "reference", "reference_grad", "shape", "baseline_grad", "baseline_train"]
)
def test_invalid_or_unfrozen_reference_not_silently_accepted(fault):
    _, actor = motor()
    obs = torch.randn(2, 112)
    ref = torch.zeros(2, 29)
    if fault == "nan":
        obs[0, 0] = float("nan")
    elif fault == "reference":
        ref = ref.double()
    elif fault == "reference_grad":
        ref.requires_grad_(True)
    elif fault == "shape":
        obs = obs[:, :111]
    elif fault == "baseline_grad":
        actor.baseline.requires_grad_(True)
    else:
        actor.baseline.train()
    with pytest.raises(ValueError):
        actor(obs, ref)


def state():
    q = np.zeros(43, np.float64)
    q[:3] = [1.0, 2.0, 0.8]
    q[3] = 1.0
    q[36:39] = [1.0, 3.0, 0.11]
    v = np.zeros(41, np.float64)
    v[:3] = [0.0, 1.0, 0.0]
    v[35:38] = [0.0, 3.0, 0.0]
    anchor = np.array([1.0, 4.0, 0.11], np.float64)
    return q, v, anchor


def test_causal_body_cues_rotate_vectors_without_mutating_or_future_labels():
    q, v, a = state()
    q[3:7] = [np.sqrt(0.5), 0, 0, np.sqrt(0.5)]
    before = [x.copy() for x in (q, v, a)]
    result = receiver_context14(q, v, a, foot_contact_seen=True, other_body_contact_seen=False)
    np.testing.assert_allclose(result[:3], [1.0, 0.0, -0.69], atol=1e-7)
    np.testing.assert_allclose(result[3:6], [2.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(result[6:9], [2.0, 0.0, -0.69], atol=1e-7)
    np.testing.assert_allclose(result[9:12], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_array_equal(result[12:], [1.0, 0.0])
    assert result.dtype == np.float32
    for x, y in zip((q, v, a), before, strict=True):
        np.testing.assert_array_equal(x, y)


@pytest.mark.parametrize("fault", ["nan", "unit", "dtype", "shape", "flag"])
def test_bad_cue_state_is_rejected_not_padded_or_repaired(fault):
    q, v, a = state()
    flag = False
    if fault == "nan":
        v[0] = np.nan
    elif fault == "unit":
        q[3] = 0.5
    elif fault == "dtype":
        a = a.astype(np.float32)
    elif fault == "shape":
        q = q[:-1]
    else:
        flag = 1
    with pytest.raises(ValueError):
        receiver_context14(q, v, a, foot_contact_seen=flag, other_body_contact_seen=False)
