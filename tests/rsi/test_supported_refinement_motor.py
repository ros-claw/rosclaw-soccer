import pytest
import torch
from test_refinement_motor_adapter import parent_model

from rosclaw_soccer.rsi.feature_support_gate import FeatureSupportGate
from rosclaw_soccer.rsi.refinement_motor_adapter import RefinementMotorAdapter
from rosclaw_soccer.rsi.supported_refinement_motor import SupportedRefinementMotor


def fixture(active):
    parent = parent_model().eval().requires_grad_(False)
    refined = RefinementMotorAdapter(parent)
    with torch.no_grad():
        refined.head[-1].bias.fill_(2)
    obs = torch.randn(1, 1, 160)
    with torch.no_grad():
        _, _, e = parent.forward_with_evidence(obs, parent.initial_state())
    anchors = e["critic_features"][0].clone()
    if not active:
        anchors += 100
    gate = FeatureSupportGate(
        anchors, torch.ones(288, dtype=torch.float64), torch.full((1,), 0.01, dtype=torch.float64)
    )
    return parent, refined, SupportedRefinementMotor(refined, gate), obs


def test_zero_support_preserves_whole_acquired_motor_not_the_unqualified_global_head():
    parent, refined, motor, obs = fixture(False)
    pstate, sstate = parent.initial_state(), motor.initial_state()
    with torch.no_grad():
        for _ in range(8):
            expected, pstate = parent(obs, pstate)
            actual, sstate, e = motor.forward_with_evidence(obs, sstate)
            assert torch.equal(expected, actual)
            assert not e["support_activation"].any()
            assert all(torch.equal(x, y) for x, y in zip(pstate, sstate, strict=True))
            obs = torch.randn_like(obs)
        unqualified, _ = refined(obs, refined.initial_state())
        acquired, _ = parent(obs, parent.initial_state())
        assert not torch.equal(unqualified, acquired)


def test_full_support_reuses_actual_trained_head_inside_original_authority():
    _, refined, motor, obs = fixture(True)
    with torch.no_grad():
        expected, state = refined(obs, refined.initial_state())
        actual, after, e = motor.forward_with_evidence(obs, motor.initial_state())
    assert torch.equal(expected, actual)
    assert e["support_activation"].item() == 1
    assert e["final_mean_delta_radians"].abs().max() <= 0.2
    assert all(torch.equal(x, y) for x, y in zip(state, after, strict=True))
    assert e["physical_retention_proven"] is False


def test_candidate_is_readonly_and_cannot_mutate_or_alias_caller_network():
    parent, refined, motor, _ = fixture(False)
    ptrs = {p.data_ptr() for p in refined.parameters()}
    assert not any(p.data_ptr() in ptrs or p.requires_grad for p in motor.parameters())
    assert not motor.training and not motor.refinement.training
    assert any(p.requires_grad for p in refined.parameters())
    with pytest.raises(ValueError):
        motor.train()
    one, two = motor.initial_state(), motor.initial_state()
    assert one[0].data_ptr() != two[0].data_ptr()
