import pytest
import torch

from rosclaw_soccer.rsi.paid_contact_residual import ContactResidual, DecoupledContactResidual
from rosclaw_soccer.rsi.paid_successor_features import (
    decoupled_successor_config,
    staged_successor_config,
)


def configured(cls, logit):
    residual = cls().eval()
    with torch.no_grad():
        residual.net[-1].weight.zero_()
        residual.net[-1].bias.fill_(6.0)
        residual.gate[-1].weight.zero_()
        residual.gate[-1].bias.fill_(logit)
    return residual


def test_admitted_decoupled_head_can_represent_full_declared_authority():
    x = torch.zeros(2, 288)
    old, new = configured(ContactResidual, 0.01), configured(DecoupledContactResidual, 0.01)
    assert old(x).max() < 0.101
    assert 0.199 < new(x).max() <= 0.2


@pytest.mark.parametrize("logit", [-4.0, -0.1, -0.001])
def test_declined_contexts_still_have_exact_zero_correction(logit):
    new = configured(DecoupledContactResidual, logit)
    assert new(torch.zeros(3, 288)).count_nonzero() == 0


@pytest.mark.parametrize("fault", ["shape", "dtype", "nonfinite"])
def test_decoupled_head_preserves_original_feature_contract(fault):
    x = torch.zeros(2, 288)
    if fault == "shape":
        x = x[:, :160]
    elif fault == "dtype":
        x = x.double()
    else:
        x[0, 0] = float("nan")
    with pytest.raises(ValueError):
        DecoupledContactResidual()(x)


def test_ablation_keeps_seed_sampling_and_authority_unchanged():
    old, new = staged_successor_config(0), decoupled_successor_config(0)
    assert old["seed"] == new["seed"] and old["bound_radians"] == new["bound_radians"] == 0.2
    assert new["gate_scales_amplitude"] is False and "gate_scales_amplitude" not in old
