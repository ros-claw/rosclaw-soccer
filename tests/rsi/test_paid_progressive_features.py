import numpy as np
import pytest
import torch
from test_paid_successor_features import fixture

from rosclaw_soccer.rsi.paid_contact_residual import DecoupledContactResidual
from rosclaw_soccer.rsi.paid_progressive_features import progressive_features


def data():
    base, raw = fixture()
    parent = DecoupledContactResidual().eval().requires_grad_(False)
    with torch.no_grad():
        parent.net[-1].bias.fill_(0.5)
        parent.gate[-1].bias.fill_(1)
        prior = parent(torch.zeros(1, 288))[0].numpy()
    raw["NN_nominal_action"][:] = prior
    raw["NN_action"][:] = prior
    return base, parent, raw


def test_acquired_success_is_zero_relative_to_parent_not_zero_relative_to_r2():
    base, parent, raw = data()
    x, correction, prior = progressive_features(
        base, parent, raw, torch.ones(29), recorded_policy="parent", protect=True
    )
    assert x.shape == (3, 288)
    assert not correction.any() and prior.min() > 0
    assert np.array_equal(prior, raw["NN_action"])


def test_teacher_correction_is_own_parent_difference_not_reused_r2_label():
    base, parent, raw = data()
    raw["NN_nominal_action"][:] = 0
    raw["NN_action"][:] = -0.19
    _, correction, prior = progressive_features(
        base, parent, raw, torch.ones(29), recorded_policy="base", protect=False
    )
    assert np.allclose(correction, -0.19 - prior) and (correction < -0.2).all()


@pytest.mark.parametrize(
    "fault",
    [
        "before",
        "after",
        "nominal",
        "action",
        "nan",
        "dtype",
        "bound",
        "parent_training",
        "parent_grad",
    ],
)
def test_no_borrowed_memory_false_policy_or_unbounded_teacher_labels(fault):
    base, parent, raw = data()
    if fault in ("before", "after"):
        raw["NN_h_" + fault][1, 0, 0, 0] += 1
    elif fault == "nominal":
        raw["NN_nominal_action"][1, 0] = 0
    elif fault in ("action", "bound"):
        raw["NN_action"][1, 0] = 0.3
    elif fault == "nan":
        raw["NN_obs"][1, 0] = np.nan
    elif fault == "dtype":
        raw["NN_action"] = raw["NN_action"].astype(np.float64)
    elif fault == "parent_training":
        parent.train()
    else:
        parent.requires_grad_(True)
    with pytest.raises(ValueError):
        progressive_features(
            base, parent, raw, torch.ones(29), recorded_policy="parent", protect=True
        )


def test_r2_success_cannot_be_relabelled_as_parent_success_if_parent_would_change_it():
    base, parent, raw = data()
    raw["NN_nominal_action"][:] = 0
    raw["NN_action"][:] = 0
    with pytest.raises(ValueError, match="not actually emitted"):
        progressive_features(
            base, parent, raw, torch.ones(29), recorded_policy="base", protect=True
        )
