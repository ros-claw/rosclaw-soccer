import numpy as np
import pytest
import torch

from rosclaw_soccer.rsi.paid_negative_core import (
    NegativeCoreGate,
    gated_neural_delta,
    negative_core_certificate,
    negative_core_config,
)
from rosclaw_soccer.rsi.paid_replay_neural import ReplayMotorHead


def feature(values):
    x = torch.zeros(len(values), 288)
    x[:, 0] = torch.tensor(values, dtype=torch.float32)
    return x


def test_anchors_core_and_broad_feedback_not_positive_cones():
    gate = NegativeCoreGate(feature([10]), feature([0]))
    gains = gate(feature([0, 0.5, 5, 10, 20]))[:, 0].numpy()
    assert np.array_equal(gains[:2], np.zeros(2))
    assert gains[3] == 1 and 0 < gains[2] < 1 and 0 < gains[4] < 1
    assert abs(gains[2] - 4 / 9) < 1e-7
    assert negative_core_config()["new_neural_optimizer_updates"] == 0


def test_conflicting_observation_labels_fail_closed():
    gate = NegativeCoreGate(feature([0]), feature([0]))
    with pytest.raises(ValueError, match="disjoint"):
        gate(feature([0]))


def test_exact_motor_parent_in_core_and_unchanged_authority_outside():
    head = ReplayMotorHead(torch.zeros(288), torch.ones(288))
    with torch.no_grad():
        head.net[-1].bias.fill_(1)
    gate = NegativeCoreGate(feature([10]), feature([0]))
    x = feature([0, 0.5, 5, 10, 20])
    prior = torch.full((5, 29), 0.1)
    result = gated_neural_delta(head, gate, x, prior)
    assert torch.equal(result[:2], prior[:2])
    assert (result[2:] > prior[2:]).all() and result.abs().max() <= 0.2


def test_full_anchor_and_brute_force_certificate_on_toy_train():
    head = ReplayMotorHead(torch.zeros(288), torch.ones(288)).eval().requires_grad_(False)
    x = feature([0, 10])
    y = np.zeros((2, 29), np.float32)
    y[1, 0] = 0.1
    z = dict(
        features=x.numpy(), target=y, parent_delta=np.zeros_like(y), active=np.array([False, True])
    )
    gate = NegativeCoreGate(x[1:], x[:1])
    cert = negative_core_certificate(head, gate, z)
    assert cert["protected_gain_zero_rows"] == cert["active_gain_one_rows"] == 1
    assert cert["maximum_protected_motor_change_rad"] == 0
    assert not cert["physical_qualification"] and cert["brute_force_queries"] == 32


def test_frozen_unseen_fixture_rejects_this_new_model_before_reading_it(tmp_path):
    from rosclaw_soccer.rsi.paid_candidate_dispatch import negative_core_train_candidate

    with pytest.raises(ValueError, match="never old frozen unseen"):
        negative_core_train_candidate(
            tmp_path, "newhash", {}, dict(plan_kind="frozen_growth_pair_unseen_local_probe32")
        )
