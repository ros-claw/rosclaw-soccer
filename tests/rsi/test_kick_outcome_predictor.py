import numpy as np
import pytest

from rosclaw_soccer.rsi.kick_outcome_predictor import (
    group_weights,
    grouped_metrics,
    grouped_partition,
    make_predictor,
)


def test_partition_initial_states_not_proposal_rows():
    groups = np.array([f"class{k}-state{i}" for k in range(4) for i in range(14)])
    strata = np.repeat(np.arange(4), 14)
    split = grouped_partition(groups, strata)
    assert len(split["train_groups"]) == 40
    assert len(split["validation_groups"]) == 16
    assert not set(split["train_groups"]) & set(split["validation_groups"])
    assert grouped_partition(np.repeat(groups, 3), np.repeat(strata, 3)) == split
    with pytest.raises(ValueError):
        grouped_partition(groups[:-1], strata[:-1])
    with pytest.raises(ValueError):
        grouped_partition(np.append(groups, groups[0]), np.append(strata, 1))


def test_equal_group_mass_and_metrics_not_row_mass():
    groups = np.array(["a", "b", "b", "b"])
    np.testing.assert_allclose(group_weights(groups), [0.5, 1 / 6, 1 / 6, 1 / 6])
    y = np.array([[0, 0, 0], [1, 1, 1], [1, 1, 1], [1, 1, 1]])
    metrics = grouped_metrics(np.ones((4, 3)), y, groups)
    np.testing.assert_allclose(metrics["group_weighted_brier"], [0.5] * 3)
    assert metrics["safe_high_confidence_error"] == pytest.approx(0.5)
    assert grouped_metrics(np.zeros((4, 3)), y, groups)["safe_high_confidence_error"] is None
    with pytest.raises(ValueError):
        grouped_metrics(np.full((4, 3), np.nan), y, groups)


def test_predictor_is_differentiable_but_has_no_motor_authority():
    torch = pytest.importorskip("torch")
    torch.set_num_threads(1)
    model = make_predictor(np.zeros(549, np.float32), np.ones(549, np.float32))
    x = torch.zeros((2, 549), dtype=torch.float32)
    p = model.probabilities(x)
    assert p.shape == (2, 3)
    assert bool((p[:, 2] <= p[:, :2].min(dim=1).values).all())
    p.sum().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())
    assert model.simulation_only and not model.motor_authority
    with pytest.raises(ValueError):
        model(torch.full_like(x, float("nan")))
    with pytest.raises(ValueError):
        make_predictor(np.zeros(549, np.float32), np.zeros(549, np.float32))
