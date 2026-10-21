import pytest

from rosclaw_soccer.rsi.paid_router_review import compare_router_frontier


def row(q, stable=True):
    return dict(quality_by_kick=q, stable=stable)


def test_equal_aggregate_hides_regression_and_is_rejected():
    result = compare_router_frontier(
        [row([True, True, False]), row([True, True, True])],
        [row([True, True, True]), row([True, True, False])],
    )
    assert result["reference_quality"] == result["candidate_quality"]
    assert result["quality_gains"] == [[0, 2]] and result["quality_losses"] == [[1, 2]]
    assert not result["retention_pass"] and result["has_actual_quality_growth"]


def test_noop_retention_is_not_growth():
    result = compare_router_frontier([row([True], False)], [row([True], False)])
    assert result["retention_pass"] and not result["has_actual_quality_growth"]
    assert result["unique_declared_contexts"] == 1
    assert result["statistical_independence_not_established"]


def test_body_loss_cannot_be_hidden_by_new_goals():
    result = compare_router_frontier([row([True, False, False])], [row([True, True], False)])
    assert result["body_losses"] == [0] and not result["retention_pass"]


def test_growth_preserves_every_case():
    result = compare_router_frontier([row([True, True, False])], [row([True, True, True])])
    assert result["retention_pass"] and result["has_actual_quality_growth"]


@pytest.mark.parametrize(
    "bad",
    [[], [row([], False)], [row([1, True, True])], [row([True])], [row([True, True, True], 1)]],
)
def test_incomplete_or_false_grading_cannot_pass(bad):
    with pytest.raises(ValueError):
        compare_router_frontier([row([True, True, True])], bad)
