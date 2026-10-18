import pytest

from rosclaw_soccer.rsi.paired_learning_assessment import assess_pairs


def test_rescues_do_not_hide_regressions_or_new_unsafe_failures():
    controls = {
        "a": dict(task_pass=True, safe=True),
        "b": dict(task_pass=False, safe=True),
        "c": dict(task_pass=False, safe=False),
    }
    evaluations = {
        "a": dict(task_pass=False, safe=False),
        "b": dict(task_pass=True, safe=True),
        "c": dict(task_pass=False, safe=True),
    }
    result = assess_pairs(controls, evaluations)
    assert result["rescued_pairs"] == result["regressed_pairs"] == 1
    assert result["net_success_gain"] == 0
    assert result["newly_unsafe_pairs"] == 1
    assert len(result["rows"]) == 3
    assert not result["promotion_authorized"]


@pytest.mark.parametrize(
    "controls,evaluations",
    [
        ({}, {}),
        ({"a": dict(task_pass=False, safe=True)}, {}),
        ({"a": dict(task_pass=True, safe=False)}, {"a": dict(task_pass=True, safe=True)}),
        ({"a": dict(task_pass=1, safe=True)}, {"a": dict(task_pass=True, safe=True)}),
        ({"a": dict(task_pass=False)}, {"a": dict(task_pass=True, safe=True)}),
    ],
)
def test_missing_failure_or_malformed_labels_rejected(controls, evaluations):
    with pytest.raises(ValueError):
        assess_pairs(controls, evaluations)
