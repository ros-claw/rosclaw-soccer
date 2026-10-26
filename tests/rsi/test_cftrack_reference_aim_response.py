import pytest

from rosclaw_soccer.rsi.cftrack_reference_aim_response import propose_reference_aim


def metric(x, y):
    return dict(private_safe=True, post60_observed=True, forward_60_m=x, lateral_60_m=y)


def test_one_negative_aim_probe_yields_bounded_model_proposal():
    result = propose_reference_aim(metric(0.86, 0.32), metric(1.4, 0.10))
    assert -0.7 <= result["reference_goal_offset_y_m"] <= 0
    x, y = result["predicted_xy"]
    assert x >= 1.2 - 1e-9 and abs(y) <= 0.2 * x + 1e-9
    assert result["actual_task_goal_and_h300_unchanged"]
    assert result["physical_feasibility_not_implied"]


@pytest.mark.parametrize("bad", ["unsafe", "censored", "nonfinite"])
def test_invalid_probe_abstains(bad):
    value = metric(1.4, 0.1)
    if bad == "unsafe":
        value["private_safe"] = False
    if bad == "censored":
        value["post60_observed"] = False
    if bad == "nonfinite":
        value["lateral_60_m"] = float("nan")
    with pytest.raises(ValueError):
        propose_reference_aim(metric(0.86, 0.32), value)


def test_unresponsive_reference_is_not_a_model_success():
    with pytest.raises(ValueError, match="no numerically certified"):
        propose_reference_aim(metric(0.86, 0.32), metric(0.86, 0.32))
