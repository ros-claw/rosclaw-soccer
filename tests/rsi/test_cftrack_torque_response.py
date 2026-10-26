import pytest

from rosclaw_soccer.rsi.cftrack_torque_response import propose_torque_response


def metric(x, y):
    return dict(
        private_safe=True,
        post60_observed=True,
        first_foot_control=71,
        forward_60_m=x,
        lateral_60_m=y,
    )


def propose(a, b):
    return propose_torque_response(
        a, b, body_hash="sha256:" + "a" * 64, parent_policy_hash="sha256:" + "b" * 64
    )


def test_one_positive_probe_fits_current_response_and_certifies_math_only():
    plan, report = propose(metric(1.3, 0.67), metric(1.6, 0.47))
    assert plan.start_control == 46 and plan.peak_control == 24
    assert 0 < plan.fraction_effort_hg[7] <= 0.25
    assert sum(v != 0 for v in plan.fraction_effort_hg) == 1
    x, y = report["predicted_xy"]
    assert x >= 1.2 and abs(y) <= 0.2 * x + 1e-9
    assert report["physical_feasibility_not_implied"]
    assert report["neural_optimizer_updates"] == 0


@pytest.mark.parametrize("bad", ["unsafe", "censored", "nonfinite", "late"])
def test_bad_anchor_is_not_converted_to_teacher(bad):
    m = metric(1.3, 0.67)
    if bad == "unsafe":
        m["private_safe"] = False
    if bad == "censored":
        m["post60_observed"] = False
    if bad == "nonfinite":
        m["forward_60_m"] = float("nan")
    if bad == "late":
        m["first_foot_control"] = 140
    with pytest.raises(ValueError):
        propose(m, metric(1.6, 0.47))


def test_no_effect_or_wrong_direction_cannot_relax_margin():
    with pytest.raises(ValueError, match="no numerically certified"):
        propose(metric(1.3, 0.67), metric(1.3, 0.67))
    with pytest.raises(ValueError, match="no numerically certified"):
        propose(metric(1.3, 0.67), metric(1.3, 0.8))
