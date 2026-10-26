import pytest

from rosclaw_soccer.rsi.cftrack_multi_axis_torque_response import propose_multi_axis_torque_response


def metric(x, y):
    return dict(
        private_safe=True,
        post60_observed=True,
        first_foot_control=71,
        forward_60_m=x,
        lateral_60_m=y,
    )


def test_current_positive_probes_produce_one_bounded_certified_proposal():
    parent = metric(0.75, 0.57)
    supports = {
        7: metric(0.88, 0.35),
        8: metric(0.95, 0.47),
        11: metric(0.79, 0.57),
        12: metric(0.80, 0.52),
    }
    plan, report = propose_multi_axis_torque_response(
        parent, supports, body_hash="sha256:" + "a" * 64, parent_policy_hash="sha256:" + "b" * 64
    )
    x, y = report["predicted_xy"]
    assert x >= 1.2 - 1e-9 and abs(y) <= 0.2 * x + 1e-9
    assert all(0 <= v <= 0.25 for v in plan.fraction_effort_hg)
    assert all(plan.fraction_effort_hg[i] == 0 for i in range(29) if i not in (7, 8, 11, 12))
    assert report["physical_feasibility_not_implied"] and report["neural_optimizer_updates"] == 0


def test_unsafe_and_censored_axes_cannot_contribute():
    parent = metric(0.75, 0.57)
    bad = metric(100.0, -100.0)
    bad["private_safe"] = False
    censored = metric(100.0, -100.0)
    censored["post60_observed"] = False
    supports = {7: bad, 8: censored, 11: bad, 12: censored}
    with pytest.raises(ValueError, match="all current positive"):
        propose_multi_axis_torque_response(
            parent,
            supports,
            body_hash="sha256:" + "a" * 64,
            parent_policy_hash="sha256:" + "b" * 64,
        )


def test_single_unsafe_axis_stays_zero_when_another_safe_axis_is_feasible():
    parent = metric(0.75, 0.57)
    bad = metric(100.0, -100.0)
    bad["private_safe"] = False
    supports = {7: bad, 8: metric(1.2, 0.2), 11: metric(0.75, 0.57), 12: metric(0.75, 0.57)}
    plan, report = propose_multi_axis_torque_response(
        parent, supports, body_hash="sha256:" + "a" * 64, parent_policy_hash="sha256:" + "b" * 64
    )
    assert plan.fraction_effort_hg[7] == 0
    assert report["excluded_unsafe_or_censored_axes"] == [7]


def test_missing_axis_or_insufficient_authority_abstains():
    parent = metric(0.75, 0.57)
    with pytest.raises(ValueError, match="all four"):
        propose_multi_axis_torque_response(
            parent,
            {7: metric(0.8, 0.5)},
            body_hash="sha256:" + "a" * 64,
            parent_policy_hash="sha256:" + "b" * 64,
        )
    supports = {axis: metric(0.75, 0.57) for axis in (7, 8, 11, 12)}
    with pytest.raises(ValueError, match="no numerically certified"):
        propose_multi_axis_torque_response(
            parent,
            supports,
            body_hash="sha256:" + "a" * 64,
            parent_policy_hash="sha256:" + "b" * 64,
        )
