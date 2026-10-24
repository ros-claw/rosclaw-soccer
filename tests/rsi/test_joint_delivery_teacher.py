import copy

import pytest
from test_refinement_growth_review import rows

from rosclaw_soccer.rsi.joint_delivery_teacher import joint_delivery_rank, pilot_config


def row(error, speed, *, received=True):
    o = copy.deepcopy(rows()[0]["outcome"])
    o["B_legal_functional_receive_before_deadline"] = received
    o["incoming"]["B_first_ball_contact_substep"] = 20 if received else None
    o["incoming"]["incoming_target_error_3d_before_B_contact_m"] = error
    o["incoming_arrival_speed_mps"] = speed
    o["combined_precision_and_functional_receive"] = bool(
        error is not None and speed is not None and error <= 0.1 and speed <= 3 and received
    )
    return dict(outcome=o)


def test_joint_objective_cannot_be_hidden_by_controlled_receive_alone():
    precise_fast, slow_miss = row(0.0266, 3.59), row(0.4, 2.7)
    assert joint_delivery_rank(row(0.08, 2.9)) > joint_delivery_rank(precise_fast)
    assert joint_delivery_rank(row(0.09, 2.9, received=False)) > joint_delivery_rank(precise_fast)
    assert joint_delivery_rank(row(0.08, 2.9)) > joint_delivery_rank(row(0.08, 2.9, received=False))
    assert joint_delivery_rank(row(0.0266, 3.1)) > joint_delivery_rank(precise_fast)
    assert joint_delivery_rank(row(0.2, 2.7)) > joint_delivery_rank(slow_miss)


def test_safety_precedes_precision_and_unobserved_is_not_fabricated_success():
    a = row(0.01, 2)
    bad = copy.deepcopy(a)
    bad["outcome"]["passer_clean_foot"] = False
    assert joint_delivery_rank(a) > joint_delivery_rank(bad)
    assert joint_delivery_rank(row(None, None))[2:] == (0, -1e6)


@pytest.mark.parametrize("v", [float("nan"), float("inf"), -1, True])
def test_bad_observed_metrics_rejected(v):
    with pytest.raises(ValueError):
        joint_delivery_rank(row(v, 2))


def test_exact_bounded_two_pilot_budget_and_no_fresh_or_optimizer_claim():
    for state, seed in ((25, 115025), (26, 115023)):
        c = pilot_config(state)
        assert c["source_expert_seed"] == seed and c["maximum_worlds"] == 38
        assert c["maximum_native_steps"] == 152000
        assert c["Fresh_cases_read"] == c["new_optimizer_updates"] == 0
    with pytest.raises(ValueError):
        pilot_config(24)
