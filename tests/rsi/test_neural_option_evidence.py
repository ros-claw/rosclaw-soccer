import copy

import pytest
from test_refinement_growth_review import rows

from rosclaw_soccer.rsi.neural_option_evidence import neural_option_evidence


def test_measured_experts_can_offer_goal_and_safety_improvement_without_relabeling_failures():
    p = rows()
    p[42]["outcome"]["passer_clean_foot"] = False
    experts = {s: copy.deepcopy(p) for s in (115023, 115024, 115025)}
    experts[115023][40]["outcome"]["B_legal_functional_receive_before_deadline"] = True
    experts[115024][42]["outcome"]["passer_clean_foot"] = True
    result = neural_option_evidence(p, experts)
    assert result["parent_controlled"] == 40 and result["selected_observed_controlled"] == 41
    assert result["plan"][40]["selected_seed"] == 115023
    assert result["plan"][42]["selected_seed"] == 115024
    assert result["plan"][42]["selected_observed_controlled"] is False
    assert result["plan"][42]["reason"] == "clean_contact_neural_expert_not_goal_success"
    assert result["composition_executed"] is False
    assert all(r["selected_seed"] is None for r in result["plan"][:40])


def test_illegal_receiver_contact_does_not_become_a_qualified_repair():
    p = rows()
    p[40]["outcome"]["passer_clean_foot"] = False
    experts = {s: copy.deepcopy(p) for s in (115023, 115024, 115025)}
    for x in experts.values():
        x[40]["outcome"]["passer_clean_foot"] = True
        x[40]["outcome"]["B_excluded_contact"] = True
    assert neural_option_evidence(p, experts)["plan"][40]["selected_seed"] is None


def test_selector_plan_cannot_silently_use_a_different_or_partial_exam():
    p = rows()
    experts = {s: copy.deepcopy(p) for s in (115023, 115024, 115025)}
    experts[115023][0]["course"]["target"] = [1, 2, 3]
    with pytest.raises(ValueError):
        neural_option_evidence(p, experts)
