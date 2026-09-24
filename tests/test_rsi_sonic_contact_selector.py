import copy

import pytest

from rosclaw_soccer.rsi.sonic_contact_selector import SCHEMA, choose_lateral
from rosclaw_soccer.sim.contracts import hash_json


def _model(threshold=0.0875):
    body = {
        "schema": SCHEMA,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "threshold_y_m": threshold,
    }
    return {**body, "model_hash": hash_json(body)}


def test_selector_is_a_bounded_deterministic_decision_stump():
    model = _model()
    assert choose_lateral(model, ball_x_m=1.8, ball_y_m=0.04) == -0.1
    assert choose_lateral(model, ball_x_m=1.8, ball_y_m=0.08) == -0.1
    assert choose_lateral(model, ball_x_m=1.8, ball_y_m=0.12) == 0.0
    assert choose_lateral(model, ball_x_m=1.8, ball_y_m=0.16) == 0.0


def test_selector_rejects_tampered_or_out_of_domain_inputs():
    model = _model()
    forged = copy.deepcopy(model)
    forged["threshold_y_m"] = 0.13
    with pytest.raises(ValueError, match="integrity"):
        choose_lateral(forged, ball_x_m=1.8, ball_y_m=0.1)
    with pytest.raises(ValueError, match="bounded"):
        choose_lateral(model, ball_x_m=2.0, ball_y_m=0.1)
    with pytest.raises(ValueError, match="bounded"):
        choose_lateral(model, ball_x_m=1.8, ball_y_m=float("nan"))
