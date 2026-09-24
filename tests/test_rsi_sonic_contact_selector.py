import copy

import pytest

from rosclaw_soccer.rsi.sonic_contact_selector import SCHEMA, SCHEMA_V2, SCHEMA_V3, choose_lateral
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


def test_phase_aware_v2_selects_right_foot_course_only_in_far_high_zone():
    body = {
        "schema": SCHEMA_V2,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "threshold_y_m": 0.0875,
        "far_distance_switch_x_m": 1.85,
        "far_right_foot_switch_y_m": 0.1,
    }
    model = {**body, "model_hash": hash_json(body)}
    assert choose_lateral(model, ball_x_m=1.8, ball_y_m=0.12) == 0.0
    assert choose_lateral(model, ball_x_m=1.9, ball_y_m=0.04) == -0.1
    assert choose_lateral(model, ball_x_m=1.9, ball_y_m=0.12) == 0.1
    assert choose_lateral(model, ball_x_m=2.0, ball_y_m=0.16) == 0.1
    forged = copy.deepcopy(model)
    forged["far_distance_switch_x_m"] = 2.0
    forged["model_hash"] = hash_json(
        {key: value for key, value in forged.items() if key != "model_hash"}
    )
    with pytest.raises(ValueError, match="phase-aware"):
        choose_lateral(forged, ball_x_m=1.9, ball_y_m=0.12)


def test_continuous_v3_interpolates_only_inside_bounded_far_course():
    body = {
        "schema": SCHEMA_V3,
        "activation_ceiling": "SIM_ONLY",
        "promotion_authorized": False,
        "threshold_y_m": 0.0875,
        "far_distance_switch_x_m": 1.85,
        "far_right_foot_switch_y_m": 0.1,
        "continuous_right_foot_start_x_m": 1.95,
        "continuous_right_foot_knots": [[0.04, 0.04], [0.08, 0.06], [0.12, 0.1], [0.16, 0.1]],
    }
    model = {**body, "model_hash": hash_json(body)}
    assert choose_lateral(model, ball_x_m=2.1, ball_y_m=0.05) == 0.045
    assert choose_lateral(model, ball_x_m=2.1, ball_y_m=0.09) == 0.07
    assert choose_lateral(model, ball_x_m=2.1, ball_y_m=0.13) == 0.1
    assert choose_lateral(model, ball_x_m=1.9, ball_y_m=0.13) == 0.1
    forged = copy.deepcopy(model)
    forged["continuous_right_foot_knots"][0][1] = 0.5
    forged["model_hash"] = hash_json(
        {key: value for key, value in forged.items() if key != "model_hash"}
    )
    with pytest.raises(ValueError, match="teacher knots"):
        choose_lateral(forged, ball_x_m=2.1, ball_y_m=0.05)
