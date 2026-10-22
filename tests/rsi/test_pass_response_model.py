"""Synthetic regression mechanics only, not a qualified physical teacher."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.pass_response_model import feasible_first_rank, response_proposals


def data():
    rng = np.random.default_rng(1)
    z = rng.uniform(-0.4, 0.4, (30, 2, 6))
    z[0] = 0
    base = np.array([0.1, 0.05, 0.01, 0.3])
    y = z.reshape(30, 12)[:, :4] + base
    return z, y, base


def test_fits_real_data_shape_and_proposes_bounded_steps_without_claiming_physics():
    z, y, base = data()
    result = response_proposals(z, y, start=np.zeros((2, 6)), start_response=base)
    assert len(result["models"]) == 3 and len(result["candidates"]) == 6
    assert not result["predictions_are_physical_evidence"]
    assert result["student_optimizer_updates"] == 0
    errors = []
    for proposal in result["candidates"]:
        candidate = np.array(proposal["latent"])
        assert abs(candidate).max() <= 0.2 + 1e-12
        errors.append(np.linalg.norm(candidate.reshape(12)[:4] + base))
    assert min(errors) < np.linalg.norm(base)


def test_insufficient_or_rank_deficient_data_is_an_explicit_negative_not_fake_fit():
    z, y, base = data()
    result = response_proposals(z[:13], y[:13], start=np.zeros((2, 6)), start_response=base)
    assert result["status"] == "INSUFFICIENT_SAFE_UNIQUE_DATA" and result["candidates"] == []
    z[:, :, 1:] = 0
    result = response_proposals(z, y, start=np.zeros((2, 6)), start_response=base)
    assert result["status"] == "RANK_DEFICIENT_SAFE_DATA"


def test_duplicate_trajectories_cannot_inflate_training_support():
    z, y, base = data()
    z[1] = z[0]
    with pytest.raises(ValueError, match="unique"):
        response_proposals(z, y, start=np.zeros((2, 6)), start_response=base)


def test_feasible_witness_beats_v1_soft_costs_infeasible_incumbent():
    common = dict(
        whole_body_healthy=True,
        recovery_observed=True,
        clean_foot=True,
        first_foot_contact_substep=1000,
    )
    infeasible = dict(
        common,
        minimum_goal_error_3d_m=0.0499,
        arrival_speed_mps=3.064,
        precision_0p1m_moderate_speed_success=False,
    )
    feasible = dict(
        common,
        minimum_goal_error_3d_m=0.0556,
        arrival_speed_mps=2.743,
        precision_0p1m_moderate_speed_success=True,
    )
    assert feasible_first_rank(feasible) > feasible_first_rank(infeasible)


def test_already_slow_delivery_is_not_pushed_to_increase_speed():
    z, y, base = data()
    y[:, 3] -= 0.8
    base[3] -= 0.8
    result = response_proposals(z, y, start=np.zeros((2, 6)), start_response=base)
    assert result["safe_speed_is_not_accelerated_to_margin"]
    assert abs(np.array(result["candidates"][0]["latent"]).reshape(12)[3]) < 0.02
