"""Synthetic observation/fit math, not native football learning evidence."""

import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.observed_ball_response import (
    WIDTH,
    fit,
    fit_matched_effect,
    observation,
    predict,
    predict_matched_effect,
)
from rosclaw_soccer.sim.contracts import hash_json


def inputs():
    return dict(
        frame=73,
        side=0,
        foot_xyz=np.array([1.0, 0.1, 0.05]),
        previous_foot_xyz=np.array([0.99, 0.095, 0.05]),
        ball_xyz=np.array([1.1, 0.13, 0.11]),
        ball_velocity=np.array([0.5, 0.1, 0.0]),
        leg_joint_angles=np.array([0.0, 0.0, 0.0, 0.2, -0.1, 0.05]),
        root_angular_velocity=np.array([0.1, 0.2, 0.3]),
        previous_forces=np.array([4.0, 2.0, 0.0, 0.0, 0.0, 0.0]),
        requested_wrench_coefficients=np.array([0.0, 1.0, 0.0, -1.0]),
    )


def test_current_past_only_and_exact_feature_meaning():
    a = inputs()
    x = observation(**a)
    assert x.shape == (WIDTH,) and x.dtype == np.float64
    np.testing.assert_allclose(x[:3], [0.1, 0.03, 0.06])
    np.testing.assert_allclose(x[3:6], [0.0, -0.15, 0.0], atol=1e-14)
    np.testing.assert_array_equal(x[12:16], [0.0, 1.0, 0.0, -1.0])
    assert x[-1] == 0.35
    with pytest.raises(TypeError):
        observation(**a, future_velocity=np.zeros(3))


def test_left_right_reflection_canonicalizes_state_and_wrench():
    a = inputs()
    x = observation(**a)
    b = copy.deepcopy(a)
    b["side"] = 1
    for key in ("foot_xyz", "previous_foot_xyz", "ball_xyz", "ball_velocity"):
        b[key][1] *= -1
    b["leg_joint_angles"][5] *= -1
    b["root_angular_velocity"][0] *= -1
    b["previous_forces"][:2] = b["previous_forces"][:2][::-1]
    b["requested_wrench_coefficients"][[1, 3]] *= -1
    np.testing.assert_array_equal(x, observation(**b))


@pytest.mark.parametrize(
    "change",
    [
        dict(frame=296),
        dict(side=True),
        dict(side=1),
        dict(previous_forces=np.zeros(6)),
        dict(foot_xyz=np.full(3, np.nan)),
        dict(requested_wrench_coefficients=np.full(4, 2.0)),
    ],
)
def test_malformed_or_unobserved_contact_rejected(change):
    with pytest.raises(ValueError):
        observation(**(inputs() | change))


def test_actual_small_core_fit_exports_no_actor_authority():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(6, WIDTH)).astype(np.float64)
    y = x[:, :3] * 0.1
    model = fit(x, y, seed=1019, steps=4)
    assert model["optimizer_steps"] == 12 and model["actor_optimizer_updates"] == 0
    assert model["partial_observation_not_markov"] and model["used_in_control"] is False
    assert predict(model, x).shape == (6, 3)
    assert np.isfinite(predict(model, x)).all()
    bad = copy.deepcopy(model)
    bad["runtime_execution_authorized"] = True
    bad["model_hash"] = hash_json({k: v for k, v in bad.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        predict(bad, x)
    bad = copy.deepcopy(model)
    bad["center"][0] += 1
    with pytest.raises(ValueError):
        predict(bad, x)


def test_no_target_shape_broadcast_or_undeclared_seed():
    x = np.zeros((2, WIDTH))
    y = np.zeros((2, 3))
    with pytest.raises(ValueError):
        fit(x, y[:, 0], seed=1019, steps=4)
    with pytest.raises(ValueError):
        fit(x, y, seed=0, steps=4)


def test_matched_effect_has_separate_schema_and_exact_zero_boundary():
    rng = np.random.default_rng(7)
    zero = rng.normal(size=(3, WIDTH))
    zero[:, 12:16] = 0
    active = zero.copy()
    active[:, 12] = 1
    baseline = rng.normal(size=(3, 3))
    changed = baseline + np.array([0.1, 0.02, 0.0])
    model = fit_matched_effect(zero, active, baseline, changed, seed=2027, steps=4)
    assert model["optimizer_steps"] == 12
    assert model["training_matched_physical_pairs"] == 3
    assert model["mathematical_zero_identity_rows"] == 3
    assert model["identity_rows_are_new_physical_examples"] is False
    assert (
        model["target"]
        == "mirrored_actual_paired_total_controller_conditioned_action_effect_80ms_m_s"
    )
    np.testing.assert_array_equal(predict_matched_effect(model, zero), np.zeros((3, 3)))
    assert np.isfinite(predict_matched_effect(model, active)).all()
    with pytest.raises(ValueError):
        predict(model, active)


def test_mismatched_partial_states_or_zero_training_action_rejected():
    zero = np.zeros((2, WIDTH))
    active = zero.copy()
    active[:, 12] = 1
    y = np.zeros((2, 3))
    changed = active.copy()
    changed[0, 0] = 1
    with pytest.raises(ValueError):
        fit_matched_effect(zero, changed, y, y, seed=1019, steps=4)
    with pytest.raises(ValueError):
        fit_matched_effect(zero, zero, y, y, seed=1019, steps=4)
    with pytest.raises(ValueError):
        fit_matched_effect(zero, active, y, y[:, 0], seed=1019, steps=4)


def test_resampling_preserves_physical_count_and_zero_identity():
    zero = np.zeros((2, WIDTH))
    active = zero.copy()
    active[:, 12] = 1
    y = np.zeros((2, 3))
    indices = np.array([0, 1, 1], dtype=np.int64)
    model = fit_matched_effect(
        zero, active, y, y + 0.1, seed=1019, steps=2, resample_indices=indices
    )
    assert model["training_matched_physical_pairs"] == 2
    assert model["training_resampled_pair_rows"] == 3
    assert model["mathematical_zero_identity_rows"] == 3
    assert model["resampling_duplicates_are_new_physical_examples"] is False
    assert model["optimizer_steps"] == 6
    np.testing.assert_array_equal(predict_matched_effect(model, zero), np.zeros((2, 3)))


@pytest.mark.parametrize(
    "indices",
    [np.array([0, 0]), np.array([0, 2]), np.array([0.0, 1.0]), np.array([[0, 1]])],
)
def test_resampling_cannot_drop_or_invent_a_training_pair(indices):
    zero = np.zeros((2, WIDTH))
    active = zero.copy()
    active[:, 12] = 1
    y = np.zeros((2, 3))
    with pytest.raises(ValueError):
        fit_matched_effect(zero, active, y, y, seed=1019, steps=2, resample_indices=indices)
