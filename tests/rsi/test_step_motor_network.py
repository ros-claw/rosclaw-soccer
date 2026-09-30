import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES
from rosclaw_soccer.rsi.step_motor_network import fit_model, predict_delta, validate_model
from rosclaw_soccer.sim.contracts import hash_json


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    rng = np.random.default_rng(314)
    n = 500
    arrays = dict(
        observation=rng.normal(size=(n, 134)),
        applied_delta_rad=np.full((n, 12), 0.02),
        terminal_return_label=np.ones(n),
        actor_supervision_mask=np.ones(n, dtype=bool),
        trajectory_index=np.repeat(np.arange(10), 50),
    )
    manifest = dict(
        schema="soccer.rsi.causal_step_motor_teacher_bank.v1",
        partition="TRAIN_CONSUMED",
        feature_names=list(FEATURE_NAMES),
        sample_count=n,
        trajectories=[dict(index=i, seed=20264000 + i) for i in range(10)],
    )
    manifest["report_hash"] = hash_json(manifest)
    return fit_model(manifest, arrays, epochs=2)


def test_grouped_warm_start_remains_unqualified(model):
    assert model["actor"]["dimensions"] == [134, 128, 128, 12]
    assert not set(model["metrics"]["validation_seed_clusters"]) & set(
        model["metrics"]["training_seed_clusters"]
    )
    assert model["physics_qualified"] is False
    assert model["online_rl_qualified"] is False
    validate_model(model)


def test_neural_output_remains_bounded_and_zero_before_decision(model):
    kwargs = dict(
        baseline=np.zeros(12), limits=np.tile([-1.0, 1.0], (12, 1)), previous=np.zeros(12)
    )
    assert np.array_equal(predict_delta(model, np.zeros(134), frame=29, **kwargs), np.zeros(12))
    delta = predict_delta(model, np.zeros(134), frame=30, **kwargs)
    assert delta.shape == (12,)
    assert np.max(np.abs(delta)) <= 0.012
    kwargs["previous"] = delta
    again = predict_delta(model, np.zeros(134), frame=31, **kwargs)
    assert np.max(np.abs(again - delta)) <= 0.012 + 1e-12


def test_invalid_baseline_is_never_pushed_further_outside_limits(model):
    delta = predict_delta(
        model,
        np.zeros(134),
        frame=30,
        baseline=np.full(12, 1.2),
        limits=np.tile([-1.0, 1.0], (12, 1)),
        previous=np.zeros(12),
    )
    assert np.all(delta <= 0)


@pytest.mark.parametrize(
    "field", ["physics_qualified", "hardware_authorized", "runtime_execution_authorized"]
)
def test_resealed_artifact_cannot_grant_authority(model, field):
    changed = copy.deepcopy(model)
    changed[field] = True
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_model(changed)


def test_nonfinite_observation_and_predecision_action_fail_closed(model):
    kwargs = dict(
        baseline=np.zeros(12), limits=np.tile([-1.0, 1.0], (12, 1)), previous=np.zeros(12)
    )
    with pytest.raises(ValueError):
        predict_delta(model, np.full(134, float("nan")), frame=30, **kwargs)
    kwargs["previous"] = np.ones(12) * 0.01
    with pytest.raises(ValueError):
        predict_delta(model, np.zeros(134), frame=20, **kwargs)
