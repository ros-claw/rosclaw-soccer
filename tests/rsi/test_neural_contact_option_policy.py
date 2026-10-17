import copy
import json

import numpy as np
import pytest

from rosclaw_soccer.rsi.initial_context_control_portfolio import ControlOutcome
from rosclaw_soccer.rsi.neural_contact_option_policy import (
    features,
    fit,
    select_option,
    validate_model,
)
from rosclaw_soccer.sim.contracts import hash_json


def data():
    contexts = ((2.25, 0.0, -0.4), (2.3, 0.0, 0.5), (2.65, 0.0, -0.3), (2.7, 0.0, 0.4))
    outcomes = tuple(
        tuple(ControlOutcome(j == i, j == i, 2.0 if j == i else -10.0) for j in range(8))
        for i in range(4)
    )
    return contexts, outcomes


def train(**kwargs):
    contexts, outcomes = data()
    return fit(
        contexts,
        outcomes,
        sample_weights=(1.0,) * 4,
        option_hashes=tuple(hash_json(i) for i in range(8)),
        training_data_hash=hash_json("synthetic fixture, not physics"),
        execution_binding_hash=hash_json("fixture world"),
        seed=7,
        steps=3,
        **kwargs,
    )


def test_real_optimizer_update_reload_and_no_physical_success_claim():
    import torch

    before = torch.random.get_rng_state().clone()
    model, report = train()
    assert torch.equal(before, torch.random.get_rng_state())
    assert report["optimizer_steps"] == 3 and report["parameters_changed"]
    assert report["initial_parameter_hash"] != report["final_parameter_hash"]
    assert report["numpy_reload_maximum_error"] < 1e-11
    assert report["resubstitution_only"] and not report["football_improvement_proven"]
    assert report["new_physical_executions"] == 0
    restored = json.loads(json.dumps(model))
    validate_model(restored)
    assert train()[0] == model
    query = select_option(
        restored,
        data()[0][0],
        option_hashes=tuple(model["option_hashes"]),
        execution_binding_hash=model["execution_binding_hash"],
    )
    assert 0 <= query["option_index"] < 8
    assert not query["evaluated_task_outcome_is_input"]
    assert not query["success_or_safety_guaranteed"]
    assert not query["scores_are_calibrated_success_probabilities"]


@pytest.mark.parametrize(
    "change",
    [
        dict(hardware_authorized=True),
        dict(optimizer_steps=True),
        dict(architecture=[3, 8]),
        dict(option_hashes=[{}] * 8),
    ],
)
def test_invalid_model_cannot_gain_authority_or_change_shape(change):
    model, _ = train()
    model.update(change)
    with pytest.raises(ValueError):
        validate_model(model)


def test_whole_model_seal_and_parameter_finiteness_required():
    model, _ = train()
    damaged = copy.deepcopy(model)
    damaged["layers"][0]["weight"][0][0] += 1
    with pytest.raises(ValueError, match="seal"):
        validate_model(damaged)
    damaged["layers"][0]["weight"][0][0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        validate_model(damaged)


@pytest.mark.parametrize(
    "context", [(2.5, 0.0, 0.4), (2.3, 0.0, float("nan")), (2.3, 0, 0.4), [2.3, 0.0, 0.4]]
)
def test_no_clipping_or_unknown_features(context):
    with pytest.raises(ValueError):
        features(context)


def test_catalog_and_world_binding_are_inference_inputs_not_exam_labels():
    model, _ = train()
    with pytest.raises(ValueError, match="binding"):
        select_option(
            model,
            data()[0][0],
            option_hashes=tuple(reversed(model["option_hashes"])),
            execution_binding_hash=model["execution_binding_hash"],
        )


def test_all_failure_rows_are_retained_without_nan_or_fabricated_success():
    contexts, outcomes = data()
    failed = tuple(ControlOutcome(False, False, -10.0) for _ in range(8))
    model, report = fit(
        contexts,
        (failed, *outcomes[1:]),
        sample_weights=(0.5, 0.5, 1.0, 1.0),
        option_hashes=tuple(hash_json(i) for i in range(8)),
        execution_binding_hash=hash_json("world"),
        training_data_hash=hash_json("synthetic"),
        seed=10,
        steps=2,
    )
    assert report["training_contexts"] == 4 and report["feasible_training_contexts"] == 3
    assert np.isfinite(report["last_loss"])
    assert not model["teacher_qualified"]
