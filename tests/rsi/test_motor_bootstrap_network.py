import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.motor_bootstrap_network import (
    actor_parameters,
    fit_bootstrap,
    validate_model,
)
from rosclaw_soccer.rsi.precontact_proprio_policy import FEATURE_NAMES
from rosclaw_soccer.sim.contracts import hash_json


def bank():
    samples, teachers = [], []
    for course in range(12):
        observation = [float(course) / 12] * 13
        action = [float(course) / 120] * 36 + [0.25 - course * 0.02]
        teachers.append(
            {"observation": observation.copy(), "teacher_motor_parameters": action.copy()}
        )
        for _candidate in range(32):
            samples.append(
                {
                    "observation": observation.copy(),
                    "motor_parameters": action.copy(),
                    "learning_labels": {
                        "reward": course / 12,
                        "high_quality": True,
                        "clean_foot_only": True,
                    },
                    "audit_metadata": {"course": [course, 0]},
                }
            )
    result = {
        "schema": "soccer.rsi.causal_motor_actor_critic_bootstrap_bank.v1",
        "partition": "TRAIN_CONSUMED",
        "sample_count": 384,
        "distinct_consumed_course_count": 12,
        "observation_names": list(FEATURE_NAMES),
        "critic_samples": samples,
        "actor_teacher_samples": teachers,
        "successful_teacher_course_count": 12,
        "runtime_selection_authorized": False,
        "promotion_authorized": False,
        "fresh_holdout_open_authorized": False,
    }
    result["report_hash"] = hash_json(result)
    return result


@pytest.fixture(scope="module")
def model():
    pytest.importorskip("torch")
    return fit_bootstrap(bank(), epochs=2)


def test_numerical_json_model_is_bounded_and_not_a_runtime_authorization(model):
    validate_model(model)
    for value in (-1000, 0, 1000):
        parameters = actor_parameters(model, tuple([value] * 13))
        assert parameters.shape == (37,)
        assert np.max(np.abs(parameters[:36])) <= 0.16
        assert -0.35 <= parameters[36] <= 0.25
    assert model["runtime_execution_authorized"] is False
    assert model["learning_kind"] == "offline_imitation_and_outcome_regression"
    assert model["evaluation_partition"] == "TRAIN_CONSUMED_ONLY"


def test_weights_or_authority_tamper_invalidates_model(model):
    for key, value in (("runtime_execution_authorized", True), ("learning_kind", "online RL")):
        changed = copy.deepcopy(model)
        changed[key] = value
        changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
        with pytest.raises(ValueError):
            validate_model(changed)
    changed = copy.deepcopy(model)
    changed["actor"]["layers"][0]["weight"][0][0] += 1
    with pytest.raises(ValueError):
        validate_model(changed)


def test_source_seal_does_not_override_invalid_numerical_architecture(model):
    changed = copy.deepcopy(model)
    changed["actor"]["scale"][0] = 0
    changed["model_hash"] = hash_json({k: v for k, v in changed.items() if k != "model_hash"})
    with pytest.raises(ValueError):
        validate_model(changed)


def test_incomplete_bank_cannot_fit_or_be_called_online_learning():
    data = bank()
    data["sample_count"] = 12
    data["report_hash"] = hash_json({k: v for k, v in data.items() if k != "report_hash"})
    with pytest.raises(ValueError):
        fit_bootstrap(data, epochs=2)
    with pytest.raises(ValueError):
        fit_bootstrap(bank(), epochs=0)


@pytest.mark.parametrize(
    "kind",
    [
        "unknown_teacher",
        "altered_context",
        "unbounded_teacher",
        "false_success",
        "inconsistent_replay",
    ],
)
def test_self_resealed_bank_cannot_override_physical_learning_contract(kind):
    data = bank()
    if kind == "unknown_teacher":
        data["actor_teacher_samples"][0]["observation"][0] = -999
    elif kind == "altered_context":
        data["critic_samples"][0]["observation"] = [-999.0] * 13
    elif kind == "unbounded_teacher":
        data["actor_teacher_samples"][0]["teacher_motor_parameters"][0] = 0.17
    elif kind == "inconsistent_replay":
        data["critic_samples"][0]["learning_labels"]["reward"] += 1
    else:
        for sample in data["critic_samples"]:
            sample["learning_labels"]["high_quality"] = False
    data["report_hash"] = hash_json({k: v for k, v in data.items() if k != "report_hash"})
    with pytest.raises(ValueError):
        fit_bootstrap(data, epochs=2)
