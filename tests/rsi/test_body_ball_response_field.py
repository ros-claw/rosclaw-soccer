import hashlib
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi.body_ball_response_field import BodyBallResponseField


def models():
    result = []
    for output in (38, 456, 456, 190):
        value = dict(
            schema="rosclaw.growth.context_prediction_mlp.v1",
            source_hash="sha256:"
            + hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
            input_mean=np.zeros(112).tolist(),
            input_scale=np.ones(112).tolist(),
            target_mean=np.zeros(output).tolist(),
            target_scale=np.ones(output).tolist(),
            layers=[
                dict(weight=np.zeros((b, a)).tolist(), bias=np.zeros(b).tolist())
                for a, b in ((112, 128), (128, 64), (64, output))
            ],
            prediction_only=True,
            activation_ceiling="SIM_ONLY",
            motor_policy=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )
        result.append(value)
    baseline = result[0]
    baseline["target_mean"][3] = 0.1
    baseline["layers"][0]["weight"][0][100] = 1
    baseline["layers"][1]["weight"][0][0] = 1
    baseline["layers"][2]["weight"][35][0] = 1
    result[-1]["layers"][-1]["bias"][35 * 5 + 2] = 1
    for value in result:
        value["model_hash"] = reference._hash(value)
    return result


def test_body_ball_field_zero_and_arm_induced_ball_response_reference_exact():
    values = models()
    field = BodyBallResponseField(values[0], values[1:])
    compiled = BodyBallResponseField(values[0], values[1:], implementation="compiled")
    q, v, target = np.zeros((2, 43)), np.zeros((2, 41)), np.zeros((2, 29))
    q[:, 3] = q[:, 39] = 1
    v[:, 35] = 1
    args = dict(qpos=q, qvel=v, nominal_target=target)
    predicted = field.predict_next_velocity(**args)
    assert not predicted.flags.writeable
    np.testing.assert_array_equal(predicted[:, 3], [0.1, 0.1])
    np.testing.assert_array_equal(predicted[:, 35], [1, 1])
    np.testing.assert_array_equal(predicted, compiled.predict_next_velocity(**args))
    delta = np.zeros((2, 29))
    assert not np.any(field.predict_effect(**args, target_increment=delta))
    delta[:, 26] = 0.01
    actual = field.predict_effect(**args, target_increment=delta)
    np.testing.assert_array_equal(actual, compiled.predict_effect(**args, target_increment=delta))
    expected = 0.5 * float(np.tanh(np.tanh(np.float32(0.01)))) + 0.005
    np.testing.assert_allclose(actual[:, 35], expected, atol=1e-9, rtol=0)
    assert not np.any(actual[:, :35]) and not actual.flags.writeable
    values[0]["hardware_authorized"] = True
    np.testing.assert_array_equal(actual, field.predict_effect(**args, target_increment=delta))
    assert field.contract()["motor_policy"] is False
    for bad in (np.full((2, 29), 0.021), np.zeros((2, 12))):
        with pytest.raises(ValueError):
            field.predict_effect(**args, target_increment=bad)
    with pytest.raises(ValueError):
        BodyBallResponseField(values[0], values[1:])
