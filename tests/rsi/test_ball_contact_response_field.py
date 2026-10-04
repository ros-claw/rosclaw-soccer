import hashlib
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi.ball_contact_response_field import BallContactResponseField


def model(width=141):
    value = dict(
        schema="rosclaw.growth.context_prediction_mlp.v1",
        source_hash="sha256"
        + ":"
        + hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
        input_mean=np.zeros(width).tolist(),
        input_scale=np.ones(width).tolist(),
        target_mean=[5.0, 6.0, 7.0],
        target_scale=np.ones(3).tolist(),
        layers=[
            dict(weight=np.zeros((b, a)).tolist(), bias=np.zeros(b).tolist())
            for a, b in ((width, 128), (128, 64), (64, 3))
        ],
        prediction_only=True,
        activation_ceiling="SIM_ONLY",
        motor_policy=False,
        promotion_authorized=False,
        hardware_authorized=False,
    )
    value["layers"][0]["weight"][0][width - 29 + 8] = 1
    value["layers"][1]["weight"][0][0] = 1
    value["layers"][2]["weight"][0][0] = 1
    value["model_hash"] = reference._hash(value)
    return value


def test_action_conditioning_zero_effect_ownership_and_reference_parity():
    value = model()
    field = BallContactResponseField(value)
    compiled = BallContactResponseField(value, implementation="compiled")
    q, v, target = np.zeros((2, 43)), np.zeros((2, 41)), np.zeros((2, 29))
    q[:, 3] = q[:, 39] = 1
    args = dict(qpos=q, qvel=v, nominal_target=target)
    delta = np.zeros((2, 29))
    np.testing.assert_array_equal(
        field.predict_effect(**args, target_increment=delta), np.zeros((2, 3))
    )
    delta[:, 8] = [0.01, -0.02]
    result = field.predict_effect(**args, target_increment=delta)
    np.testing.assert_array_equal(result, compiled.predict_effect(**args, target_increment=delta))
    assert result[0, 0] > 0 and result[1, 0] < 0
    assert not result.flags.writeable and not np.any(result[:, 1:])
    value["hardware_authorized"] = True
    np.testing.assert_array_equal(result, field.predict_effect(**args, target_increment=delta))
    with pytest.raises(ValueError):
        BallContactResponseField(value)
    for bad in (np.full((2, 29), np.nan), np.full((2, 29), 0.02001), np.zeros((2, 12))):
        with pytest.raises(ValueError):
            field.predict_effect(**args, target_increment=bad)
    assert field.contract()["contact_jacobian_linearity_assumed"] is False


def test_explicit_geometry_representation_cannot_silently_change_input_contract():
    value = model(197)
    with pytest.raises(ValueError):
        BallContactResponseField(value)
    field = BallContactResponseField(value, contact_geometry=True)
    q, v, target = np.zeros((1, 43)), np.zeros((1, 41)), np.zeros((1, 29))
    q[:, 3] = q[:, 39] = 1
    args = dict(qpos=q, qvel=v, nominal_target=target, target_increment=target)
    with pytest.raises(ValueError):
        field.predict_effect(**args)
    assert not np.any(field.predict_effect(**args, foot_contact_features=np.zeros((1, 56))))
    assert field.contract()["input_dimensions"] == 197
    with pytest.raises(ValueError):
        BallContactResponseField(model()).predict_effect(
            **args, foot_contact_features=np.zeros((1, 56))
        )
