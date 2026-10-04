import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi.body_response_field import BodyResponseField


def models():
    def model(width, output):
        layers = [
            dict(weight=np.zeros((b, a)).tolist(), bias=np.zeros(b).tolist())
            for a, b in ((width, 128), (128, 64), (64, output))
        ]
        return dict(
            schema="rosclaw.growth.context_prediction_mlp.v1",
            source_hash="sha256:"
            + hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
            input_mean=np.zeros(width).tolist(),
            input_scale=np.ones(width).tolist(),
            target_mean=np.zeros(output).tolist(),
            target_scale=np.ones(output).tolist(),
            layers=layers,
            prediction_only=True,
            activation_ceiling="SIM_ONLY",
            motor_policy=False,
            promotion_authorized=False,
            hardware_authorized=False,
        )

    old, new = model(1064, 35), model(103, 420)
    old["layers"][0]["weight"][0][1000] = 1.0
    old["layers"][1]["weight"][0][0] = 1.0
    old["layers"][2]["weight"][0][0] = 1.0
    new["layers"][2]["bias"][0] = 1.0
    for value in (old, new):
        value["model_hash"] = reference._hash(value)
    return old, new


def inputs():
    q = np.zeros((2, 43))
    q[:, 2] = 0.7
    q[:, 3] = 1
    return dict(
        qpos=q,
        qvel=np.zeros((2, 41)),
        nominal_target=np.zeros((2, 29)),
        relative_ball=np.zeros((2, 6)),
        foundation_input=np.zeros((2, 994)),
        target_increment=np.zeros((2, 12)),
    )


def test_exact_zero_fixed_blend_private_models_and_rejections():
    old, new = models()
    field = BodyResponseField(old, new)
    batch = inputs()
    assert np.array_equal(field.predict_effect(**batch), np.zeros((2, 35)))
    batch["target_increment"][:, 0] = 0.01
    baseline = copy.deepcopy(batch)
    actual = field.predict_effect(**batch)
    expected = 0.5 * float(np.tanh(np.tanh(np.float32(0.01)))) + 0.005
    np.testing.assert_allclose(actual[:, 0], expected, atol=1e-9, rtol=0)
    assert not actual.flags.writeable
    for key in batch:
        assert np.array_equal(batch[key], baseline[key])
    old["hardware_authorized"] = True
    assert np.array_equal(field.predict_effect(**batch), actual)
    assert field.contract()["motor_policy"] is False
    for value in (0.03, np.nan):
        bad = inputs()
        bad["target_increment"][:, 0] = value
        with pytest.raises(ValueError):
            field.predict_effect(**bad)
    with pytest.raises(ValueError):
        BodyResponseField(old, new)
