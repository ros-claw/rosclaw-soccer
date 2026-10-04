import copy
import hashlib
from pathlib import Path

import numpy as np
import pytest
from rosclaw.growth import context_prediction_mlp as reference

from rosclaw_soccer.rsi.body_response_field import BodyResponseField
from rosclaw_soccer.rsi.body_response_guidance import BodyResponseRecoveryProposal


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


def recovery_models():
    old, new = models()
    old["layers"][2]["weight"][0][0] = 0.0
    old["layers"][2]["weight"][3][0] = 1.0
    old["target_mean"][3] = 0.1
    new["layers"][2]["bias"][0] = 0.0
    new["layers"][2]["bias"][36] = 1.0
    for value in (old, new):
        value.pop("model_hash")
        value["model_hash"] = reference._hash(value)
    return [(copy.deepcopy(old), copy.deepcopy(new)) for _ in range(4)]


def recovery_inputs():
    batch = {k: v[:1].copy() for k, v in inputs().items() if k != "target_increment"}
    return dict(**batch, previous_increment=np.zeros((1, 12)), contact_phase=2, protected=False)


def test_recovery_proposal_bounds_protection_ownership_and_cost():
    pairs = recovery_models()
    guidance = BodyResponseRecoveryProposal(pairs)
    batch = recovery_inputs()
    before = copy.deepcopy(batch)
    result = guidance.propose(**batch)
    assert result["active"] and not result["fallback"]
    assert result["target_increment"][0] == -0.002
    assert max(abs(v) for v in result["target_increment"]) <= 0.002
    assert result["predicted_candidate_cost"] < result["predicted_baseline_cost"]
    assert result["runtime_execution_authorized"] is False
    for key in (
        "qpos",
        "qvel",
        "nominal_target",
        "relative_ball",
        "foundation_input",
        "previous_increment",
    ):
        np.testing.assert_array_equal(batch[key], before[key])
    pairs[0][0]["hardware_authorized"] = True
    assert guidance.propose(**batch) == result
    for phase, protected in ((0, False), (1, False), (2, True)):
        blocked = guidance.propose(**dict(batch, contact_phase=phase, protected=protected))
        assert blocked["target_increment"] == [0.0] * 12 and not blocked["active"]
    for key in ("qpos", "qvel", "foundation_input", "previous_increment"):
        bad = copy.deepcopy(batch)
        bad[key][0, 0] = np.nan
        rejected = guidance.propose(**bad)
        assert rejected["fallback"] and rejected["target_increment"] == [0.0] * 12
    with pytest.raises(ValueError):
        guidance.propose(**dict(batch, contact_phase=True))
    with pytest.raises(ValueError):
        BodyResponseRecoveryProposal(pairs[:3])


def test_recovery_rejects_dependency_changes(monkeypatch):
    from rosclaw_soccer.rsi import body_response_guidance

    guidance = BodyResponseRecoveryProposal(recovery_models())
    monkeypatch.setattr(body_response_guidance, "hash_bytes", lambda _: "changed")
    result = guidance.propose(**recovery_inputs())
    assert result["fallback"] and result["target_increment"] == [0.0] * 12
