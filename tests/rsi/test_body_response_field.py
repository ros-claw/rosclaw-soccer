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


def test_explicit_early_contact_profile_never_precedes_measured_contact():
    guidance = BodyResponseRecoveryProposal(recovery_models(), active_contact_phases=(1, 2))
    batch = recovery_inputs()
    assert guidance.propose(**dict(batch, contact_phase=0))["active"] is False
    for phase in (1, 2):
        result = guidance.propose(**dict(batch, contact_phase=phase))
        assert result["active"] and result["contact_phase"] == phase
        assert max(abs(v) for v in result["target_increment"]) <= 0.002
    for phases in ((0, 1, 2), (True, 2), [1, 2], (1,)):
        with pytest.raises(ValueError):
            BodyResponseRecoveryProposal(recovery_models(), active_contact_phases=phases)


def test_full29_body_response_predicts_arm_intervention_without_executing_it():
    old, new = models()
    old["layers"][0]["weight"][0][1000] = 0.0
    old["layers"][0]["weight"][0][1026] = 1.0
    parts = [copy.deepcopy(new) for _ in range(3)]
    for p, n in zip(parts, (12, 12, 5), strict=True):
        p["target_mean"] = np.zeros(35 * n).tolist()
        p["target_scale"] = np.ones(35 * n).tolist()
        p["layers"][-1] = dict(
            weight=np.zeros((35 * n, 64)).tolist(), bias=np.zeros(35 * n).tolist()
        )
    parts[2]["layers"][-1]["bias"][2] = 1.0
    for value in (old, *parts):
        value.pop("model_hash")
        value["model_hash"] = reference._hash(value)
    field = BodyResponseField(old, parts, implementation="compiled", action_dimensions=29)
    batch = inputs()
    batch["target_increment"] = np.zeros((2, 29))
    assert not np.any(field.predict_effect(**batch))
    batch["target_increment"][:, 26] = 0.01
    result = field.predict_effect(**batch)
    expected = 0.5 * float(np.tanh(np.tanh(np.float32(0.01)))) + 0.005
    np.testing.assert_allclose(result[:, 0], expected, atol=1e-9, rtol=0)
    assert field.contract()["target_increment_dimensions"] == 29
    assert field.contract()["motor_policy"] is False
    with pytest.raises(ValueError):
        BodyResponseField(old, parts)


def test_full29_recovery_proposal_can_use_arm_response_with_same_strict_bounds():
    old, new = recovery_models()[0]
    old["layers"][0]["weight"][0][1000] = 0.0
    old["layers"][0]["weight"][0][1026] = 1.0
    parts = [copy.deepcopy(new) for _ in range(3)]
    for part, width in zip(parts, (12, 12, 5), strict=True):
        part["target_mean"] = np.zeros(35 * width).tolist()
        part["target_scale"] = np.ones(35 * width).tolist()
        part["layers"][-1] = dict(
            weight=np.zeros((35 * width, 64)).tolist(), bias=np.zeros(35 * width).tolist()
        )
    parts[2]["layers"][-1]["bias"][3 * 5 + 2] = 1.0
    for value in (old, *parts):
        value.pop("model_hash")
        value["model_hash"] = reference._hash(value)
    pairs = [(copy.deepcopy(old), copy.deepcopy(parts)) for _ in range(4)]
    proposal = BodyResponseRecoveryProposal(pairs, action_dimensions=29)
    batch = recovery_inputs()
    batch["previous_increment"] = np.zeros((1, 29))
    result = proposal.propose(**batch)
    assert result["active"] and not result["fallback"]
    assert len(result["target_increment"]) == 29
    assert result["target_increment"][26] == -0.002
    assert result["target_increment"][:12] == [0.0] * 12
    assert result["predicted_candidate_cost"] < result["predicted_baseline_cost"]
    assert proposal.contract()["target_increment_dimensions"] == 29
    assert proposal.contract()["runtime_execution_authorized"] is False
    for kwargs in (dict(contact_phase=0), dict(protected=True)):
        blocked = proposal.propose(**dict(batch, **kwargs))
        assert blocked["target_increment"] == [0.0] * 29 and not blocked["active"]
    bad = dict(batch, previous_increment=np.zeros((1, 12)))
    assert proposal.propose(**bad)["fallback"]
    for dimension in (True, 13, 28):
        with pytest.raises(ValueError):
            BodyResponseRecoveryProposal(pairs, action_dimensions=dimension)
    with pytest.raises(ValueError):
        BodyResponseRecoveryProposal(pairs)


def test_recovery_execution_causal_parent_protection_and_bundle(monkeypatch):
    from types import SimpleNamespace

    from rosclaw_soccer.rsi import body_response_guidance_execution as module
    from rosclaw_soccer.sim.contracts import hash_json

    parent_hash = "sha256:" + "a" * 64
    policy = dict(
        proposal_memory_motor_proof={},
        step_motor_proof=dict(
            model=dict(schema="soccer.rsi.proposal_memory_motor.v1", model_hash=parent_hash)
        ),
    )
    bundle = module.make_bundle(parent_hash, recovery_models())
    execution = module.BodyResponseGuidanceExecution(bundle, policy)
    independent = module.BodyResponseGuidanceExecution(bundle, policy)
    monkeypatch.setattr(module, "features_at_frame", lambda *args, **kwargs: np.zeros(134))
    guard = SimpleNamespace(gate=lambda context: 1.0)
    decoder = SimpleNamespace(features=lambda state: state, _guard=guard)
    q = recovery_inputs()["qpos"]
    body = dict(
        canonical_qpos=np.repeat(q[None], 32, axis=0),
        canonical_qvel=np.zeros((32, 1, 41)),
        foundation_neural_decoder_input=np.zeros((32, 1, 994)),
        force_n=np.zeros((32, 1, 6)),
        ball_position_before_step_m=np.zeros((32, 1, 3)),
        root_pose_xyzw_m=np.zeros((32, 1, 7)),
        ball_linear_velocity_before_step_m_s=np.zeros((32, 1, 3)),
        root_velocity_world=np.zeros((32, 1, 6)),
    )
    body["force_n"][0, 0, 0] = 2
    previous = np.zeros(12)
    for frame in range(31):
        args = dict(
            frame=frame,
            nominal_target=np.zeros(29),
            parent_delta=np.zeros(12),
            previous_final=previous,
            limits=np.tile([-1.0, 1.0], (12, 1)),
        )
        actual, status = execution.advance(decoder, body, **args)
        expected, reference_status = independent.advance(decoder, body, **args)
        np.testing.assert_array_equal(actual, expected)
        assert status == reference_status
        if frame < 30:
            assert not status["active"] and not np.any(actual)
        else:
            assert status["active"] and actual[0] == -0.002
        previous = actual
    guard.gate = lambda context: 0.0
    actual, status = execution.advance(
        decoder,
        body,
        frame=31,
        nominal_target=np.zeros(29),
        parent_delta=np.zeros(12),
        previous_final=previous,
        limits=np.tile([-1.0, 1.0], (12, 1)),
    )
    assert status["protected"] and not np.any(actual)
    for key, value in (
        ("hardware_authorized", True),
        ("parent_model_hash", "sha256:" + "b" * 64),
        ("source_hash", "changed"),
    ):
        bad = copy.deepcopy(bundle)
        bad[key] = value
        bad["bundle_hash"] = hash_json({k: v for k, v in bad.items() if k != "bundle_hash"})
        with pytest.raises(ValueError):
            module.BodyResponseGuidanceExecution(bad, policy)
    with pytest.raises(ValueError):
        module.BodyResponseGuidanceExecution(bundle, dict(step_motor_proof=None))
