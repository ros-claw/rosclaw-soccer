import copy

import numpy as np
import pytest
from rosclaw.growth.causal_residual_memory import initial_parameters

from rosclaw_soccer.rsi.reference_student_decoder import ReferenceStudentDecoder
from tests.rsi.test_pre_parent_reference_features import measured_trace


def decoder(parameters=None):
    return ReferenceStudentDecoder(
        initial_parameters(135, 12, hidden_dimension=64, seed=7)
        if parameters is None
        else parameters,
        mean=np.zeros(135),
        scale=np.ones(135),
        world_hash="sha256:" + "a" * 64,
        parent_model_hash="sha256:" + "b" * 64,
    )


def boundary():
    base = np.zeros(29, dtype=np.float64)
    base[::2] = -0.0
    bounds = np.tile([-2.0, 2.0], (29, 1))
    return base, bounds


def test_zero_gru_preserves_all_original_target_bits_without_physical_claim():
    policy, trace = decoder(), measured_trace()
    base, bounds = boundary()
    for frame in range(300):
        record = policy.advance(trace, frame=frame, navigation_target=base, joint_limits=bounds)
        assert record["pre_parent_joint_target_rad"].tobytes() == base.tobytes()
        assert not record["applied_reference_increment_rad"].any()
        assert record["input_features"][-1] == (300 - frame) / 300
    assert policy.next_frame == 300
    contract = policy.contract()
    assert not contract["physics_qualified"] and not contract["training_provenance_authenticated"]
    assert contract["parent_must_recompute_after_reference"]
    assert contract["architecture"] == [135, 64, 12]


def test_nonzero_numeric_probe_is_bounded_and_uses_own_projected_past_reference():
    parameters = initial_parameters(135, 12, seed=7)
    parameters["head_bias"] = [2.0] * 12  # Numeric test probe, NOT learned physical weights.
    policy, trace = decoder(parameters), measured_trace()
    base, bounds = boundary()
    bounds[0] = [-2.0, 0.02]
    base[1] = 3.0  # Already bad Foundation target: never enlarge its violation.
    for frame in range(32):
        record = policy.advance(trace, frame=frame, navigation_target=base, joint_limits=bounds)
        if frame < 30:
            assert not record["applied_reference_increment_rad"].any()
        else:
            assert record["applied_reference_increment_rad"][0] == 0.02
            assert record["applied_reference_increment_rad"][1] == 0
            assert np.max(np.abs(record["applied_reference_increment_rad"])) <= 0.35
        assert record["pre_parent_joint_target_rad"][12:].tobytes() == base[12:].tobytes()
    names = policy.contract()["feature_names"]
    previous = [
        i for i, n in enumerate(names) if n.startswith("previous_applied_reference_increment:")
    ]
    assert record["input_features"][previous[0]] == 0.02
    assert record["input_features"][previous[1]] == 0
    assert not policy.contract()["runtime_execution_authorized"]


def test_numeric_weights_normalization_and_records_are_owned_per_episode():
    parameters = initial_parameters(135, 12, seed=7)
    original = copy.deepcopy(parameters)
    policy, trace = decoder(parameters), measured_trace()
    parameters["head_bias"][0] = 2.0
    base, bounds = boundary()
    record = policy.advance(trace, frame=0, navigation_target=base, joint_limits=bounds)
    record["recurrent_hidden_state"][:] = 100
    fresh = decoder(original)
    policy_record = policy.advance(trace, frame=1, navigation_target=base, joint_limits=bounds)
    fresh.advance(trace, frame=0, navigation_target=base, joint_limits=bounds)
    fresh_record = fresh.advance(trace, frame=1, navigation_target=base, joint_limits=bounds)
    np.testing.assert_array_equal(
        policy_record["recurrent_hidden_state"], fresh_record["recurrent_hidden_state"]
    )
    assert not policy_record["requested_reference_increment_rad"].any()


@pytest.mark.parametrize("tamper", ["skip", "bool", "nan", "limit", "dtype"])
def test_invalid_input_does_not_consume_recurrent_frame(tamper):
    policy, trace = decoder(), measured_trace()
    base, bounds = boundary()
    frame = 1 if tamper == "skip" else False if tamper == "bool" else 0
    if tamper == "nan":
        base[0] = np.nan
    elif tamper == "limit":
        bounds[0] = [2, -2]
    elif tamper == "dtype":
        base = base.astype(np.float32)
    with pytest.raises(ValueError):
        policy.advance(trace, frame=frame, navigation_target=base, joint_limits=bounds)
    assert policy.next_frame == 0
