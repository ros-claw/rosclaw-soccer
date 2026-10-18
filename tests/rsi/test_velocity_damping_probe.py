import dataclasses

import numpy as np
import pytest

from rosclaw_soccer.rsi.velocity_damping_probe import MODES, VelocityDampingProbe
from rosclaw_soccer.sim.contracts import G1_HARD_TORQUE_LIMITS, hash_json


@pytest.mark.parametrize("mode", MODES)
def test_fixed_feedback_law_and_per_joint_caps(mode):
    probe = VelocityDampingProbe(tuple(0.0 for _ in range(29)), mode)
    velocity = np.linspace(-20, 20, 29)
    original = velocity.copy()
    offset = probe.torque_offset(velocity)
    cap = np.minimum(2, 0.04 * np.asarray(G1_HARD_TORQUE_LIMITS))
    np.testing.assert_array_equal(offset, np.clip(-0.5 * velocity, -cap, cap))
    np.testing.assert_array_equal(velocity, original)
    assert np.max(np.abs(offset)) <= 2
    assert abs(offset[20]) <= 0.2  # wrist: 4% of 5 Nm, not a 2 Nm wrist impulse


def test_same_reference_and_measurement_give_same_first_action_for_both_rates():
    reference = tuple(float(value) for value in np.linspace(-1, 1, 29))
    velocity = np.linspace(1, -1, 29)
    held, feedback = (VelocityDampingProbe(reference, mode) for mode in MODES)
    np.testing.assert_array_equal(held.torque_offset(velocity), feedback.torque_offset(velocity))
    assert held.contract()["contract_hash"] != feedback.contract()["contract_hash"]
    with pytest.raises(dataclasses.FrozenInstanceError):
        held.mode = MODES[1]


def test_contract_is_sealed_local_and_does_not_claim_learning_or_authority():
    contract = VelocityDampingProbe((0.0,) * 29, MODES[0]).contract()
    assert contract["contract_hash"] == hash_json(
        {key: value for key, value in contract.items() if key != "contract_hash"}
    )
    assert contract["activation_ceiling"] == "SIM_ONLY"
    assert contract["local_diagnostic_only"] is True
    assert contract["trained_optimizer_updates"] == 0
    for key in (
        "learned",
        "privileged_contact_or_constraint_forces_are_inputs",
        "full_policy_or_recovery_qualified",
        "runtime_execution_authorized",
        "promotion_authorized",
        "hardware_authorized",
    ):
        assert contract[key] is False


@pytest.mark.parametrize("mode", [None, True, "REAL", "anything", 0])
def test_unknown_mode_is_rejected(mode):
    with pytest.raises(ValueError, match="fixed mode"):
        VelocityDampingProbe((0.0,) * 29, mode)


@pytest.mark.parametrize("reference", [(), [0.0] * 29, (0,) * 29, (True,) * 29])
def test_reference_requires_exact_immutable_floats(reference):
    with pytest.raises(ValueError, match="velocity reference"):
        VelocityDampingProbe(reference, MODES[0])


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1001.0])
def test_nonfinite_or_unbounded_reference_rejected(value):
    with pytest.raises(ValueError, match="velocity reference"):
        VelocityDampingProbe((value,) * 29, MODES[0])


@pytest.mark.parametrize("velocity", [np.zeros(28), np.zeros(29, dtype=np.float32), [0.0] * 29])
def test_measurement_shape_type_and_dtype_required(velocity):
    with pytest.raises(ValueError, match="velocity measurement"):
        VelocityDampingProbe((0.0,) * 29, MODES[0]).torque_offset(velocity)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), 1001.0])
def test_invalid_measurement_does_not_silently_clip_or_replace(value):
    with pytest.raises(ValueError, match="velocity measurement"):
        VelocityDampingProbe((0.0,) * 29, MODES[0]).torque_offset(np.full(29, value))


def test_constant_pd_target_shift_and_matched_torque_shift_are_algebraically_equivalent():
    rng = np.random.default_rng(20261018)
    limits = np.asarray(G1_HARD_TORQUE_LIMITS)
    for _ in range(100):
        target, q, velocity = rng.normal(size=(3, 29))
        kp, kd = rng.uniform(20, 100, 29), rng.uniform(1, 6, 29)
        delta = rng.uniform(-0.012, 0.012, 29)
        shifted_target = np.clip(kp * (target + delta - q) - kd * velocity, -limits, limits)
        shifted_torque = np.clip(kp * (target - q) - kd * velocity + kp * delta, -limits, limits)
        np.testing.assert_allclose(shifted_target, shifted_torque, rtol=0, atol=2e-12)
