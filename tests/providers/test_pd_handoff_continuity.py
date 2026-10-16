import numpy as np
import pytest

from rosclaw_soccer.providers.g1.pd_handoff_continuity import PDHandoffContinuityOffset


def arguments(seed=1):
    rng = np.random.default_rng(seed)
    return dict(
        joint_position=rng.normal(size=29),
        joint_velocity=rng.normal(size=29),
        previous_target=rng.normal(size=29),
        contact_target=rng.normal(size=29),
        old_kp=rng.uniform(20, 100, 29),
        old_kd=rng.uniform(0, 5, 29),
        new_kp=rng.uniform(20, 100, 29),
        new_kd=rng.uniform(0, 5, 29),
    )


@pytest.mark.parametrize("seed", range(8))
def test_entry_torque_matches_old_pd_and_original_clipping(seed):
    values = arguments(seed)
    bridge = PDHandoffContinuityOffset(**values)
    target, offset = bridge.propose(relative_frame=0, contact_target=values["contact_target"])
    q, v = values["joint_position"], values["joint_velocity"]
    old = values["old_kp"] * (values["previous_target"] - q) - values["old_kd"] * v
    new = values["new_kp"] * (target - q) - values["new_kd"] * v
    np.testing.assert_allclose(new, old, atol=1e-12, rtol=0)
    np.testing.assert_allclose(np.clip(new, -25, 25), np.clip(old, -25, 25), atol=1e-12, rtol=0)
    np.testing.assert_array_equal(target, values["contact_target"] + offset)


def test_fixed_release_and_no_caller_aliasing():
    values = arguments()
    original = values["contact_target"].copy()
    bridge = PDHandoffContinuityOffset(**values)
    for value in values.values():
        value[:] = 999
    norms = []
    for frame in range(30):
        raw = original + frame * 0.001
        target, offset = bridge.propose(relative_frame=frame, contact_target=raw)
        norms.append(np.linalg.norm(offset))
        if frame >= 20:
            np.testing.assert_array_equal(offset, np.zeros(29))
            np.testing.assert_array_equal(target, raw)
    assert all(b <= a + 1e-12 for a, b in zip(norms, norms[1:], strict=False))


@pytest.mark.parametrize("key", list(arguments()))
def test_nonfinite_or_wrong_dtype_rejected(key):
    values = arguments()
    values[key] = np.full(29, np.nan)
    with pytest.raises(ValueError):
        PDHandoffContinuityOffset(**values)
    values = arguments()
    values[key] = values[key].astype(np.float32)
    with pytest.raises(ValueError):
        PDHandoffContinuityOffset(**values)


@pytest.mark.parametrize(
    "key,value", [("old_kp", 0.0), ("new_kp", -1.0), ("old_kd", -1.0), ("new_kd", -1.0)]
)
def test_invalid_gains_rejected(key, value):
    values = arguments()
    values[key][:] = value
    with pytest.raises(ValueError):
        PDHandoffContinuityOffset(**values)


def test_invalid_sequence_faults_without_silent_restart():
    values = arguments()
    bridge = PDHandoffContinuityOffset(**values)
    with pytest.raises(ValueError):
        bridge.propose(relative_frame=1, contact_target=values["contact_target"])
    with pytest.raises(ValueError, match="faulted"):
        bridge.propose(relative_frame=0, contact_target=values["contact_target"])


def test_entry_target_cannot_be_replaced_after_binding():
    values = arguments()
    bridge = PDHandoffContinuityOffset(**values)
    with pytest.raises(ValueError):
        bridge.propose(relative_frame=0, contact_target=values["contact_target"] + 0.01)


def test_contract_has_no_learning_or_physical_safety_claim():
    contract = PDHandoffContinuityOffset.contract()
    assert contract["release_frames"] == 20
    assert not contract["stability_proven"]
    assert not contract["neural_training"]
    assert not contract["hardware_authorized"]
