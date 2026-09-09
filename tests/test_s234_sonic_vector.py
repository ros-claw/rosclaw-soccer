from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_runup import (
    _VARIANTS,
    G1SonicRunupConfig,
    G1SonicRunupController,
)
from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker


def setup(variant="sonic_v1_1", native=False):
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(234)
    ref = rng.normal(0, 0.1, (2, 80, 36))
    ref[:, :, 3:7] /= np.linalg.norm(ref[:, :, 3:7], axis=2, keepdims=True)
    q = np.zeros((2, 43))
    q[:, :36] = ref[:, 0]
    v = rng.normal(0, 0.1, (2, 41))
    model = SimpleNamespace(
        device=torch.device("cpu"),
        qualification=SimpleNamespace(
            reference_stride=_VARIANTS[variant].reference_stride,
            heading_normalized=_VARIANTS[variant].heading_normalized,
        ),
        encode_g1=lambda features: features[:, :64],
        decode=lambda observation: torch.zeros((len(observation), 29)),
    )
    return BatchedSonicTracker(model, ref, native_velocity=native), ref, q, v


@pytest.mark.parametrize("variant", ["sonic_v1_1", "low_latency"])
def test_batched_encoder_sensor_layout_matches_existing_cpu(variant):
    tracker, ref, q, v = setup(variant)
    actual = tracker.encoder_features(3, q, v).numpy()
    for i in range(2):
        old = object.__new__(G1SonicRunupController)
        old.reference = ref[i]
        old.config = G1SonicRunupConfig(model_variant=variant)
        old._variant = _VARIANTS[variant]
        expected = old._encoder_observation(SimpleNamespace(qpos=q[i], qvel=v[i]), 3)[0, 4:644]
        np.testing.assert_allclose(actual[i], expected, atol=3e-6, rtol=3e-6)


def test_histories_are_private_and_lifecycle_explicit():
    tracker, _, q, v = setup()
    with pytest.raises(RuntimeError):
        tracker.update(0, q, v)
    tracker.reset(q, v)
    tracker.update(0, q, v)
    with pytest.raises(RuntimeError):
        tracker.update(1, q, v)
    changed = q.copy()
    changed[0, 7] += 0.2
    tracker.observe(changed, v)
    assert tracker._history[-1][1][0, 0] != tracker._history[-2][1][0, 0]
    np.testing.assert_array_equal(tracker._history[-1][1][1], tracker._history[-2][1][1])
    with pytest.raises(RuntimeError):
        tracker.observe(q, v)
    tracker.update(1, q, v)


def test_reference_and_state_are_not_mutated():
    tracker, ref, q, v = setup()
    beforeq, beforeref = q.copy(), ref.copy()
    tracker.reset(q, v)
    tracker.update(0, q, v)
    np.testing.assert_array_equal(q, beforeq)
    np.testing.assert_array_equal(ref, beforeref)
    ref[:] = 0
    assert float(tracker.reference.abs().sum()) > 0


def test_bad_or_incomplete_inputs_fail_closed():
    tracker, _, q, v = setup()
    with pytest.raises(ValueError):
        tracker.encoder_features(79, q, v)
    with pytest.raises(ValueError):
        tracker.encoder_features(False, q, v)
    q[:, 3:7] = 0
    with pytest.raises(ValueError, match="normalized"):
        tracker.reset(q, v)


def test_native_velocity_is_dense_difference_before_subsampling():
    tracker, _, q, v = setup(native=True)
    actual = tracker.encoder_features(0, q, v).numpy()
    indices = np.arange(10) * 5
    reference = tracker.reference.numpy()
    from rosclaw_soccer.providers.g1.sonic_runup import MUJOCO_TO_ISAACLAB

    expected = (reference[:, indices + 1, 7:36] - reference[:, indices, 7:36]) / 0.02
    expected = expected[:, :, MUJOCO_TO_ISAACLAB].reshape(2, 290)
    np.testing.assert_allclose(actual[:, 290:580], expected, atol=1e-6)
