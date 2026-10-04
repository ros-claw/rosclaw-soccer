from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_prior_reframing import _rotation, reframe_low_latency_prior


def test_matches_actual_legacy_tracker_encoder_features():
    pytest.importorskip("torch")
    from rosclaw_soccer.providers.g1.sonic_vector import BatchedSonicTracker

    rng = np.random.default_rng(637)
    reference = rng.normal(size=(1, 20, 36)).astype(np.float32)
    reference[:, :, 3:7] /= np.linalg.norm(reference[:, :, 3:7], axis=2)[:, :, None]
    model = SimpleNamespace(
        device="cpu",
        variant="low_latency",
        qualification=SimpleNamespace(reference_stride=1, heading_normalized=False),
    )
    tracker = BatchedSonicTracker(model, reference, low_latency_legacy_encoder_layout=True)
    old_q = np.zeros((1, 43), dtype=np.float32)
    old_q[0, 3:7] = rng.normal(size=4)
    old_q[0, 3:7] /= np.linalg.norm(old_q[0, 3:7])
    new_q = old_q.copy()
    new_q[0, 3:7] = rng.normal(size=4)
    new_q[0, 3:7] /= np.linalg.norm(new_q[0, 3:7])
    velocity = np.zeros((1, 41), dtype=np.float32)
    original = tracker.encoder_features(0, old_q, velocity).numpy()[0]
    expected = tracker.encoder_features(0, new_q, velocity).numpy()[0]
    actual = reframe_low_latency_prior(original, old_q[0, 3:7], new_q[0, 3:7])
    assert np.array_equal(actual[:580], expected[:580])
    assert np.allclose(actual[580:], expected[580:], atol=5e-7, rtol=0)


def prior(root, futures):
    result = np.zeros(640, dtype=np.float32)
    result[:580] = np.linspace(-0.1, 0.1, 580)
    relative = np.stack([_rotation(root).T @ _rotation(q) for q in futures])
    result[580:634] = relative[:, [0, 0, 1, 1, 2, 2], [0, 1, 0, 1, 0, 1]].reshape(54)
    return result


def test_noncommuting_root_reframing_matches_original_world_references():
    rng = np.random.default_rng(18)
    old, new = rng.normal(size=(2, 4))
    old /= np.linalg.norm(old)
    new /= np.linalg.norm(new)
    futures = rng.normal(size=(9, 4))
    futures /= np.linalg.norm(futures, axis=1)[:, None]
    source = prior(old, futures)
    before = source.copy()
    actual = reframe_low_latency_prior(source, old, new)
    assert np.allclose(actual, prior(new, futures), atol=3e-7, rtol=0)
    assert np.array_equal(actual[:580], source[:580])
    assert np.array_equal(actual[634:], source[634:])
    assert not actual.flags.writeable
    assert np.array_equal(source, before)


def test_identical_and_equivalent_signed_roots_preserve_all_bits():
    root = np.asarray((1.0, 0.0, 0.0, 0.0))
    features = prior(root, np.tile(root, (9, 1)))
    assert np.array_equal(reframe_low_latency_prior(features, root, root), features)
    assert np.array_equal(reframe_low_latency_prior(features, root, -root), features)


@pytest.mark.parametrize("fault", ["nonunit", "nonfinite", "layout", "rotation"])
def test_rejects_unsupported_or_corrupted_prior(fault):
    root = np.asarray((1.0, 0.0, 0.0, 0.0))
    features = prior(root, np.tile(root, (9, 1)))
    if fault == "nonunit":
        root *= 2
    elif fault == "nonfinite":
        features[0] = np.nan
    elif fault == "layout":
        features[-1] = 1
    else:
        features[580] = 3
    with pytest.raises(ValueError):
        reframe_low_latency_prior(features, root, root)
