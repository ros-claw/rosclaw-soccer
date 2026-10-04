import numpy as np
import pytest

from rosclaw_soccer.providers.g1.sonic_prior_reframing import _rotation, reframe_low_latency_prior


def prior(root, futures):
    result = np.zeros(640, dtype=np.float32)
    result[:580] = np.linspace(-0.1, 0.1, 580)
    relative = np.stack([_rotation(root).T @ _rotation(q) for q in futures])
    result[580:634] = relative[:, [0, 0, 1, 1, 0, 1], [0, 1, 0, 1, 2, 2]].reshape(54)
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
