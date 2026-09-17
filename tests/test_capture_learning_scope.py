"""Actual execution eligibility is not a tactical intent or a success label."""

import numpy as np
import pytest

from rosclaw_soccer.training.capture_learning_scope import capture_learning_segments


def trace():
    n = 12
    context = np.zeros((n, 2), dtype=bool)
    context[4:7, 0] = True
    context[10:, 1] = True
    active = context.copy()
    active[5, 0] = False
    return {
        "time": np.arange(n) * 0.02,
        "training_return_event_code": np.array([0, 1, 2, 3, 0, 0, 0, 1, 2, 3, 0, 0]),
        "post_receive_capture_context": context,
        "capture_live_foundation_active": context.copy(),
        "residual_active": active,
        "residual_observations": np.ones((n, 2, 56)),
    }


def prepare(value, exploration=None):
    if exploration is None:
        exploration = value["post_receive_capture_context"].copy()
    return capture_learning_segments(value, exploration_mask=exploration)


def test_preserves_measured_masks_and_excludes_reset_gaps():
    value = trace()
    original = {k: v.copy() for k, v in value.items()}
    parts, counts = prepare(value)
    assert [len(p["time"]) for p in parts] == [4, 3]
    assert counts["eligible_samples_by_column"] == [2, 2]
    assert counts["exploration_samples_by_column"] == [3, 2]
    assert counts["eligible_samples"] == 4
    # Exploration on a physically inactive frame remains recorded, but cannot train.
    assert parts[0]["residual_exploration_mask"][2, 0]
    assert not parts[0]["residual_active"][2, 0]
    for key in value:
        np.testing.assert_array_equal(value[key], original[key])
    parts[0]["post_receive_capture_context"][:] = False
    np.testing.assert_array_equal(
        value["post_receive_capture_context"], original["post_receive_capture_context"]
    )


def test_inactive_foundation_is_not_silently_used_as_live_training():
    value = trace()
    value["capture_live_foundation_active"][4, 0] = False
    with pytest.raises(ValueError, match="live foundation"):
        prepare(value)


@pytest.mark.parametrize(
    "field", ["post_receive_capture_context", "capture_live_foundation_active", "residual_active"]
)
@pytest.mark.parametrize("fault", ["numeric", "shape", "missing"])
def test_masks_require_explicit_aligned_booleans(field, fault):
    value = trace()
    exploration = value["post_receive_capture_context"].copy()
    if fault == "numeric":
        value[field] = value[field].astype(float)
    elif fault == "shape":
        value[field] = value[field][:-1]
    else:
        del value[field]
    with pytest.raises(ValueError):
        prepare(value, exploration)


def test_no_exploration_outside_actual_capture():
    value = trace()
    mask = value["post_receive_capture_context"].copy()
    mask[6, 1] = True
    with pytest.raises(ValueError, match="capture"):
        prepare(value, mask)


def test_no_exploration_during_external_throw():
    value = trace()
    value["post_receive_capture_context"][2, 0] = True
    value["capture_live_foundation_active"][2, 0] = True
    with pytest.raises(ValueError, match="returned-live"):
        prepare(value)


def test_rejects_false_foundation_context():
    value = trace()
    value["capture_live_foundation_active"][6, 1] = True
    with pytest.raises(ValueError, match="subset"):
        prepare(value)


def test_zero_samples_is_explicit_not_success():
    value = trace()
    _, counts = prepare(value, np.zeros((12, 2), bool))
    assert counts["eligible_samples"] == 0
    assert counts["eligible_samples_by_column"] == [0, 0]


def test_does_not_overwrite_preexisting_learning_or_sampling_labels():
    for key in ("residual_exploration_mask", "residual_learning_mask"):
        value = trace()
        value[key] = np.zeros((12, 2), bool)
        with pytest.raises(ValueError, match="already"):
            prepare(value)


def test_requires_observed_return_lifecycle():
    value = trace()
    del value["training_return_event_code"]
    with pytest.raises(ValueError, match="lifecycle"):
        prepare(value)


@pytest.mark.parametrize("fault", ["dtype", "shape", "time", "lifecycle"])
def test_invalid_exploration_clock_or_events_fail_closed(fault):
    value = trace()
    mask = value["post_receive_capture_context"].copy()
    if fault == "dtype":
        mask = mask.astype(float)
    elif fault == "shape":
        mask = mask[:-1]
    elif fault == "time":
        value["time"][4] = np.nan
    else:
        value["training_return_event_code"][2] = 3
    with pytest.raises(ValueError):
        prepare(value, mask)
