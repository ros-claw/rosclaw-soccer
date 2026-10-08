import numpy as np
import pytest

from rosclaw_soccer.rsi.sampling_frontier_diagnostics import motor_observation_boundaries
from rosclaw_soccer.rsi.step_motor_features import FEATURE_NAMES


def test_boundary_counts_preserve_original_feature_identity_and_phase():
    context = np.zeros((270, 135), dtype=np.float64)
    ball = FEATURE_NAMES.index("ball_relative_body:x")
    contact = FEATURE_NAMES.index("previous_ball_body_force:5")
    context[:5, ball] = 8
    context[5:8, ball] = -8
    context[9:11, contact] = 8
    context[10, 0] = 7.999999
    context[100:120, 134] = 1
    context[120:, 134] = 2
    original = context.copy()
    result = motor_observation_boundaries(context)
    np.testing.assert_array_equal(context, original)
    assert result["feature_names"] == list(FEATURE_NAMES)
    assert result["feature_coordinate_rows"] == 36180
    assert result["frames_with_any_boundary"] == 10
    assert result["features"][ball] == {
        "feature_name": "ball_relative_body:x",
        "positive_bound_frames": 5,
        "negative_bound_frames": 3,
        "minimum_recorded_value": -8.0,
        "maximum_recorded_value": 8.0,
    }
    assert result["features"][contact]["positive_bound_frames"] == 2
    assert result["features"][0]["positive_bound_frames"] == 0
    assert result["contact_phase_counts"] == {"0": 100, "1": 20, "2": 150}
    assert all(
        result[k] is False
        for k in (
            "raw_normalization_reconstructed",
            "clipped_information_loss_proven",
            "source_physics_validated_here",
            "runtime_selection_authorized",
            "training_authorized",
            "promotion_authorized",
            "hardware_authorized",
        )
    )


@pytest.mark.parametrize("shape", [(269, 135), (270, 134), (270, 136), (135,)])
def test_incomplete_or_wrong_feature_identity_is_rejected(shape):
    with pytest.raises(ValueError, match="normalized270x135"):
        motor_observation_boundaries(np.zeros(shape, dtype=np.float64))


@pytest.mark.parametrize("value", [np.nan, np.inf, -np.inf, 8.000001, -8.000001])
def test_nonfinite_or_outside_original_bound_is_rejected(value):
    context = np.zeros((270, 135), dtype=np.float64)
    context[0, 0] = value
    with pytest.raises(ValueError, match="normalized270x135"):
        motor_observation_boundaries(context)


@pytest.mark.parametrize("value", [0.5, 3, -1])
def test_invalid_contact_phase_is_rejected(value):
    context = np.zeros((270, 135), dtype=np.float64)
    context[0, 134] = value
    with pytest.raises(ValueError, match="normalized270x135"):
        motor_observation_boundaries(context)


def test_phase_reversal_and_silent_numeric_downcast_are_rejected():
    context = np.zeros((270, 135), dtype=np.float64)
    context[0, 134] = 2
    with pytest.raises(ValueError):
        motor_observation_boundaries(context)
    with pytest.raises(ValueError):
        motor_observation_boundaries(np.zeros((270, 135), dtype=np.float32))
