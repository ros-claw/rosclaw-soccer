import pytest

from scripts.rsi_review_omnicontact_native import validate_scope


def source():
    return dict(
        status="CFTRACK_NATIVE_BASELINE_RECORDED_NOT_PROMOTED",
        runtime_failure=None,
        promotion_authorized=False,
        hardware_authorized=False,
        controls=42,
        native_steps=420,
        new_neural_calls=42,
        reference_source="NPZmotion",
        reference_all_initial_velocities=True,
        reference_ball_initial_velocity=False,
        reference_phase_start=0,
        body_failed=True,
    )


def test_partial_body_failure_is_observed_not_promoted():
    assert validate_scope(source(), derived_moving_partial=True) == 42
    with pytest.raises(ValueError):
        validate_scope(source(), derived_moving_partial=False)


def test_original_full_horizon_scope_remains_supported():
    row = source() | dict(controls=600, native_steps=6000, new_neural_calls=600)
    assert validate_scope(row, derived_moving_partial=False) == 600


@pytest.mark.parametrize(
    "key,value",
    [
        ("controls", 0),
        ("controls", True),
        ("native_steps", 419),
        ("new_neural_calls", 41),
        ("runtime_failure", {"message": "bad"}),
        ("promotion_authorized", True),
        ("hardware_authorized", True),
        ("reference_all_initial_velocities", False),
        ("reference_ball_initial_velocity", True),
        ("reference_phase_start", 1),
    ],
)
def test_incomplete_or_different_protocol_rejected(key, value):
    row = source()
    row[key] = value
    with pytest.raises(ValueError):
        validate_scope(row, derived_moving_partial=True)
