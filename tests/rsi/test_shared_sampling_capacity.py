import pytest

from scripts.rsi_shared_sampling_capacity import shared_sampling_budget


def review():
    return dict(
        schema="soccer.rsi.shared_sampling_transport_review.v1",
        mean_model_hash="fixture-mean",
        complete_sampling_payload_equal=True,
        all_physical_trace_arrays_equal=True,
        physical_outcome_comparison={"measured_physical_fields_equal": True},
        new_physical_executions=2,
        motor_frames_reconstructed=600,
        largest_execution_bytes=1000,
        largest_native_log_bytes=500,
        shared_whole_model_bytes=2000,
        sampling_envelope_bytes=100,
        promotion_authorized=False,
        hardware_authorized=False,
    )


def test_all_234_runs_and_208_views_plus_next_stage_are_budgeted():
    expected = ((234 * 1500 + 2000 + 208 * 100) * 125 + 99) // 100 + 512 * 1024**2 + 2 * 1024**3
    assert (
        shared_sampling_budget(review(), mean_hash="fixture-mean", executions=234, views=208)
        == expected
    )


@pytest.mark.parametrize(
    "key,value",
    [
        ("complete_sampling_payload_equal", False),
        ("motor_frames_reconstructed", 300),
        ("largest_native_log_bytes", 0),
        ("shared_whole_model_bytes", True),
        ("promotion_authorized", True),
        ("mean_model_hash", "another-mean"),
    ],
)
def test_incomplete_or_other_policy_transport_cannot_authorize_collection(key, value):
    value_review = review()
    value_review[key] = value
    with pytest.raises(ValueError):
        shared_sampling_budget(value_review, mean_hash="fixture-mean", executions=234, views=208)
