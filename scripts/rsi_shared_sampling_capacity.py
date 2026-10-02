"""Measured complete shared-data budget; never shrink a declared curriculum."""

from typing import Any


def shared_sampling_budget(
    review: dict[str, Any], *, mean_hash: str, executions: int, views: int
) -> int:
    if (
        review.get("schema") != "soccer.rsi.shared_sampling_transport_review.v1"
        or review.get("mean_model_hash") != mean_hash
        or review.get("complete_sampling_payload_equal") is not True
        or review.get("all_physical_trace_arrays_equal") is not True
        or review.get("physical_outcome_comparison", {}).get("measured_physical_fields_equal")
        is not True
        or type(review.get("new_physical_executions")) is not int
        or review["new_physical_executions"] != 2
        or type(review.get("motor_frames_reconstructed")) is not int
        or review["motor_frames_reconstructed"] != 600
        or any(review.get(k) is not False for k in ("promotion_authorized", "hardware_authorized"))
        or type(executions) is not int
        or type(views) is not int
        or not 1 <= views < executions <= 52 * 18
    ):
        raise ValueError("exact fully audited physical sampling transport required")
    keys = (
        "largest_execution_bytes",
        "largest_native_log_bytes",
        "shared_whole_model_bytes",
        "sampling_envelope_bytes",
    )
    if any(type(review.get(k)) is not int or not 1 <= review[k] <= 512 * 1024**2 for k in keys):
        raise ValueError("complete positive bounded measured storage fields required")
    # Both traces and logs per run, one complete mean, ALL sampling envelopes.
    # Add 25% growth, 512 MiB audit scratch, and 2 GiB next fit/validation reserve.
    measured = executions * (review[keys[0]] + review[keys[1]])
    measured += review[keys[2]] + views * review[keys[3]]
    return int((measured * 125 + 99) // 100 + 512 * 1024**2 + 2 * 1024**3)
