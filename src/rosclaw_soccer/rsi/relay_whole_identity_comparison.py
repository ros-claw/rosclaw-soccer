"""Strict trace-level comparison of zero whole-receiver and original motor."""

import numpy as np


def compare_receiver_traces(original: dict[str, np.ndarray], whole: dict[str, np.ndarray]) -> dict:
    required = {"control", "obs", "action", "command", "phase"}
    extra = {"context14", "reference_action", "realized_delta_radians", "contact_flags_before"}
    if set(original) != required or set(whole) != required | extra:
        raise ValueError("exact original and whole receiver trace schemas required")
    n = len(original["control"])
    if (
        n < 1
        or any(v.ndim < 1 or len(v) != n for v in (*original.values(), *whole.values()))
        or original["action"].shape != (n, 29)
        or whole["reference_action"].shape != (n, 29)
        or whole["realized_delta_radians"].shape != (n, 29)
    ):
        raise ValueError("fully aligned original and own receiver history required")
    return dict(
        common_fields_exact=all(np.array_equal(original[k], whole[k]) for k in required),
        action_is_actual_onnx_reference=np.array_equal(whole["action"], whole["reference_action"]),
        extra_joint_delta_exact_zero=bool(np.count_nonzero(whole["realized_delta_radians"]) == 0),
        controls=n,
    )
