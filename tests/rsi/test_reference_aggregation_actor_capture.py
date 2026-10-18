"""Preserve the actor's actually executed past references, including roundoff."""

import numpy as np
import pytest

from rosclaw_soccer.rsi.pre_parent_reference_features import FEATURE_NAMES, trace_observations
from scripts.rsi_prepare_reference_state_aggregation import original_actor_rows
from tests.rsi.test_pre_parent_reference_features import measured_trace


def test_captured_past_action_is_not_overwritten_by_target_subtraction():
    trace = measured_trace()
    reconstructed = trace_observations(trace)["observation"]
    captured = reconstructed[:, None].copy()
    previous = [
        i
        for i, n in enumerate(FEATURE_NAMES)
        if n.startswith("previous_applied_reference_increment:")
    ]
    captured[41, 0, previous[0]] += 1e-16
    trace["reference_student_input_features"] = captured
    rows = original_actor_rows(trace, captured_required=True)
    assert rows["observation"].tobytes() == captured[:, 0].tobytes()
    assert rows["observation"].tobytes() != reconstructed.tobytes()
    rows["observation"][:] = 0
    assert captured[41, 0, previous[0]] != 0


@pytest.mark.parametrize("tamper", ["missing", "shape", "dtype", "state", "nan"])
def test_required_actual_actor_capture_cannot_fall_back(tamper):
    trace = measured_trace()
    captured = trace_observations(trace)["observation"][:, None]
    if tamper == "shape":
        captured = captured[:299]
    elif tamper == "dtype":
        captured = captured.astype(np.float32)
    elif tamper == "state":
        captured[41, 0, 0] += 1e-16
    elif tamper == "nan":
        captured[41, 0, 0] = np.nan
    if tamper != "missing":
        trace["reference_student_input_features"] = captured
    with pytest.raises(ValueError, match="actual captured"):
        original_actor_rows(trace, captured_required=True)


def test_original_teacher_and_off_reconstruct_without_claiming_student_capture():
    trace = measured_trace()
    assert (
        original_actor_rows(trace, captured_required=False)["observation"].tobytes()
        == trace_observations(trace)["observation"].tobytes()
    )
