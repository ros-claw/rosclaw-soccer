"""Fail-closed physical receiving-possession evidence tests."""

import numpy as np
import pytest

from rosclaw_soccer.training.receiving_possession import evaluate_receiving_possession


def _trace() -> dict[str, object]:
    n = 80
    ball_pose = np.zeros((n, 7), dtype=np.float64)
    ball_pose[:, 0] = 0.30
    pelvis_pose = np.zeros((n, 7), dtype=np.float64)
    ball_velocity = np.zeros((n, 6), dtype=np.float64)
    contact_agent = np.zeros(n, dtype=np.int64)
    contact_foot = np.zeros(n, dtype=np.int64)
    nonfoot_agent = np.zeros(n, dtype=np.int64)
    contact_agent[10] = 2
    contact_foot[10] = 1
    return {
        "ball_pose": ball_pose,
        "ball_velocity": ball_velocity,
        "pelvis_pose": pelvis_pose,
        "contact_agent_code": contact_agent,
        "contact_foot_code": contact_foot,
        "nonfoot_agent_code": nonfoot_agent,
        "agent_code": 2,
        "body_safe": True,
    }


def test_sustained_clean_foot_control() -> None:
    result = evaluate_receiving_possession(**_trace())  # type: ignore[arg-type]
    assert result["qualified"] is True
    assert result["retained_start_frame"] == 10
    assert result["best_contiguous_frames"] == 10


def test_single_contact_with_fast_ball_is_not_possession() -> None:
    trace = _trace()
    trace["ball_velocity"][:, 0] = 0.5  # type: ignore[index]
    result = evaluate_receiving_possession(**trace)  # type: ignore[arg-type]
    assert result["qualified"] is False
    assert result["best_contiguous_frames"] == 0


def test_nonfoot_contact_is_terminal() -> None:
    trace = _trace()
    trace["nonfoot_agent_code"][20] = 2  # type: ignore[index]
    result = evaluate_receiving_possession(**trace)  # type: ignore[arg-type]
    assert result["reason"] == "OWN_NONFOOT_CONTACT"


def test_foreign_touch_breaks_contiguous_window() -> None:
    trace = _trace()
    trace["contact_agent_code"][18] = 3  # type: ignore[index]
    trace["contact_foot_code"][18] = 1  # type: ignore[index]
    trace["ball_velocity"][28:, 0] = 0.5  # type: ignore[index]
    result = evaluate_receiving_possession(**trace)  # type: ignore[arg-type]
    assert result["qualified"] is False
    assert result["best_contiguous_frames"] == 9


def test_unsafe_body_rejected() -> None:
    trace = _trace()
    trace["body_safe"] = False
    assert evaluate_receiving_possession(**trace)["reason"] == "BODY_UNSAFE"  # type: ignore[arg-type]


def test_malformed_trace_rejected() -> None:
    trace = _trace()
    trace["ball_pose"][15, 0] = np.nan  # type: ignore[index]
    with pytest.raises(ValueError, match="finite aligned"):
        evaluate_receiving_possession(**trace)  # type: ignore[arg-type]
