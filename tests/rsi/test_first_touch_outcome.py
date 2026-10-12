from dataclasses import replace

import numpy as np
import pytest

from rosclaw_soccer.rsi.first_touch_outcome import (
    label_first_touch,
    outcome_contract,
    remaining_task_time,
)


def trace(frame: int | None = 239) -> dict[str, np.ndarray]:
    force = np.zeros((300, 1, 6), dtype=np.float64)
    ball = np.zeros((300, 1, 3), dtype=np.float64)
    if frame is not None:
        force[frame, 0, 1] = 2.0
        ball[frame + 1 :, 0, 0] = np.arange(1, 300 - frame) / 50
    return dict(
        force_n=force,
        ball_position_after_step_m=ball,
        pelvis_z_per_substep_m=np.full((300, 1, 10), 0.7),
    )


@pytest.mark.parametrize("frame", [0, 85, 239])
def test_last_observable_contact(frame: int) -> None:
    result = label_first_touch(trace(frame))
    assert result.quality_observed and result.task_pass
    assert result.forward_60_m == pytest.approx(1.2)
    assert result.training_reward == pytest.approx(3.2)
    assert result.event_status == "CONTACT_OBSERVED"


@pytest.mark.parametrize("frame", [240, 260, 299])
def test_censored_is_valid_data_not_zero_quality(frame: int) -> None:
    result = label_first_touch(trace(frame))
    assert result.first_foot_contact_frame == frame
    assert result.event_status == "POST_CONTACT_CENSORED"
    assert result.measurement_status == "RIGHT_CENSORED"
    assert result.forward_60_m is result.lateral_60_m is None
    assert not result.quality_observed and not result.task_pass
    assert result.training_reward == -1
    assert result.to_dict()["terminated"] is True
    assert result.to_dict()["truncated"] is False


def test_no_contact_and_safety_veto() -> None:
    assert label_first_touch(trace(None)).training_reward == -3
    for frame in (239, 260, None):
        raw = trace(frame)
        raw["pelvis_z_per_substep_m"][299, 0, 9] = 0.64
        result = label_first_touch(raw)
        assert result.event_status == "UNSAFE_CONTACT" and not result.task_pass
        assert result.training_reward == -10
        assert result.quality_observed is (frame == 239)
    raw = trace()
    raw["force_n"][280, 0, 5] = 2
    result = label_first_touch(raw)
    assert result.event_status == "UNSAFE_CONTACT" and result.forward_60_m == 1.2


@pytest.mark.parametrize(
    "field", ["force_n", "pelvis_z_per_substep_m", "ball_position_after_step_m"]
)
@pytest.mark.parametrize("value", [float("nan"), float("inf")])
def test_nonfinite_is_contract_violation(field: str, value: float) -> None:
    raw = trace()
    raw[field].flat[0] = value
    with pytest.raises(ValueError):
        label_first_touch(raw)


def test_incomplete_or_invalid_trace_is_not_censoring() -> None:
    raw = trace()
    raw["force_n"] = raw["force_n"][:-1]
    with pytest.raises(ValueError):
        label_first_touch(raw)
    raw = trace()
    raw["force_n"][0, 0, 0] = -1
    with pytest.raises(ValueError):
        label_first_touch(raw)


def test_contract_is_fresh_and_clock_is_not_gait_phase() -> None:
    contract = outcome_contract()
    contract["censored_contact_reward"] = 999
    assert outcome_contract()["censored_contact_reward"] == -1
    assert remaining_task_time(0) == 1 and remaining_task_time(299) == 0
    assert remaining_task_time(40) != remaining_task_time(0)
    for invalid in (True, -1, 300, 1.0):
        with pytest.raises(ValueError):
            remaining_task_time(invalid)


def test_outcome_is_frozen_and_labels_do_not_mutate_trace() -> None:
    raw = trace(260)
    original = {k: v.copy() for k, v in raw.items()}
    result = label_first_touch(raw)
    assert all(np.array_equal(v, original[k]) for k, v in raw.items())
    assert replace(result, failure_reason="other") != result
