from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from rosclaw_soccer.training import dynamic_takeoff_exam
from rosclaw_soccer.training.dynamic_takeoff_exam import (
    DynamicTakeoffExamConfig,
    _longest_true_run,
    evaluate_dynamic_takeoff_save,
    expanded_dynamic_takeoff_config,
)


def _trajectory(*, airborne: bool = True) -> dict[str, np.ndarray]:
    time = np.arange(0.0, 1.02, 0.02, dtype=np.float64)
    pelvis = np.zeros((time.size, 7), dtype=np.float64)
    pelvis[:, 2] = 0.77
    pelvis[:, 3] = 1.0
    velocity = np.zeros((time.size, 6), dtype=np.float64)
    contact = np.ones((time.size, 2), dtype=np.bool_)
    if airborne:
        flight = (time >= 0.30) & (time <= 0.40)
        contact[flight] = False
        pelvis[flight, 2] += np.linspace(0.0, 0.035, int(np.count_nonzero(flight)))
        velocity[(time >= 0.28) & (time <= 0.32), 2] = 0.24
        velocity[np.searchsorted(time, 0.42), 2] = -0.35
        velocity[np.searchsorted(time, 0.42), 3] = 0.40
    return {
        "time": time,
        "goalkeeper_pelvis_pose": pelvis,
        "goalkeeper_root_velocity": velocity,
        "goalkeeper_foot_contact": contact,
    }


def test_takeoff_exam_requires_real_no_foot_contact_interval(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        dynamic_takeoff_exam,
        "evaluate_dynamic_aerial_lunge_save",
        lambda **_: {"passed": True, "gates": {"post_save_recovered": True}},
    )
    result = SimpleNamespace(shot_contact_time_sec=0.20, goalkeeper_glove_contact_time_sec=0.36)
    report = evaluate_dynamic_takeoff_save(  # type: ignore[arg-type]
        result=result,
        trajectory=_trajectory(),
        config=DynamicTakeoffExamConfig(),
    )
    assert report["passed"] is True
    assert report["gates"]["glove_contact_during_flight"] is True
    assert report["metrics"]["airborne_duration_sec"] >= 0.10

    rejected = evaluate_dynamic_takeoff_save(  # type: ignore[arg-type]
        result=result,
        trajectory=_trajectory(airborne=False),
    )
    assert rejected["passed"] is False
    assert rejected["reason"] == "no ground-contact-confirmed flight interval"


def test_longest_true_run_is_inclusive_and_deterministic() -> None:
    assert _longest_true_run(np.asarray((False, True, True, False, True))) == (1, 2)
    assert _longest_true_run(np.zeros(4, dtype=np.bool_)) == (None, None)


def test_expanded_takeoff_contract_requires_longer_flight_and_capture() -> None:
    config = expanded_dynamic_takeoff_config()
    assert config.minimum_airborne_duration_sec == pytest.approx(0.15)
    assert config.minimum_flight_pelvis_rise_m == pytest.approx(0.035)
    assert config.lunge_config.lower_body_scale == pytest.approx(0.78)
    assert config.lunge_config.landing_capture_enabled is True
    assert config.lunge_config.landing_capture_sec == pytest.approx(0.20)
