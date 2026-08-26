from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.training.goalkeeper_dive_memory_exam import (
    GoalkeeperDiveMemoryExamConfig,
    _sample_cases,
)


def test_dive_memory_exam_contains_only_balanced_far_strata() -> None:
    config = GoalkeeperDiveMemoryExamConfig(shots_per_stratum=8)
    cases = _sample_cases(config)
    assert len(cases) == 24
    assert {case["stratum"] for case in cases} == {
        "far_corner_low",
        "far_corner_mid",
        "far_corner_high",
    }
    assert all(abs(float(case["target_y_m"])) >= 0.74 for case in cases)
    assert all(
        sum(case["stratum"] == name for case in cases) == 8
        for name in set(case["stratum"] for case in cases)
    )


def test_dive_memory_exam_is_sim_only_and_requires_real_physics_timestep() -> None:
    config = GoalkeeperDiveMemoryExamConfig(shots_per_stratum=8)
    with pytest.raises(ValueError, match="SIM_ONLY"):
        replace(config, hardware_authorized=True)
    with pytest.raises(ValueError, match="0.002 s"):
        replace(config, physics_substeps=8)
