from __future__ import annotations

from rosclaw_soccer.skills.goalkeeper_v2 import benchmark


def test_coverage_curriculum_has_five_deadlines_and_regions() -> None:
    assert benchmark._DEADLINES_SEC == (1.0, 0.8, 0.6, 0.5, 0.4)
    assert {item[0] for item in benchmark._TARGETS} == {
        "upper_left",
        "upper_right",
        "lower_left",
        "lower_right",
        "center",
    }
