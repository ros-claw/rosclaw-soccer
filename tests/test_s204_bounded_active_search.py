from __future__ import annotations

from dataclasses import replace

import pytest

from rosclaw_soccer.growth.bounded_active_search import (
    BoundedActiveSearchPlan,
    BoundedSearchDimension,
)


def _plan() -> BoundedActiveSearchPlan:
    return BoundedActiveSearchPlan(
        dimensions=(
            BoundedSearchDimension("gain", -1.0, 1.0),
            BoundedSearchDimension("phase", 0.0, 4.0),
        ),
        global_candidate_count=8,
        local_candidate_count=8,
    )


def test_halton_candidates_are_deterministic_bounded_and_content_bound() -> None:
    plan = _plan()

    first = plan.global_candidates()
    second = plan.global_candidates()

    assert first == second
    assert len({candidate.candidate_hash for candidate in first}) == 8
    assert all(-1.0 <= candidate.values[0] <= 1.0 for candidate in first)
    assert all(0.0 <= candidate.values[1] <= 4.0 for candidate in first)
    assert all(candidate.plan_hash == plan.plan_hash for candidate in first)


def test_local_search_keeps_warm_start_and_shrinks_trust_region() -> None:
    plan = _plan()

    candidates = plan.local_candidates((0.5, 2.0))

    assert candidates[0].stage == "WARM_START"
    assert candidates[0].values == (0.5, 2.0)
    assert len(candidates) == 9
    assert all(0.25 <= candidate.values[0] <= 0.75 for candidate in candidates)
    assert all(1.5 <= candidate.values[1] <= 2.5 for candidate in candidates)


def test_invalid_or_hardware_authorized_plans_fail_closed() -> None:
    with pytest.raises(ValueError, match="dimension is invalid"):
        BoundedSearchDimension("Bad Name", 0.0, 1.0)
    with pytest.raises(ValueError, match="plan is invalid"):
        replace(_plan(), hardware_authorized=True)
    with pytest.raises(ValueError, match="center exceeds"):
        _plan().local_candidates((2.0, 2.0))
