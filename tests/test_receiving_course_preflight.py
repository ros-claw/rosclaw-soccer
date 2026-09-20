from dataclasses import replace

import pytest

from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses
from rosclaw_soccer.training.role_receiving_courses import ReceivingCourse

BASE = ReceivingCourse("blue.defender", 123, 0.5, -0.12)


def test_bank_binding_is_deterministic_and_preserves_input():
    courses = (BASE, replace(BASE, seed=124, speed_mps=1.8))
    assert preflight_receiving_courses(courses) == preflight_receiving_courses(courses)
    assert preflight_receiving_courses(courses) != preflight_receiving_courses(courses[::-1])
    assert courses[0].speed_mps == 0.5


def test_lower_edge_perturbation_is_rejected_before_any_worker_is_needed():
    with pytest.raises(ValueError, match="launch"):
        preflight_receiving_courses((BASE, replace(BASE, seed=124, speed_mps=0.5 * 0.95)))


@pytest.mark.parametrize("speed", [True, float("nan"), float("inf"), 0.0, 3.01, "1"])
def test_invalid_speed(speed):
    with pytest.raises(ValueError):
        preflight_receiving_courses((replace(BASE, speed_mps=speed),))


@pytest.mark.parametrize("seed", [True, -1, 2**32, 1.5])
def test_invalid_seed(seed):
    with pytest.raises(ValueError):
        preflight_receiving_courses((replace(BASE, seed=seed),))


def test_no_duplicate_seed_identity_or_silent_course_clipping():
    with pytest.raises(ValueError, match="duplicate"):
        preflight_receiving_courses((BASE, replace(BASE, lateral_m=0.12)))
    with pytest.raises(ValueError):
        preflight_receiving_courses((replace(BASE, lateral_m=0.31),))
    with pytest.raises(ValueError):
        preflight_receiving_courses((replace(BASE, agent_id="missing"),))


@pytest.mark.parametrize("courses", [(), [BASE], (object(),), (BASE,) * 4097])
def test_invalid_bank(courses):
    with pytest.raises(ValueError):
        preflight_receiving_courses(courses)
