from dataclasses import replace

import pytest

from rosclaw_soccer.training.native_pass_courses import NativePassCourse
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def course(n=1):
    return NativePassCourse(n, 4.0, 0.55, "sha256:" + f"{n:064x}")


def test_actual_pass_domain_is_explicit_and_arms_do_not_duplicate_courses():
    assert preflight_receiving_courses((course(), course(2))).startswith("sha256:")
    with pytest.raises(ValueError, match="duplicate"):
        preflight_receiving_courses((course(), course()))
    with pytest.raises(ValueError, match="duplicate"):
        preflight_receiving_courses(
            (
                course(),
                replace(course(2), initial_world_state_hash=course().initial_world_state_hash),
            )
        )


@pytest.mark.parametrize(
    "field,value",
    [
        ("receiver_forward_m", 1.99),
        ("receiver_lateral_m", 1.51),
        ("receiver_lateral_m", float("nan")),
        ("context_index", True),
        ("initial_world_state_hash", "bad"),
    ],
)
def test_entire_actual_pass_bank_rejected_without_clipping(field, value):
    with pytest.raises(ValueError):
        preflight_receiving_courses((course(), replace(course(2), **{field: value})))
