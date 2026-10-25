import json

import pytest

from rosclaw_soccer.providers.g1.velocity_policy import file_hash
from rosclaw_soccer.rsi.first_touch_mechanism_expansion import validate_public_course
from rosclaw_soccer.rsi.walk_kick_hybrid_teacher import (
    compose_teacher,
    expanded_validation_courses,
    select_teacher,
)
from rosclaw_soccer.rsi.walk_kick_teacher_plan import teacher_courses
from rosclaw_soccer.training.receiving_course_preflight import preflight_receiving_courses


def report(tmp_path, fast):
    rows, pins = [], {}
    for i, c in enumerate(teacher_courses()):
        p = tmp_path / str(fast) / f"course-{i:02d}" / "candidate-00.json"
        p.parent.mkdir(parents=True)
        outcome = dict(task_pass=False)
        p.write_text(
            json.dumps(
                dict(
                    course=c.course,
                    outcome=outcome,
                    parameters=dict(
                        speed_mps=1.5 if fast else 0.8,
                        entry_gap_m=0.9,
                        reference_frame=220,
                        root_lateral_offset_m=0.0,
                    ),
                )
            )
        )
        pins[str(p)] = file_hash(p)
        rows.append(dict(course_index=i, selected="candidate-00", selected_outcome=outcome))
    return dict(
        status="PUBLIC16_CEM_ALL304_RAW_LABELS_AND_REEXECUTION_AUTHENTICATED_NOT_PROMOTED",
        experimental_fast_approach=fast,
        input_file_hashes=pins,
        results=rows,
        course_bank_hash=preflight_receiving_courses(teacher_courses()),
        promotion_authorized=False,
    )


def test_frozen_initial_state_routing_has_no_runtime_outcome_or_course_id_input(tmp_path):
    table = compose_teacher(report(tmp_path, False), report(tmp_path, True))
    for c in teacher_courses():
        selected = select_teacher(table, c.course)
        expected_fast = validate_public_course(c.course) == 3
        assert selected["experimental_fast"] is expected_fast
        assert selected["parameters"]["speed_mps"] == (1.5 if expected_fast else 0.8)
    assert not table["teacher_qualification_proven"]


def test_validation_all40_distinct_full_class_preflight_before_any_execution():
    courses = expanded_validation_courses()
    assert len(courses) == len({c.course for c in courses}) == 40
    counts = [0] * 4
    for c in courses:
        counts[validate_public_course(c.course)] += 1
    assert counts == [10] * 4
    assert preflight_receiving_courses(courses).startswith("sha256:")
    assert courses == expanded_validation_courses()


def test_altered_teacher_and_out_of_domain_requests_fail_closed(tmp_path):
    table = compose_teacher(report(tmp_path, False), report(tmp_path, True))
    with pytest.raises(ValueError):
        select_teacher(table, (2.5, 0.0, 0.4))
    table["prototypes"][0]["parameters"]["speed_mps"] = 100.0
    with pytest.raises(ValueError):
        select_teacher(table, (2.25, -0.08, -0.4))
