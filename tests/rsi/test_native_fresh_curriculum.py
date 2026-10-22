import copy
from collections import Counter

import numpy as np
import pytest

from rosclaw_soccer.rsi.native_fresh_curriculum import (
    native_fresh_courses,
    native_fresh_summary,
    physical_context_key,
)


def contexts():
    rows = []
    for i in range(20):
        source = f"soccer-standard-{i // 2 + 1:03d}_{'left' if (i // 2) % 2 else 'right'}.npz"
        case = dict(
            source=source,
            bank="train",
            not_fresh=True,
            case=i,
            profile="cached-standard",
            ball_mass_kg=0.43,
            ball_world_initial=[
                0.15 if "left" in source else -0.15,
                0.63 if i % 2 == 0 else 1.03,
                0.11,
            ],
            velocity_world_initial=[0.01, -0.2 if i % 2 == 0 else 0.2],
            goal_world=[0.0, 6.0, 0.11],
        )
        rows.append(dict(case=case, backoff_m=0.0, kind="old_first_skill"))
    rows.extend(
        [
            dict(case=copy.deepcopy(rows[4]["case"]), backoff_m=b, kind="consumed_run")
            for b in (1.6, 1.2)
        ]
    )
    return rows


def test_new_conditions_are_balanced_deterministic_and_physically_disjoint_not_renamed():
    source = contexts()
    before = copy.deepcopy(source)
    fresh = native_fresh_courses(source)
    assert source == before and fresh == native_fresh_courses(source)
    assert len(fresh) == 56
    assert Counter(c["condition"] for c in fresh[:40]) == dict(
        near_approaching=10, near_away=10, far_approaching=10, far_away=10
    )
    assert {physical_context_key(c["ctx"]) for c in fresh}.isdisjoint(
        {physical_context_key(c) for c in source}
    )
    for c in fresh:
        case = c["ctx"]["case"]
        assert case["not_fresh"] is False and case["bank"] != "train"
        assert case["ball_world_initial"][2] == 0.11 and case["ball_mass_kg"] == 0.43
        assert np.linalg.norm(case["velocity_world_initial"]) < 0.3
        assert (case["velocity_world_initial"][1] > 0) == c["condition"].endswith("away")
    renamed = copy.deepcopy(source[0])
    renamed["case"].update(bank="fresh", case=20000, not_fresh=False)
    assert physical_context_key(renamed) == physical_context_key(source[0])


def outcomes(courses):
    result = []
    for c in courses:
        for arm in ("baseline", "65023"):
            good = c["index"] < 40 or (arm == "65023" and c["index"] in (40, 42, 48, 50))
            result.append(
                dict(
                    course_index=c["index"],
                    arm=arm,
                    healthy_quality=good,
                    body_failed=False,
                    forbidden=False,
                )
            )
    return result


def test_preregistered_gain_is_separate_from_promotion_and_never_drops_failed_courses():
    courses = native_fresh_courses(contexts())
    rows = outcomes(courses)
    result = native_fresh_summary(rows, courses, [65023])
    assert result["reports"][0]["preregistered_local_Fresh_screen_pass"] is True
    assert result["reports"][0]["cohorts"]["run16"]["net"] == 4
    assert result["promotion_authorized"] is False and result["legacy52_proven"] is False
    with pytest.raises(ValueError):
        native_fresh_summary(rows[:-1], courses, [65023])


def test_equal_average_quality_cannot_hide_a_new_forbidden_contact():
    courses = native_fresh_courses(contexts())
    rows = outcomes(courses)
    for r in rows:
        if r["course_index"] == 0:
            r["healthy_quality"] = False
            if r["arm"] == "65023":
                r["forbidden"] = True
    report = native_fresh_summary(rows, courses, [65023])["reports"][0]
    assert report["cohorts"]["run16"]["net"] == 4
    assert report["critical_task_regressions"] == 1
    assert report["preregistered_local_Fresh_screen_pass"] is False


def test_repeated_reference_motion_is_not_counted_as_new_motion_families():
    source = contexts()
    for i in range(14, 20):
        source[i]["case"]["source"] = "soccer-standard-009_right.npz"
    courses = native_fresh_courses(source)
    report = native_fresh_summary(outcomes(courses), courses, [65023])
    assert len(courses) == 56 and report["known_motion_families_base"] == 8
