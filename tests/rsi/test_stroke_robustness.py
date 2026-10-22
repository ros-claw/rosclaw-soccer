import copy

import numpy as np
import pytest

from rosclaw_soccer.rsi.stroke_robustness import paired_robustness_summary, robustness_courses


def contexts():
    return {
        i: dict(
            backoff_m=float(k),
            case=dict(
                ball_world_initial=[0.0, 0.0, 0.11],
                goal_world=[0.0, -6.0, 0.11],
                velocity_world_initial=[0.0, 0.17],
                bank="train",
                not_fresh=True,
            ),
        )
        for k, i in enumerate((4, 20, 21))
    }


def test_all48_public_courses_are_deterministic_bounded_and_do_not_mutate_source():
    c = contexts()
    original = copy.deepcopy(c)
    bank = robustness_courses(c)
    assert len(bank) == 48 and c == original and bank == robustness_courses(c)
    assert [r["course_index"] for r in bank] == list(range(48))
    assert len({(r["backoff_m"], tuple(r["case"]["ball_world_initial"])) for r in bank}) == 48
    for row in bank:
        p = np.array(row["case"]["ball_world_initial"])
        assert abs(p[:2]).max() <= 0.020001 and p[2] == 0.11
        assert np.linalg.norm(row["case"]["velocity_world_initial"]) < 0.6


def rows():
    return [
        dict(
            context=c,
            course_index=16 * k + i,
            arm=arm,
            assisted=arm == "teacher",
            PPO_actor_rows=0,
            healthy_quality=arm == "teacher" and i < 13,
            result=dict(body_failed=False),
        )
        for k, c in enumerate((4, 20, 21))
        for i in range(16)
        for arm in ("baseline", "teacher")
    ]


def test_qualification_requires_each_context_no_falls_and_positive_paired_gain():
    r = rows()
    s = paired_robustness_summary(r)
    assert s["teacher_successes"] == 39 and s["teacher_locally_qualified"]
    assert not s["student_learning_proven"] and not s["fresh_generalization_proven"]
    r[1]["healthy_quality"] = False
    assert not paired_robustness_summary(r)["teacher_locally_qualified"]
    r = rows()
    r[1]["result"]["body_failed"] = True
    assert not paired_robustness_summary(r)["teacher_locally_qualified"]


@pytest.mark.parametrize("fault", ["missing", "course", "context", "arm", "credit", "grade"])
def test_no_omission_or_relabelled_teacher_credit(fault):
    r = rows()
    if fault == "missing":
        r.pop()
    elif fault == "course":
        r[1]["course_index"] += 1
    elif fault == "context":
        r[1]["context"] = 21
    elif fault == "arm":
        r[1]["arm"] = "baseline"
    elif fault == "credit":
        r[1]["PPO_actor_rows"] = 1
    else:
        r[1]["healthy_quality"] = 1
    with pytest.raises(ValueError):
        paired_robustness_summary(r)


def test_entire_bank_rejected_if_any_source_would_leave_launch_domain():
    c = contexts()
    c[21]["case"]["velocity_world_initial"] = [0.59, 0.15]
    with pytest.raises(ValueError):
        robustness_courses(c)
