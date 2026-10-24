import copy

import pytest

from rosclaw_soccer.rsi.pass_teacher_transfer_plan import (
    EXPANDED_STATES,
    PILOT_STATES,
    transfer_plans,
)


def course(state, task):
    return dict(
        state_id=state,
        task=task,
        context=20,
        assisted=True,
        side=0.55 if state % 2 else -0.55,
        ball_planar_offset_world=[state / 100.0, 0.0],
        target=[1.0, task / 10.0, 0.11],
        latent=[[0.0] * 6 for _ in range(2)],
        legacy_field="unchanged",
    )


def setup():
    courses = [course(s, t) for s in EXPANDED_STATES for t in (0, 1)]
    donors = [
        dict(
            course=course(s, t),
            controlled=not (s in (23, 29) and t == 0),
            healthy_clean=True,
            evidence_hash="sha256:" + f"{s * 2 + t:064x}",
        )
        for s in PILOT_STATES
        for t in (0, 1)
    ]
    for d in donors:
        d["course"]["latent"][0][0] = d["course"]["state_id"] / 100.0
    return courses, donors


def test_all_twelve_plans_disjoint_nearest_and_explicitly_not_success():
    courses, donors = setup()
    original = copy.deepcopy((courses, donors))
    plans = transfer_plans(courses, donors)
    assert len(plans) == 12
    for plan in plans:
        assert plan["donor_state"] % 2 == plan["state_id"] % 2
        assert plan["transferred_skill_physically_qualified"] is False
        assert plan["course"]["legacy_field"] == "unchanged"
        assert plan["student_input"] is False
    # State21 is geometrically closest to20 but on the other receiving side.
    assert plans[0]["state_id"] == 21 and plans[0]["donor_state"] == 23
    assert plans[0]["donor_controlled"] is False  # Never relabel a safe failure.
    assert (courses, donors) == original
    plans[0]["course"]["latent"][0][0] = 0.99
    assert (courses, donors) == original


@pytest.mark.parametrize(
    "defect", ["missing", "duplicate", "fresh", "side", "unsafe", "nan", "latent", "boolean"]
)
def test_incomplete_or_out_of_scope_plans_rejected(defect):
    courses, donors = setup()
    if defect == "missing":
        donors.pop()
    elif defect == "duplicate":
        courses[-1] = courses[0]
    elif defect == "fresh":
        courses[0]["state_id"] = 300
    elif defect == "side":
        courses[0]["side"] = 0.7
    elif defect == "unsafe":
        donors[0]["healthy_clean"] = False
    elif defect == "nan":
        donors[0]["course"]["ball_planar_offset_world"][0] = float("nan")
    elif defect == "latent":
        donors[0]["course"]["latent"][0][0] = 1.01
    else:
        donors[0]["controlled"] = 1
    with pytest.raises(ValueError):
        transfer_plans(courses, donors)
