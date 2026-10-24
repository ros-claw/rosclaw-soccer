"""Synthetic curriculum assertions; not actual skill evidence."""

from copy import deepcopy

import pytest

from rosclaw_soccer.rsi.pass_refinement_curriculum import refinement_curriculum
from rosclaw_soccer.rsi.pass_teacher_coverage import teacher_coverage
from rosclaw_soccer.rsi.pass_teacher_transfer_plan import EXPANDED_STATES


def outcome(passed):
    return dict(
        whole_body_healthy=True,
        passer_full100NNrecovery=True,
        passer_clean_foot=True,
        B_excluded_contact=False,
        B_legal_functional_receive_before_deadline=True,
        combined_precision_and_functional_receive=False,
        incoming=dict(
            A_first_foot_strike_substep=900, B_first_ball_contact_substep=1100 if passed else None
        ),
    )


def fixture():
    native = []
    for i in range(80):
        s, t = divmod(i, 2)
        c = dict(
            state_id=s,
            task=t,
            context=(1, 4, 20, 21)[s // 10],
            assisted=False,
            side=0.55 if s % 2 else -0.55,
            target=[0.0, float(t), 0.11],
            center=103,
            foot="left",
            ball_planar_offset_world=[s / 100.0, 0.0],
        )
        native.append(dict(index=i, course=c, outcome=outcome(i < 35 or 40 <= i < 46)))
    baseline, selected, transfers = [], [], []
    for j, row in enumerate(native[40:60]):
        s, t = row["course"]["state_id"], row["course"]["task"]
        baseline.append(dict(state_id=s, task=t, outcome=deepcopy(row["outcome"])))
        chosen = dict(
            state_id=s,
            task=t,
            outcome=outcome(j < 9),
            transfer_selected=True,
            course=row["course"] | dict(assisted=True),
        )
        selected.append(chosen)
        if s in EXPANDED_STATES:
            transfers.append(deepcopy(chosen))
    coverage = dict(
        baseline=baseline,
        selected_sources=selected,
        transfers=transfers,
        coverage=teacher_coverage(baseline, selected, transfers),
    )
    old = [
        dict(index=i, role="retained_neural17" if i < 17 else "negative", task_success=i < 17)
        for i in range(274)
    ]
    return coverage, native, old


def test_preserves_all_current_skills_and_teaches_only_actual_rescues():
    plan = refinement_curriculum(*fixture())
    assert len(plan["original_motion_anchor_indices"]) == 17
    assert len(plan["current_pass_anchor_indices"]) == 41
    assert plan["correction_teacher_count"] == 3
    assert plan["fit_histories"] == 61
    assert [(r["state_id"], r["task"]) for r in plan["correction_sources"]] == [
        (23, 0),
        (23, 1),
        (24, 0),
    ]
    assert len(plan["current_failed_indices"]) == 39
    assert plan["success_targets_replaced_by_teacher"] is False
    assert plan["student_optimizer_updates"] == 0


@pytest.mark.parametrize(
    "defect",
    ["unqualified", "missing80", "duplicate", "lost41", "oldanchor", "changedgoal", "changedball"],
)
def test_broken_or_conflicting_curricula_rejected(defect):
    cov, native, old = fixture()
    if defect == "unqualified":
        cov["coverage"]["functional_correction_teacher_qualified"] = False
    elif defect == "missing80":
        native.pop()
    elif defect == "duplicate":
        native[-1] = native[0]
    elif defect == "lost41":
        native[0]["outcome"] = outcome(False)
    elif defect == "oldanchor":
        old[0]["task_success"] = False
    elif defect == "changedgoal":
        cov["selected_sources"][6]["course"]["target"] = [10.0, 10.0, 0.0]
    else:
        cov["selected_sources"][6]["course"]["ball_planar_offset_world"] = [10.0, 10.0]
    with pytest.raises(ValueError):
        refinement_curriculum(cov, native, old)
